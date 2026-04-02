from __future__ import annotations

import contextlib
import csv
import errno
import fcntl
import json
import math
import os
import re
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Iterable, Sequence

REQUEST_PATTERN = re.compile(
    r"^\s*(?:allocate\s*=\s*)?(?P<count>\d+)\s*_gpu\s*,\s*"
    r"(?P<memory>\d+(?:\.\d+)?)\s*(?P<unit>[gGmM](?:i?[bB])?)?\s*$"
)
UNIT_FACTORS = {
    None: 1024,
    "m": 1,
    "mb": 1,
    "mib": 1,
    "g": 1024,
    "gb": 1024,
    "gib": 1024,
}


class AllocationError(RuntimeError):
    """Base error for allocation failures."""


class ProbeError(AllocationError):
    """Raised when GPU state cannot be read."""


class AllocationUnavailable(AllocationError):
    """Raised when no matching GPUs are available."""


@dataclass(frozen=True)
class AllocationRequest:
    gpu_count: int
    memory_mib: int
    raw: str


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_mib: int
    free_mib: int
    utilization_gpu: int


@dataclass(frozen=True)
class SelectionResult:
    gpu_ids: tuple[int, ...]
    gpus: tuple[GPUInfo, ...]
    required_free_mib: int


@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    gpu_ids: tuple[int, ...]
    pid: int
    command: tuple[str, ...]
    created_at: float
    updated_at: float
    expires_at: float

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "LeaseRecord":
        return cls(
            lease_id=str(payload["lease_id"]),
            gpu_ids=tuple(int(item) for item in payload["gpu_ids"]),
            pid=int(payload["pid"]),
            command=tuple(str(item) for item in payload.get("command", [])),
            created_at=float(payload["created_at"]),
            updated_at=float(payload["updated_at"]),
            expires_at=float(payload["expires_at"]),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "lease_id": self.lease_id,
            "gpu_ids": list(self.gpu_ids),
            "pid": self.pid,
            "command": list(self.command),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True)
class AllocatorConfig:
    memory_margin: float = 1.5
    utilization_threshold: int = 30
    poll_interval: float = 10.0
    lease_seconds: float = 120.0


@dataclass(frozen=True)
class AllocationDecision:
    selection: SelectionResult
    lease: LeaseRecord | None


def parse_allocate_spec(spec: str) -> AllocationRequest:
    match = REQUEST_PATTERN.match(spec)
    if not match:
        raise AllocationError(
            f"Invalid allocation spec {spec!r}. Expected COUNT_gpu,MEMORY, for example 3_gpu,40G."
        )

    gpu_count = int(match.group("count"))
    if gpu_count <= 0:
        raise AllocationError("GPU count must be greater than zero.")

    unit = match.group("unit")
    normalized_unit = unit.lower() if unit else None
    unit_factor = UNIT_FACTORS.get(normalized_unit)
    if unit_factor is None:
        raise AllocationError(f"Unsupported memory unit {unit!r}.")

    try:
        memory_value = Decimal(match.group("memory"))
    except InvalidOperation as exc:
        raise AllocationError(f"Invalid memory value in allocation spec {spec!r}.") from exc

    if memory_value <= 0:
        raise AllocationError("Requested memory must be greater than zero.")

    memory_mib = int((memory_value * unit_factor).to_integral_value(rounding=ROUND_HALF_UP))
    return AllocationRequest(gpu_count=gpu_count, memory_mib=memory_mib, raw=spec.strip())


class NvidiaSmiProbe:
    QUERY_FIELDS = "index,name,memory.total,memory.free,utilization.gpu"

    def __init__(self, executable: str = "nvidia-smi", timeout_seconds: float = 10.0):
        self.executable = executable
        self.timeout_seconds = timeout_seconds

    def query(self) -> list[GPUInfo]:
        command = [
            self.executable,
            f"--query-gpu={self.QUERY_FIELDS}",
            "--format=csv,noheader,nounits",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                text=True,
                timeout=self.timeout_seconds,
            )
        except FileNotFoundError as exc:
            raise ProbeError(f"Unable to execute {self.executable!r}.") from exc
        except subprocess.TimeoutExpired as exc:
            raise ProbeError(f"{self.executable!r} timed out after {self.timeout_seconds} seconds.") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()
            raise ProbeError(detail or f"{self.executable!r} exited with status {completed.returncode}.")

        stdout = completed.stdout.strip()
        if not stdout:
            raise ProbeError(f"{self.executable!r} returned no GPU records.")

        rows = csv.reader(stdout.splitlines())
        gpus: list[GPUInfo] = []
        for row in rows:
            if len(row) != 5:
                raise ProbeError(f"Unexpected nvidia-smi output row: {row!r}")
            gpus.append(
                GPUInfo(
                    index=self._parse_numeric_field(row[0], "index"),
                    name=row[1].strip(),
                    total_mib=self._parse_numeric_field(row[2], "memory.total"),
                    free_mib=self._parse_numeric_field(row[3], "memory.free"),
                    utilization_gpu=self._parse_numeric_field(row[4], "utilization.gpu", fallback=100),
                )
            )
        return gpus

    @staticmethod
    def _parse_numeric_field(raw: str, field_name: str, fallback: int | None = None) -> int:
        value = raw.strip()
        if value in {"N/A", "[Not Supported]"}:
            if fallback is not None:
                return fallback
            raise ProbeError(f"Field {field_name} is unavailable in nvidia-smi output.")
        try:
            return int(value)
        except ValueError as exc:
            raise ProbeError(f"Invalid integer for {field_name}: {value!r}") from exc


def select_gpus(
    gpus: Iterable[GPUInfo],
    request: AllocationRequest,
    *,
    reserved_ids: Iterable[int] = (),
    memory_margin: float,
    utilization_threshold: int,
) -> SelectionResult | None:
    reserved = set(reserved_ids)
    required_free_mib = math.ceil(request.memory_mib * memory_margin)
    eligible = [
        gpu
        for gpu in gpus
        if gpu.index not in reserved
        and gpu.free_mib >= required_free_mib
        and gpu.utilization_gpu <= utilization_threshold
    ]
    eligible.sort(key=lambda gpu: (gpu.utilization_gpu, -gpu.free_mib, gpu.index))
    if len(eligible) < request.gpu_count:
        return None

    chosen = tuple(eligible[: request.gpu_count])
    return SelectionResult(
        gpu_ids=tuple(gpu.index for gpu in chosen),
        gpus=chosen,
        required_free_mib=required_free_mib,
    )


class LeaseStore:
    def __init__(self, directory: str | os.PathLike[str]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / "leases.json"

    @contextlib.contextmanager
    def locked(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.state_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def load_active_leases(self, now: float | None = None) -> dict[str, LeaseRecord]:
        lease_records = self._read_records()
        cleaned: dict[str, LeaseRecord] = {}
        changed = False
        check_time = time.time() if now is None else now

        for record in lease_records.values():
            if self._is_stale(record, now=check_time):
                changed = True
                continue
            cleaned[record.lease_id] = record

        if changed:
            self._write_records(cleaned)
        return cleaned

    def create_lease(
        self,
        gpu_ids: Sequence[int],
        command: Sequence[str],
        *,
        lease_seconds: float,
        pid: int | None = None,
        now: float | None = None,
    ) -> LeaseRecord:
        check_time = time.time() if now is None else now
        owner_pid = os.getpid() if pid is None else pid
        lease = LeaseRecord(
            lease_id=uuid.uuid4().hex,
            gpu_ids=tuple(int(item) for item in gpu_ids),
            pid=owner_pid,
            command=tuple(command),
            created_at=check_time,
            updated_at=check_time,
            expires_at=check_time + lease_seconds,
        )
        records = self.load_active_leases(now=check_time)
        records[lease.lease_id] = lease
        self._write_records(records)
        return lease

    def renew_lease(self, lease_id: str, *, lease_seconds: float, now: float | None = None) -> LeaseRecord:
        check_time = time.time() if now is None else now
        records = self.load_active_leases(now=check_time)
        record = records.get(lease_id)
        if record is None:
            raise AllocationError(f"Lease {lease_id} no longer exists.")
        updated = LeaseRecord(
            lease_id=record.lease_id,
            gpu_ids=record.gpu_ids,
            pid=record.pid,
            command=record.command,
            created_at=record.created_at,
            updated_at=check_time,
            expires_at=check_time + lease_seconds,
        )
        records[lease_id] = updated
        self._write_records(records)
        return updated

    def release(self, lease_id: str) -> None:
        records = self.load_active_leases()
        if lease_id not in records:
            return
        del records[lease_id]
        self._write_records(records)

    def _read_records(self) -> dict[str, LeaseRecord]:
        if not self.state_path.exists() or self.state_path.stat().st_size == 0:
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AllocationError(f"Invalid lease state file: {self.state_path}") from exc
        leases = payload.get("leases", [])
        return {str(item["lease_id"]): LeaseRecord.from_dict(item) for item in leases}

    def _write_records(self, records: dict[str, LeaseRecord]) -> None:
        payload = {
            "leases": [
                records[lease_id].to_dict()
                for lease_id in sorted(records)
            ]
        }
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.directory,
            delete=False,
        ) as temp_file:
            json.dump(payload, temp_file, indent=2, sort_keys=True)
            temp_file.write("\n")
            temp_path = Path(temp_file.name)
        temp_path.replace(self.state_path)

    @staticmethod
    def _is_stale(record: LeaseRecord, *, now: float) -> bool:
        if record.expires_at <= now:
            return True
        return not _pid_exists(record.pid)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        if exc.errno == errno.ESRCH:
            return False
        return True
    return True


class GPUAllocator:
    def __init__(self, probe: NvidiaSmiProbe, lease_store: LeaseStore, config: AllocatorConfig):
        self.probe = probe
        self.lease_store = lease_store
        self.config = config

    def allocate(
        self,
        request: AllocationRequest,
        *,
        command: Sequence[str],
        wait: bool,
        reserve: bool,
    ) -> AllocationDecision:
        while True:
            with self.lease_store.locked():
                active_leases = self.lease_store.load_active_leases()
                reserved_ids = sorted(
                    {
                        gpu_id
                        for lease in active_leases.values()
                        for gpu_id in lease.gpu_ids
                    }
                )
                gpus = self.probe.query()
                if len(gpus) < request.gpu_count:
                    raise AllocationUnavailable(
                        f"Requested {request.gpu_count} GPUs, but only {len(gpus)} GPUs are visible."
                    )

                selection = select_gpus(
                    gpus,
                    request,
                    reserved_ids=reserved_ids,
                    memory_margin=self.config.memory_margin,
                    utilization_threshold=self.config.utilization_threshold,
                )
                if selection is not None:
                    lease = None
                    if reserve:
                        lease = self.lease_store.create_lease(
                            selection.gpu_ids,
                            command,
                            lease_seconds=self.config.lease_seconds,
                        )
                    return AllocationDecision(selection=selection, lease=lease)

            if not wait:
                raise AllocationUnavailable(
                    self._build_unavailable_message(
                        request=request,
                        gpus=gpus,
                        reserved_ids=reserved_ids,
                    )
                )
            time.sleep(self.config.poll_interval)

    def _build_unavailable_message(
        self,
        *,
        request: AllocationRequest,
        gpus: Sequence[GPUInfo],
        reserved_ids: Sequence[int],
    ) -> str:
        required_free_mib = math.ceil(request.memory_mib * self.config.memory_margin)
        eligible = [
            gpu.index
            for gpu in gpus
            if gpu.index not in set(reserved_ids)
            and gpu.free_mib >= required_free_mib
            and gpu.utilization_gpu <= self.config.utilization_threshold
        ]
        return (
            f"No allocation available for {request.raw!r}: need {request.gpu_count} GPUs with at least "
            f"{required_free_mib} MiB free and utilization <= {self.config.utilization_threshold}. "
            f"Currently eligible: {eligible or 'none'}, reserved: {list(reserved_ids) or 'none'}."
        )


class LeaseHeartbeat:
    def __init__(self, lease_store: LeaseStore, lease: LeaseRecord, lease_seconds: float):
        self.lease_store = lease_store
        self.lease = lease
        self.lease_seconds = lease_seconds
        self.interval = max(1.0, lease_seconds / 3.0)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gpu-alloc-heartbeat", daemon=True)

    def __enter__(self) -> "LeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop_event.set()
        self._thread.join(timeout=self.interval + 1.0)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            with self.lease_store.locked():
                try:
                    self.lease = self.lease_store.renew_lease(
                        self.lease.lease_id,
                        lease_seconds=self.lease_seconds,
                    )
                except AllocationError:
                    return


def run_command_with_lease(
    command: Sequence[str],
    *,
    env: dict[str, str],
    lease_store: LeaseStore,
    lease: LeaseRecord,
    lease_seconds: float,
) -> int:
    child = subprocess.Popen(list(command), env=env)
    handled_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        handled_signals.append(signal.SIGHUP)
    previous_handlers: dict[int, object] = {}

    def _forward(signum, _frame) -> None:
        if child.poll() is None:
            child.send_signal(signum)

    for signum in handled_signals:
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, _forward)

    try:
        with LeaseHeartbeat(lease_store, lease, lease_seconds):
            return child.wait()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        with lease_store.locked():
            lease_store.release(lease.lease_id)
