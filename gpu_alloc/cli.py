from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from gpu_alloc.core import (
    AllocationError,
    AllocationUnavailable,
    AllocatorConfig,
    GPUAllocator,
    LeaseStore,
    NvidiaSmiProbe,
    parse_allocate_spec,
    run_command_with_lease,
)

DEFAULT_STATE_DIR = str(Path.cwd() / ".gpu-alloc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpu-alloc",
        description="Reserve GPUs with nvidia-smi, then run a child command with CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "allocate_spec",
        nargs="?",
        help="Allocation request such as 3_gpu,40G. This is the preferred form for direct CLI use.",
    )
    parser.add_argument(
        "--allocate",
        help="Allocation request such as 3_gpu,40G. Kept for compatibility with older invocations.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Fail immediately if no matching GPUs are currently available.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print CUDA_VISIBLE_DEVICES=... without launching a child command.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=_env_default_float("GPU_ALLOC_POLL_INTERVAL", 10.0),
        help="Seconds between retries while waiting for GPUs. Default: 10 or $GPU_ALLOC_POLL_INTERVAL if set.",
    )
    parser.add_argument(
        "--memory-margin",
        type=float,
        default=_env_default_float("GPU_ALLOC_MEMORY_MARGIN", 1.5),
        help="Safety factor applied to the requested memory. Default: 1.5 or $GPU_ALLOC_MEMORY_MARGIN if set.",
    )
    parser.add_argument(
        "--utilization-threshold",
        type=int,
        default=_env_default_int("GPU_ALLOC_UTILIZATION_THRESHOLD", 30),
        help="Maximum utilization.gpu value for an eligible device. Default: 30 or $GPU_ALLOC_UTILIZATION_THRESHOLD if set.",
    )
    parser.add_argument(
        "--lease-seconds",
        type=float,
        default=_env_default_float("GPU_ALLOC_LEASE_SECONDS", 120.0),
        help="Lease TTL in seconds. Default: 120 or $GPU_ALLOC_LEASE_SECONDS if set.",
    )
    parser.add_argument(
        "--state-dir",
        default=os.environ.get("GPU_ALLOC_STATE_DIR", DEFAULT_STATE_DIR),
        help=f"Directory for lock and lease files. Default: {DEFAULT_STATE_DIR}.",
    )
    parser.add_argument(
        "--nvidia-smi-path",
        default=os.environ.get("GPU_ALLOC_NVIDIA_SMI_PATH", "nvidia-smi"),
        help="Path to nvidia-smi. Default: nvidia-smi from PATH.",
    )
    return parser


def split_command(argv: Sequence[str] | None) -> tuple[list[str], list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--" not in raw_args:
        return raw_args, []
    split_index = raw_args.index("--")
    return raw_args[:split_index], raw_args[split_index + 1 :]


def normalize_command(command: Sequence[str]) -> list[str]:
    values = list(command)
    if values[:1] == ["--"]:
        values = values[1:]
    return values


def _env_default_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return fallback
    try:
        return float(value)
    except ValueError as exc:
        raise AllocationError(f"Invalid {name} value: {value!r}. Expected a number.") from exc


def _env_default_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return fallback
    try:
        return int(value)
    except ValueError as exc:
        raise AllocationError(f"Invalid {name} value: {value!r}. Expected an integer.") from exc


def resolve_allocate_value(
    positional_value: str | None,
    option_value: str | None,
    env: dict[str, str],
) -> str:
    if positional_value and option_value and positional_value != option_value:
        raise AllocationError(
            f"Conflicting allocation requests: positional {positional_value!r} and --allocate {option_value!r}."
        )
    value = positional_value or option_value or env.get("ALLOCATE")
    if not value:
        raise AllocationError("Missing allocation request. Pass ALLOCATE, --allocate, or a positional spec.")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    try:
        parser = build_parser()
        parser_args, command_args = split_command(argv)
        args = parser.parse_args(parser_args)
        command = normalize_command(command_args)

        if not args.print_only and not command:
            parser.error("a child command is required unless --print-only is set")

        allocate_value = resolve_allocate_value(args.allocate_spec, args.allocate, os.environ)
        request = parse_allocate_spec(allocate_value)
        config = AllocatorConfig(
            memory_margin=args.memory_margin,
            utilization_threshold=args.utilization_threshold,
            poll_interval=args.poll_interval,
            lease_seconds=args.lease_seconds,
        )
        probe = NvidiaSmiProbe(args.nvidia_smi_path)
        lease_store = LeaseStore(args.state_dir)
        allocator = GPUAllocator(probe, lease_store, config)
        decision = allocator.allocate(
            request,
            command=command,
            wait=not args.no_wait,
            reserve=not args.print_only,
        )
    except AllocationUnavailable as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except AllocationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    cuda_visible_devices = ",".join(str(item) for item in decision.selection.gpu_ids)
    if args.print_only:
        print(f"CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
        return 0

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return run_command_with_lease(
        command,
        env=env,
        lease_store=lease_store,
        lease=decision.lease,
        lease_seconds=args.lease_seconds,
        gpu_ids=decision.selection.gpu_ids,
        cuda_visible_devices=cuda_visible_devices,
    )
