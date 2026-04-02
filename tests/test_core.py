from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from gpu_alloc.core import (
    AllocationRequest,
    AllocationUnavailable,
    AllocatorConfig,
    GPUAllocator,
    GPUInfo,
    LeaseStore,
    ProbeError,
    NvidiaSmiProbe,
    parse_allocate_spec,
    select_gpus,
)


class ParseAllocateSpecTests(unittest.TestCase):
    def test_parses_gib_values(self) -> None:
        request = parse_allocate_spec("allocate=3_gpu,40G")
        self.assertEqual(request, AllocationRequest(gpu_count=3, memory_mib=40960, raw="allocate=3_gpu,40G"))

    def test_parses_mib_values(self) -> None:
        request = parse_allocate_spec("1_gpu,2048MiB")
        self.assertEqual(request.memory_mib, 2048)

    def test_rejects_invalid_values(self) -> None:
        with self.assertRaisesRegex(Exception, "Invalid allocation spec"):
            parse_allocate_spec("three_gpu,40G")


class NvidiaSmiProbeTests(unittest.TestCase):
    def test_parses_csv_output(self) -> None:
        probe = NvidiaSmiProbe("/usr/bin/nvidia-smi")
        completed = mock.Mock(returncode=0, stdout="0, H100, 81920, 70000, 4\n", stderr="")

        with mock.patch("gpu_alloc.core.subprocess.run", return_value=completed):
            gpus = probe.query()

        self.assertEqual(len(gpus), 1)
        self.assertEqual(gpus[0].free_mib, 70000)
        self.assertEqual(gpus[0].utilization_gpu, 4)

    def test_raises_probe_error_on_nonzero_exit(self) -> None:
        probe = NvidiaSmiProbe("/usr/bin/nvidia-smi")
        completed = mock.Mock(returncode=9, stdout="", stderr="driver not ready")

        with mock.patch("gpu_alloc.core.subprocess.run", return_value=completed):
            with self.assertRaises(ProbeError):
                probe.query()


class SelectGpuTests(unittest.TestCase):
    def test_prefers_low_utilization_then_more_free_memory(self) -> None:
        request = AllocationRequest(gpu_count=2, memory_mib=40960, raw="2_gpu,40G")
        gpus = [
            GPUInfo(index=0, name="A", total_mib=81920, free_mib=62000, utilization_gpu=20),
            GPUInfo(index=1, name="B", total_mib=81920, free_mib=70000, utilization_gpu=10),
            GPUInfo(index=2, name="C", total_mib=81920, free_mib=65000, utilization_gpu=10),
        ]

        selection = select_gpus(
            gpus,
            request,
            reserved_ids=[2],
            memory_margin=1.5,
            utilization_threshold=30,
        )

        self.assertIsNotNone(selection)
        self.assertEqual(selection.gpu_ids, (1, 0))


class LeaseStoreTests(unittest.TestCase):
    def test_removes_stale_lease_for_dead_pid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LeaseStore(temp_dir)
            with store.locked():
                lease = store.create_lease([0], ["python"], lease_seconds=10, pid=999999)
                leases = store.load_active_leases()
            self.assertNotIn(lease.lease_id, leases)


class GPUAllocatorTests(unittest.TestCase):
    def test_waits_until_gpu_is_eligible(self) -> None:
        request = AllocationRequest(gpu_count=1, memory_mib=40960, raw="1_gpu,40G")
        config = AllocatorConfig(memory_margin=1.5, utilization_threshold=30, poll_interval=0.01, lease_seconds=60)

        class FakeProbe:
            def __init__(self) -> None:
                self.calls = 0

            def query(self):
                self.calls += 1
                if self.calls == 1:
                    return [GPUInfo(index=0, name="A", total_mib=81920, free_mib=50000, utilization_gpu=5)]
                return [GPUInfo(index=0, name="A", total_mib=81920, free_mib=70000, utilization_gpu=5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            allocator = GPUAllocator(FakeProbe(), LeaseStore(temp_dir), config)
            decision = allocator.allocate(request, command=["python"], wait=True, reserve=False)

        self.assertEqual(decision.selection.gpu_ids, (0,))

    def test_no_wait_returns_allocation_unavailable(self) -> None:
        request = AllocationRequest(gpu_count=1, memory_mib=40960, raw="1_gpu,40G")
        config = AllocatorConfig(memory_margin=1.5, utilization_threshold=30, poll_interval=0.01, lease_seconds=60)

        class FakeProbe:
            def query(self):
                return [GPUInfo(index=0, name="A", total_mib=81920, free_mib=50000, utilization_gpu=5)]

        with tempfile.TemporaryDirectory() as temp_dir:
            allocator = GPUAllocator(FakeProbe(), LeaseStore(temp_dir), config)
            with self.assertRaises(AllocationUnavailable):
                allocator.allocate(request, command=["python"], wait=False, reserve=False)
