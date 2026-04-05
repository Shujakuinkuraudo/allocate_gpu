from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from unittest import mock

from gpu_alloc.core import (
    AllocationRequest,
    AllocationUnavailable,
    AllocatorConfig,
    EmailNotificationConfig,
    GPUAllocator,
    GPUInfo,
    LeaseStore,
    ProbeError,
    NvidiaSmiProbe,
    load_email_notification_config,
    parse_allocate_spec,
    run_command_with_lease,
    select_gpus,
    send_completion_email_notification,
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


class EmailNotificationConfigTests(unittest.TestCase):
    def test_returns_none_when_notifications_disabled(self) -> None:
        self.assertIsNone(load_email_notification_config({}))

    def test_parses_enabled_configuration(self) -> None:
        config = load_email_notification_config(
            {
                "GPU_ALLOC_EMAIL_ENABLED": "1",
                "GPU_ALLOC_EMAIL_SMTP_HOST": "smtp.example.com",
                "GPU_ALLOC_EMAIL_SMTP_FROM": "bot@example.com",
                "GPU_ALLOC_EMAIL_SMTP_TO": "user1@example.com, user2@example.com",
                "GPU_ALLOC_EMAIL_SMTP_USERNAME": "bot",
                "GPU_ALLOC_EMAIL_SMTP_PASSWORD": "secret",
                "GPU_ALLOC_EMAIL_SMTP_USE_STARTTLS": "true",
                "GPU_ALLOC_EMAIL_SUBJECT_PREFIX": "train",
            }
        )

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(config.smtp_host, "smtp.example.com")
        self.assertEqual(config.smtp_port, 587)
        self.assertEqual(config.from_addr, "bot@example.com")
        self.assertEqual(config.to_addrs, ("user1@example.com", "user2@example.com"))
        self.assertEqual(config.username, "bot")
        self.assertEqual(config.password, "secret")
        self.assertTrue(config.use_starttls)
        self.assertEqual(config.subject_prefix, "train")


class EmailNotificationSendTests(unittest.TestCase):
    def test_send_completion_email_notification_uses_starttls_and_login(self) -> None:
        config = EmailNotificationConfig(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_addr="bot@example.com",
            to_addrs=("user@example.com",),
            username="bot",
            password="secret",
            use_starttls=True,
            subject_prefix="train",
        )
        smtp_client = mock.MagicMock()
        smtp_client.__enter__.return_value = smtp_client

        with mock.patch("gpu_alloc.core.smtplib.SMTP", return_value=smtp_client) as smtp_class:
            send_completion_email_notification(
                config,
                command=["python", "train.py"],
                gpu_ids=(0, 2),
                cuda_visible_devices="0,2",
                start_time=100.0,
                end_time=112.5,
                exit_code=0,
            )

        smtp_class.assert_called_once_with("smtp.example.com", 587, timeout=30)
        smtp_client.starttls.assert_called_once_with()
        smtp_client.login.assert_called_once_with("bot", "secret")
        message = smtp_client.send_message.call_args.args[0]
        self.assertEqual(message["Subject"], "[train] gpu-alloc job finished: success")
        body = message.get_content()
        self.assertIn("Status: success", body)
        self.assertIn("Exit code: 0", body)
        self.assertIn("Command: python train.py", body)
        self.assertIn("GPU IDs: 0, 2", body)
        self.assertIn("CUDA_VISIBLE_DEVICES: 0,2", body)
        self.assertIn("Duration: 12.5s", body)

    def test_send_completion_email_notification_uses_ssl_transport(self) -> None:
        config = EmailNotificationConfig(
            smtp_host="smtp.example.com",
            smtp_port=465,
            from_addr="bot@example.com",
            to_addrs=("user@example.com",),
            use_ssl=True,
        )
        smtp_client = mock.MagicMock()
        smtp_client.__enter__.return_value = smtp_client

        with mock.patch("gpu_alloc.core.smtplib.SMTP_SSL", return_value=smtp_client) as smtp_ssl:
            with mock.patch("gpu_alloc.core.smtplib.SMTP") as smtp_plain:
                send_completion_email_notification(
                    config,
                    command=["python", "train.py"],
                    gpu_ids=(0,),
                    cuda_visible_devices="0",
                    start_time=10.0,
                    end_time=11.0,
                    exit_code=3,
                )

        smtp_ssl.assert_called_once_with("smtp.example.com", 465, timeout=30)
        smtp_plain.assert_not_called()


class RunCommandWithLeaseTests(unittest.TestCase):
    def test_sends_notification_and_releases_lease_after_child_exit(self) -> None:
        env = {
            "GPU_ALLOC_EMAIL_ENABLED": "1",
            "GPU_ALLOC_EMAIL_SMTP_HOST": "smtp.example.com",
            "GPU_ALLOC_EMAIL_SMTP_FROM": "bot@example.com",
            "GPU_ALLOC_EMAIL_SMTP_TO": "user@example.com",
        }
        child = mock.Mock()
        child.wait.return_value = 7
        child.poll.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LeaseStore(temp_dir)
            with store.locked():
                lease = store.create_lease([0], ["python", "train.py"], lease_seconds=10)

            with mock.patch("gpu_alloc.core.subprocess.Popen", return_value=child):
                with mock.patch("gpu_alloc.core.LeaseHeartbeat", return_value=contextlib.nullcontext()):
                    with mock.patch("gpu_alloc.core.signal.getsignal", return_value=mock.sentinel.handler):
                        with mock.patch("gpu_alloc.core.signal.signal"):
                            with mock.patch("gpu_alloc.core.send_completion_email_notification") as sender:
                                exit_code = run_command_with_lease(
                                    ["python", "train.py"],
                                    env=env,
                                    lease_store=store,
                                    lease=lease,
                                    lease_seconds=10,
                                    gpu_ids=(0,),
                                    cuda_visible_devices="0",
                                )

            self.assertEqual(exit_code, 7)
            self.assertEqual(sender.call_args.kwargs["exit_code"], 7)
            self.assertEqual(sender.call_args.kwargs["gpu_ids"], (0,))
            with store.locked():
                self.assertEqual(store.load_active_leases(), {})

    def test_notification_failure_does_not_override_child_exit_code(self) -> None:
        env = {
            "GPU_ALLOC_EMAIL_ENABLED": "1",
            "GPU_ALLOC_EMAIL_SMTP_HOST": "smtp.example.com",
            "GPU_ALLOC_EMAIL_SMTP_FROM": "bot@example.com",
            "GPU_ALLOC_EMAIL_SMTP_TO": "user@example.com",
        }
        child = mock.Mock()
        child.wait.return_value = 0
        child.poll.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LeaseStore(temp_dir)
            with store.locked():
                lease = store.create_lease([0], ["python", "train.py"], lease_seconds=10)

            stderr = io.StringIO()
            with mock.patch("gpu_alloc.core.subprocess.Popen", return_value=child):
                with mock.patch("gpu_alloc.core.LeaseHeartbeat", return_value=contextlib.nullcontext()):
                    with mock.patch("gpu_alloc.core.signal.getsignal", return_value=mock.sentinel.handler):
                        with mock.patch("gpu_alloc.core.signal.signal"):
                            with mock.patch(
                                "gpu_alloc.core.send_completion_email_notification",
                                side_effect=RuntimeError("smtp down"),
                            ):
                                with mock.patch("sys.stderr", new=stderr):
                                    exit_code = run_command_with_lease(
                                        ["python", "train.py"],
                                        env=env,
                                        lease_store=store,
                                        lease=lease,
                                        lease_seconds=10,
                                        gpu_ids=(0,),
                                        cuda_visible_devices="0",
                                    )

        self.assertEqual(exit_code, 0)
        self.assertIn("Warning: failed to send completion email: smtp down", stderr.getvalue())

    def test_invalid_notification_config_does_not_block_child_execution(self) -> None:
        child = mock.Mock()
        child.wait.return_value = 0
        child.poll.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LeaseStore(temp_dir)
            with store.locked():
                lease = store.create_lease([0], ["python", "train.py"], lease_seconds=10)

            stderr = io.StringIO()
            with mock.patch("gpu_alloc.core.subprocess.Popen", return_value=child):
                with mock.patch("gpu_alloc.core.LeaseHeartbeat", return_value=contextlib.nullcontext()):
                    with mock.patch("gpu_alloc.core.signal.getsignal", return_value=mock.sentinel.handler):
                        with mock.patch("gpu_alloc.core.signal.signal"):
                            with mock.patch("sys.stderr", new=stderr):
                                exit_code = run_command_with_lease(
                                    ["python", "train.py"],
                                    env={"GPU_ALLOC_EMAIL_ENABLED": "1"},
                                    lease_store=store,
                                    lease=lease,
                                    lease_seconds=10,
                                    gpu_ids=(0,),
                                    cuda_visible_devices="0",
                                )

        self.assertEqual(exit_code, 0)
        self.assertIn("Email notification disabled:", stderr.getvalue())

    def test_launch_failure_sends_notification_and_releases_lease(self) -> None:
        env = {
            "GPU_ALLOC_EMAIL_ENABLED": "1",
            "GPU_ALLOC_EMAIL_SMTP_HOST": "smtp.example.com",
            "GPU_ALLOC_EMAIL_SMTP_FROM": "bot@example.com",
            "GPU_ALLOC_EMAIL_SMTP_TO": "user@example.com",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            store = LeaseStore(temp_dir)
            with store.locked():
                lease = store.create_lease([0], ["python", "train.py"], lease_seconds=10)

            with mock.patch("gpu_alloc.core.subprocess.Popen", side_effect=FileNotFoundError("python")):
                with mock.patch("gpu_alloc.core.send_completion_email_notification") as sender:
                    with self.assertRaises(FileNotFoundError):
                        run_command_with_lease(
                            ["python", "train.py"],
                            env=env,
                            lease_store=store,
                            lease=lease,
                            lease_seconds=10,
                            gpu_ids=(0,),
                            cuda_visible_devices="0",
                        )

            self.assertEqual(sender.call_args.kwargs["exit_code"], None)
            self.assertIsInstance(sender.call_args.kwargs["launch_error"], FileNotFoundError)
            with store.locked():
                self.assertEqual(store.load_active_leases(), {})
