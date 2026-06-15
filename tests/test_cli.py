from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from gpu_alloc.cli import (
    DEFAULT_NAMESPACE,
    DEFAULT_STATE_BASE_DIR,
    build_parser,
    main,
    normalize_command,
    resolve_allocate_value,
    resolve_command,
    resolve_state_dir,
    split_command,
)
from gpu_alloc.core import AllocationDecision, AllocationRequest, AllocationStatus, GPUInfo, SelectionResult


class CliHelperTests(unittest.TestCase):
    def test_normalize_command_strips_separator(self) -> None:
        self.assertEqual(normalize_command(["--", "python", "train.py"]), ["python", "train.py"])

    def test_split_command_separates_child_args(self) -> None:
        self.assertEqual(
            split_command(["1_gpu,40G", "--no-wait", "--", "pixi", "run", "python", "1.py"]),
            (["1_gpu,40G", "--no-wait"], ["pixi", "run", "python", "1.py"]),
        )

    def test_cli_value_overrides_environment(self) -> None:
        self.assertEqual(
            resolve_allocate_value("1_gpu,24G", None, {"ALLOCATE": "2_gpu,40G"}),
            "1_gpu,24G",
        )

    def test_option_value_used_when_positional_missing(self) -> None:
        self.assertEqual(
            resolve_allocate_value(None, "1_gpu,24G", {"ALLOCATE": "2_gpu,40G"}),
            "1_gpu,24G",
        )

    def test_conflicting_values_raise(self) -> None:
        with self.assertRaisesRegex(Exception, "Conflicting allocation requests"):
            resolve_allocate_value("1_gpu,24G", "2_gpu,40G", {})

    def test_resolve_state_dir_uses_hostname_and_namespace(self) -> None:
        with mock.patch("gpu_alloc.cli.socket.gethostname", return_value="lab-a100"):
            self.assertEqual(
                resolve_state_dir(DEFAULT_STATE_BASE_DIR, "run1"),
                str(Path(DEFAULT_STATE_BASE_DIR) / "lab-a100" / "run1"),
            )

    def test_resolve_command_shell_and_file(self) -> None:
        self.assertEqual(resolve_command([], shell="echo hi", command_file=""), ["/bin/sh", "-lc", "echo hi"])
        self.assertEqual(resolve_command([], shell="", command_file="/tmp/x.sh"), ["/bin/sh", "/tmp/x.sh"])

    def test_resolve_command_rejects_multiple_modes(self) -> None:
        with self.assertRaises(Exception):
            resolve_command(["python", "train.py"], shell="echo hi", command_file="")


class MainTests(unittest.TestCase):
    def _fake_allocator(self, decision: AllocationDecision) -> mock.Mock:
        fake_allocator = mock.Mock()
        fake_allocator.allocate.return_value = decision
        fake_allocator.explain.return_value = AllocationStatus(
            request=AllocationRequest(gpu_count=1, memory_mib=1024, raw="1_gpu,1G"),
            required_free_mib=1024,
            utilization_threshold=30,
            reserved_ids=(1,),
            eligible_ids=(0,),
            gpus=(GPUInfo(index=0, name="A", total_mib=81920, free_mib=70000, utilization_gpu=5),),
            next_retry_in=10.0,
            attempt=1,
            strategy="pack",
        )
        return fake_allocator

    def test_environment_defaults_are_used_for_allocator_config(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(1,),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=None,
        )
        fake_allocator = self._fake_allocator(decision)

        env = {
            "ALLOCATE": "1_gpu,40G",
            "GPU_ALLOC_POLL_INTERVAL": "3.5",
            "GPU_ALLOC_MEMORY_MARGIN": "1.8",
            "GPU_ALLOC_UTILIZATION_THRESHOLD": "12",
            "GPU_ALLOC_LEASE_SECONDS": "45",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch("gpu_alloc.cli.GPUAllocator", return_value=fake_allocator) as allocator_class:
                with mock.patch("gpu_alloc.cli.NvidiaSmiProbe"):
                    with mock.patch("gpu_alloc.cli.LeaseStore"):
                        stdout = io.StringIO()
                        with redirect_stdout(stdout):
                            exit_code = main(["--print-only"])

        self.assertEqual(exit_code, 0)
        config = allocator_class.call_args.args[2]
        self.assertEqual(config.poll_interval, 3.5)
        self.assertEqual(config.memory_margin, 1.8)
        self.assertEqual(config.utilization_threshold, 12)
        self.assertEqual(config.lease_seconds, 45.0)
        self.assertEqual(config.strategy, "pack")

    def test_cli_flags_override_environment_defaults(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(1,),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=None,
        )
        fake_allocator = self._fake_allocator(decision)

        env = {
            "ALLOCATE": "1_gpu,40G",
            "GPU_ALLOC_POLL_INTERVAL": "3.5",
            "GPU_ALLOC_MEMORY_MARGIN": "1.8",
            "GPU_ALLOC_UTILIZATION_THRESHOLD": "12",
            "GPU_ALLOC_LEASE_SECONDS": "45",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with mock.patch("gpu_alloc.cli.GPUAllocator", return_value=fake_allocator) as allocator_class:
                with mock.patch("gpu_alloc.cli.NvidiaSmiProbe"):
                    with mock.patch("gpu_alloc.cli.LeaseStore"):
                        stdout = io.StringIO()
                        with redirect_stdout(stdout):
                            exit_code = main(
                                [
                                    "--print-only",
                                    "--poll-interval",
                                    "9",
                                    "--memory-margin",
                                    "2.0",
                                    "--utilization-threshold",
                                    "5",
                                    "--lease-seconds",
                                    "30",
                                    "--strategy",
                                    "spread",
                                ]
                            )

        self.assertEqual(exit_code, 0)
        config = allocator_class.call_args.args[2]
        self.assertEqual(config.poll_interval, 9.0)
        self.assertEqual(config.memory_margin, 2.0)
        self.assertEqual(config.utilization_threshold, 5)
        self.assertEqual(config.lease_seconds, 30.0)
        self.assertEqual(config.strategy, "spread")

    def test_print_only_outputs_selected_devices(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(1, 3),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=None,
        )

        fake_allocator = self._fake_allocator(decision)

        with mock.patch.dict(os.environ, {"ALLOCATE": "2_gpu,40G"}, clear=False):
            with mock.patch("gpu_alloc.cli.GPUAllocator", return_value=fake_allocator):
                with mock.patch("gpu_alloc.cli.NvidiaSmiProbe"):
                    with mock.patch("gpu_alloc.cli.LeaseStore"):
                        stdout = io.StringIO()
                        with redirect_stdout(stdout):
                            exit_code = main(["--print-only"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue().strip(), "CUDA_VISIBLE_DEVICES=1,3")
        request = fake_allocator.allocate.call_args.args[0]
        self.assertEqual(request, AllocationRequest(gpu_count=2, memory_mib=40960, raw="2_gpu,40G"))

    def test_main_returns_child_exit_code(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(0,),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=mock.Mock(),
        )
        fake_allocator = self._fake_allocator(decision)

        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch("gpu_alloc.cli.GPUAllocator", return_value=fake_allocator):
                with mock.patch("gpu_alloc.cli.NvidiaSmiProbe"):
                    with mock.patch("gpu_alloc.cli.LeaseStore"):
                        with mock.patch("gpu_alloc.cli.run_command_with_lease", return_value=7) as runner:
                            exit_code = main(["1_gpu,40G", "--", "python", "train.py"])

        self.assertEqual(exit_code, 7)
        args, kwargs = runner.call_args
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(kwargs["gpu_ids"], (0,))
        self.assertEqual(kwargs["cuda_visible_devices"], "0")
        self.assertEqual(args[0], ["python", "train.py"])

    def test_main_preserves_notification_environment_for_runner(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(2,),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=mock.Mock(),
        )
        fake_allocator = self._fake_allocator(decision)

        email_env = {
            "GPU_ALLOC_EMAIL_ENABLED": "1",
            "GPU_ALLOC_EMAIL_SMTP_HOST": "smtp.example.com",
            "GPU_ALLOC_EMAIL_SMTP_FROM": "bot@example.com",
            "GPU_ALLOC_EMAIL_SMTP_TO": "user@example.com",
        }
        with mock.patch.dict(os.environ, email_env, clear=False):
            with mock.patch("gpu_alloc.cli.GPUAllocator", return_value=fake_allocator):
                with mock.patch("gpu_alloc.cli.NvidiaSmiProbe"):
                    with mock.patch("gpu_alloc.cli.LeaseStore"):
                        with mock.patch("gpu_alloc.cli.run_command_with_lease", return_value=0) as runner:
                            exit_code = main(["1_gpu,40G", "--", "python", "train.py"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(runner.call_args.kwargs["env"]["GPU_ALLOC_EMAIL_ENABLED"], "1")

    def test_explain_json_outputs_machine_readable_state(self) -> None:
        fake_allocator = self._fake_allocator(
            AllocationDecision(
                selection=SelectionResult(gpu_ids=(0,), gpus=(), required_free_mib=1024),
                lease=None,
            )
        )
        stdout = io.StringIO()
        with mock.patch("gpu_alloc.cli.GPUAllocator", return_value=fake_allocator):
            with mock.patch("gpu_alloc.cli.NvidiaSmiProbe"):
                with mock.patch("gpu_alloc.cli.LeaseStore"):
                    with redirect_stdout(stdout):
                        exit_code = main(["--explain", "--json", "1_gpu,1G"])
        self.assertEqual(exit_code, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["request"]["raw"], "1_gpu,1G")

    def test_watch_passes_status_callback_to_allocator(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(gpu_ids=(0,), gpus=(), required_free_mib=1024),
            lease=None,
        )
        fake_allocator = self._fake_allocator(decision)
        with mock.patch("gpu_alloc.cli.GPUAllocator", return_value=fake_allocator):
            with mock.patch("gpu_alloc.cli.NvidiaSmiProbe"):
                with mock.patch("gpu_alloc.cli.LeaseStore"):
                    main(["--watch", "--print-only", "1_gpu,1G"])
        self.assertIn("status_callback", fake_allocator.allocate.call_args.kwargs)
