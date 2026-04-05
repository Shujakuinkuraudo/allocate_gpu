from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from gpu_alloc.cli import main, normalize_command, resolve_allocate_value, split_command
from gpu_alloc.core import AllocationDecision, AllocationRequest, SelectionResult


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


class MainTests(unittest.TestCase):
    def test_environment_defaults_are_used_for_allocator_config(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(1,),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=None,
        )
        fake_allocator = mock.Mock()
        fake_allocator.allocate.return_value = decision

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

    def test_cli_flags_override_environment_defaults(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(1,),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=None,
        )
        fake_allocator = mock.Mock()
        fake_allocator.allocate.return_value = decision

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
                                ]
                            )

        self.assertEqual(exit_code, 0)
        config = allocator_class.call_args.args[2]
        self.assertEqual(config.poll_interval, 9.0)
        self.assertEqual(config.memory_margin, 2.0)
        self.assertEqual(config.utilization_threshold, 5)
        self.assertEqual(config.lease_seconds, 30.0)

    def test_invalid_environment_default_returns_error(self) -> None:
        stderr = io.StringIO()

        with mock.patch.dict(os.environ, {"GPU_ALLOC_MEMORY_MARGIN": "bad"}, clear=False):
            with redirect_stderr(stderr):
                exit_code = main(["1_gpu,40G", "--print-only"])

        self.assertEqual(exit_code, 1)
        self.assertIn("Invalid GPU_ALLOC_MEMORY_MARGIN value", stderr.getvalue())

    def test_print_only_outputs_selected_devices(self) -> None:
        decision = AllocationDecision(
            selection=SelectionResult(
                gpu_ids=(1, 3),
                gpus=(),
                required_free_mib=61440,
            ),
            lease=None,
        )

        fake_allocator = mock.Mock()
        fake_allocator.allocate.return_value = decision

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
        fake_allocator = mock.Mock()
        fake_allocator.allocate.return_value = decision

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
        fake_allocator = mock.Mock()
        fake_allocator.allocate.return_value = decision

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
