from __future__ import annotations

import io
import os
import unittest
from contextlib import redirect_stdout
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
        self.assertEqual(args[0], ["python", "train.py"])
