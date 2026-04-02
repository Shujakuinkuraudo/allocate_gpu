# AGENTS.md

## Project Summary

This repository contains `gpu-alloc`, a Python CLI that selects GPUs based on
free memory and GPU utilization, records a local lease, sets
`CUDA_VISIBLE_DEVICES`, and then executes a child command.

Preferred end-user command shape:

```bash
gpu-alloc 1_gpu,40G -- pixi run uv run -- python train.py
```

## Repository Layout

- `gpu_alloc/cli.py`: CLI parsing and process launch entrypoint
- `gpu_alloc/core.py`: allocation parsing, `nvidia-smi` probing, selection,
  lease storage, and child-process execution
- `tests/test_core.py`: unit tests for parsing, probing, selection, and lease
  behavior
- `tests/test_cli.py`: unit tests for CLI behavior
- `pixi.toml`: pixi workspace and package metadata
- `pyproject.toml`: Python package metadata and console script definition

## Development Commands

- Run tests with `pixi run test`
- Alternate test command: `python3 -m unittest discover -s tests -t . -v`
- Show CLI help with `python3 -m gpu_alloc --help`
- Install as a global pixi tool with:

```bash
pixi global install --environment gpu-alloc --path "$(pwd)" --expose gpu-alloc=gpu-alloc
```

## Implementation Notes

- The preferred CLI input is the positional allocation spec:
  `gpu-alloc 3_gpu,40G -- ...`
- Compatibility inputs still exist:
  - `--allocate 3_gpu,40G`
  - `ALLOCATE=3_gpu,40G`
- GPU selection currently requires:
  - `memory.free >= requested_memory * memory_margin`
  - `utilization.gpu <= utilization_threshold`
- Eligible GPUs are sorted by:
  - lower utilization
  - higher free memory
  - lower GPU index
- Lease state is stored in `./.gpu-alloc/leases.json` by default
- `--print-only` intentionally does not create a lease

## Editing Guidance

- Keep the tool standard-library only unless there is a clear need for a new
  dependency
- Do not commit `.venv/`, `.pixi/`, `.gpu-alloc/`, `*.egg-info/`, or
  `__pycache__/`
- If changing CLI behavior, update both:
  - tests in `tests/test_cli.py`
  - user-facing examples in `README.md`
- If changing allocation rules or lease behavior, update:
  - tests in `tests/test_core.py`
  - the defaults and behavior notes in `README.md`

## Environment Assumptions

- Real allocation tests require a machine where `/usr/bin/nvidia-smi` works
- Sandbox execution may not see the NVIDIA driver even when the host can
- For real-machine validation, compare:
  - raw query output from `nvidia-smi --query-gpu=...`
  - allocator output from `gpu-alloc ... --print-only`
