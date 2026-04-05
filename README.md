# gpu-alloc

`gpu-alloc` is a small Python CLI for reserving GPUs before starting a CUDA job.
It reads GPU state from `nvidia-smi`, chooses GPUs that satisfy a free-memory
threshold and a utilization threshold, records a local lease, sets
`CUDA_VISIBLE_DEVICES`, and then launches the target command.

## Requirements

- Linux with a working NVIDIA driver
- `nvidia-smi` available on the host
- Python 3.10+
- `pixi` if you want to run it through the workspace task

## Usage

Run a command after reserving three GPUs where each GPU has at least `40G * 1.5`
free memory and is below the utilization threshold:

```bash
gpu-alloc 3_gpu,40G -- pixi run uv run -- python train.py
```

If you are running from the workspace without a global install:

```bash
pixi run gpu-alloc -- 3_gpu,40G -- pixi run uv run -- python train.py
```

Compatibility forms still work:

```bash
gpu-alloc --allocate 2_gpu,24G --print-only
ALLOCATE=2_gpu,24G gpu-alloc --print-only
```

The `--print-only` mode prints `CUDA_VISIBLE_DEVICES=...` and does not create a
lease, so it is useful for inspection but not for race-free reservation.

Common CLI defaults can also come from environment variables:

```bash
export GPU_ALLOC_MEMORY_MARGIN=1.5
export GPU_ALLOC_UTILIZATION_THRESHOLD=30
export GPU_ALLOC_POLL_INTERVAL=10
export GPU_ALLOC_LEASE_SECONDS=120
```

CLI flags still take precedence over these environment defaults.

## Email Notification

`gpu-alloc` can send a summary email after the child command finishes. This is
configured entirely through environment variables.

Required variables:

```bash
export GPU_ALLOC_EMAIL_ENABLED=1
export GPU_ALLOC_EMAIL_SMTP_HOST=smtp.example.com
export GPU_ALLOC_EMAIL_SMTP_FROM=bot@example.com
export GPU_ALLOC_EMAIL_SMTP_TO=you@example.com
```

Optional variables:

```bash
export GPU_ALLOC_EMAIL_SMTP_PORT=587
export GPU_ALLOC_EMAIL_SMTP_USERNAME=bot@example.com
export GPU_ALLOC_EMAIL_SMTP_PASSWORD=secret
export GPU_ALLOC_EMAIL_SMTP_USE_SSL=0
export GPU_ALLOC_EMAIL_SMTP_USE_STARTTLS=1
export GPU_ALLOC_EMAIL_SUBJECT_PREFIX=train
```

Behavior notes:

- The email is sent after the child command exits, regardless of success or
  failure.
- `--print-only` does not send email because no child command is launched.
- Notification setup errors or SMTP delivery failures are reported to `stderr`
  and do not change the child command exit code.
- If `GPU_ALLOC_EMAIL_SMTP_PORT` is unset, the default is `465` for SSL and
  `587` otherwise.

## Pixi Global Install

Install the current project as a globally exposed command:

```bash
pixi global install --environment gpu-alloc --path "$(pwd)" --expose gpu-alloc=gpu-alloc
```

Install directly from the public GitHub repository:

```bash
pixi global install --environment gpu-alloc --git https://github.com/Shujakuinkuraudo/allocate_gpu.git --expose gpu-alloc=gpu-alloc
```

Notes:

- `pixi global install https://github.com/...` is not supported by pixi. Use
  `--git <url>` instead.
- `pixi global install gpu-alloc` is not available yet because that requires
  publishing the built package to a searchable conda channel such as
  prefix.dev, anaconda.org, or conda-forge.

After that, the preferred command shape is:

```bash
gpu-alloc 1_gpu,40G -- pixi run uv run -- python 1.py
```

## Defaults

- Memory safety factor: `1.5`
- GPU utilization threshold: `30`
- Wait mode: enabled
- Poll interval: `10s`
- Lease TTL: `120s`, refreshed while the child process is alive
- Lease directory: `./.gpu-alloc`
- CLI env defaults: `GPU_ALLOC_MEMORY_MARGIN`, `GPU_ALLOC_UTILIZATION_THRESHOLD`,
  `GPU_ALLOC_POLL_INTERVAL`, `GPU_ALLOC_LEASE_SECONDS`
- Email notification: disabled unless `GPU_ALLOC_EMAIL_ENABLED=1`

## Development

Run the test suite:

```bash
python3 -m unittest discover -s tests -t . -v
```

Or through pixi:

```bash
pixi run test
```
