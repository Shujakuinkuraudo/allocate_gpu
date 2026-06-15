# gpu-alloc

`gpu-alloc` is a Python CLI for reserving GPUs before starting a CUDA job.
It reads GPU state from `nvidia-smi`, chooses GPUs that satisfy a free-memory
threshold and a utilization threshold, records a lease in a shared host-level
state directory, sets `CUDA_VISIBLE_DEVICES`, and then launches the target
command.

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

## Shared State And Namespaces

By default, lease state is no longer tied to the current working directory.
The default base state directory is:

```bash
/var/tmp/gpu-alloc
```

The effective lease directory is:

```bash
/var/tmp/gpu-alloc/$HOSTNAME/default
```

You can isolate separate callers with `--namespace`:

```bash
gpu-alloc --namespace eval-a 1_gpu,40G -- python train.py
gpu-alloc --namespace eval-b 1_gpu,40G -- python train.py
```

Or override the base directory entirely:

```bash
gpu-alloc --state-dir /scratch/gpu-alloc --namespace team-a 1_gpu,40G -- python train.py
```

The corresponding environment variables are:

```bash
export GPU_ALLOC_STATE_DIR=/var/tmp/gpu-alloc
export GPU_ALLOC_NAMESPACE=default
```

CLI flags still take precedence over environment defaults.

## Waiting And Explainability

If GPUs are not available, `gpu-alloc` can wait and periodically print the
current state:

```bash
gpu-alloc --watch --verbose 1_gpu,40G -- python train.py
```

Useful options:

- `--watch`: periodically print waiting records
- `--watch-interval`: control how often waiting records are emitted
- `--verbose`: include per-GPU candidate details
- `--no-wait`: fail immediately instead of retrying

The waiting record includes:

- requested allocation
- required free memory threshold
- utilization threshold
- currently reserved GPU IDs
- currently eligible GPU IDs
- next retry delay
- per-GPU free memory / utilization details when `--verbose` is set

## Explain Mode

You can inspect the current allocation state without launching a child command:

```bash
gpu-alloc --explain 1_gpu,40G
```

Machine-readable JSON is also available:

```bash
gpu-alloc --explain --json 1_gpu,40G
```

## Selection Strategy

Two selection strategies are available:

- `pack` (default): prefer lower utilization, then higher free memory, then lower index
- `spread`: prefer higher free memory first, then lower utilization

Examples:

```bash
gpu-alloc --strategy pack 2_gpu,40G -- python train.py
gpu-alloc --strategy spread 2_gpu,40G -- python train.py
```

## Shell-Friendly Command Entry

If quoting through tmux / ssh / bash is painful, use `--shell`:

```bash
gpu-alloc 1_gpu,40G --shell 'python -m vllm.entrypoints.openai.api_server --model foo'
```

Or use `--command-file`:

```bash
gpu-alloc 1_gpu,40G --command-file ./run-train.sh
```

Do not combine `--shell`, `--command-file`, and a child command after `--`.

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

- The email is sent after the child command exits, regardless of success or failure.
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

## Defaults

- Memory safety factor: `1.5`
- GPU utilization threshold: `30`
- Wait mode: enabled
- Poll interval: `10s`
- Watch interval: `10s`
- Lease TTL: `45s`, refreshed while the allocator process is alive
- Base lease directory: `/var/tmp/gpu-alloc`
- Default namespace: `default`
- Default strategy: `pack`
- CLI env defaults:
  - `GPU_ALLOC_MEMORY_MARGIN`
  - `GPU_ALLOC_UTILIZATION_THRESHOLD`
  - `GPU_ALLOC_POLL_INTERVAL`
  - `GPU_ALLOC_WATCH_INTERVAL`
  - `GPU_ALLOC_LEASE_SECONDS`
  - `GPU_ALLOC_STATE_DIR`
  - `GPU_ALLOC_NAMESPACE`
  - `GPU_ALLOC_STRATEGY`
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
