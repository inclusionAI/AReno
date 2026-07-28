# Issue #257 — GPU run history implementation notes

## Scope

The CLI can opt into periodic per-device GPU memory, utilization, and
temperature sampling during `trainer.fit()`. Sampling is off by default, uses
no dependency beyond `nvidia-smi`, and never enters the rollout, loss, or
optimizer path.

User-facing behavior is documented in:

- `docs/cli/training.rst`
- `docs/cli/observability.rst`

## CLI contract

| Option | Default | Contract |
|---|---:|---|
| `--gpu-stats` | off | Enable sampling for this training run |
| `--gpu-stats-interval-s` | `5.0` | Positive sampling interval in seconds |
| `--gpu-stats-history` | `1000` | Positive total row bound for memory and JSONL |

Both bounds are validated before model or worker initialization. The same
validation is present on `TrainerConfig` for non-Click construction.

## Device mapping

`nvidia-smi` reports physical indices and GPU UUIDs. AReno records those
identities and maps them to the logical CUDA order used by the run:

- With `CUDA_VISIBLE_DEVICES=3,1`, physical GPU 3 becomes logical device 0
  and physical GPU 1 becomes logical device 1.
- GPU UUID selectors and unambiguous UUID prefixes are supported.
- Without `CUDA_VISIBLE_DEVICES`, the first `world_size` physical indices are
  selected.

Every sample contains logical `index`, `physical_index`, and `uuid`. This makes
multi-GPU results auditable after the run instead of relying on host-global
indices alone.

## Bounded artifacts

The sampler owns a `deque(maxlen=gpu_stats_history)`. After each tick it
atomically replaces `gpu_stats.<pid>.jsonl` with that bounded snapshot. The
temporary-file rename means a process interruption leaves either the previous
complete snapshot or the new one, never a partially written JSON line.

At shutdown AReno writes `gpu_stats_summary.<pid>.json` with:

- run PID, interval, history cap, duration, and selected devices;
- logical-to-physical mapping, UUID, and GPU name;
- peak memory used, total memory, mean utilization, maximum temperature, and
  sample count per device;
- the affected stage and a bounded diagnostic message if sampling degraded.

All timestamps are Unix wall-clock seconds so histories from different runs
can be compared.

## Failure contract

GPU observability is best effort:

- missing `nvidia-smi` is reported as a discovery failure;
- timeout/non-zero exit is reported as a query failure;
- malformed or empty output is reported as a parse failure;
- sampling, shutdown, and artifact failures produce warnings;
- telemetry cleanup never replaces the original training exception.

The daemon thread has no access to training samples.

## CPU verification

`tests/test_gpu_stats_cpu.py` uses fake samples and covers:

- multi-device CSV parsing and missing fields;
- numeric and UUID `CUDA_VISIBLE_DEVICES` mapping;
- mapping through the real sampler worker loop;
- bounded memory and bounded JSONL snapshots;
- idempotent shutdown;
- missing-tool and sampler-failure diagnostics;
- CLI lifecycle integration through config, sampler, JSONL, and summary JSON;
- disabled/default behavior and artifact failure isolation.

`tests/test_train_cli_config_cpu.py` covers help grouping and clear validation
errors for both public bounds.

Run:

```bash
python -m pytest tests/ -k cpu
ruff check .
ruff format --check .
```

## Minimal NVIDIA/Kaggle validation

Run a tiny two-step training job with a one-second interval:

```bash
CUDA_VISIBLE_DEVICES=0 areno train \
  --algo sft \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path tatsu-lab/alpaca \
  --model-hub hf \
  --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
  --world-size 1 \
  --tp-size 1 \
  --batch-size 1 \
  --mini-bs 1 \
  --max-prompt-tokens 128 \
  --max-new-tokens 64 \
  --attn-backend native \
  --max-steps 2 \
  --gpu-stats \
  --gpu-stats-interval-s 1 \
  --gpu-stats-history 120 \
  --metrics-log-dir /tmp/areno/gpu-check
```

Then verify:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/tmp/areno/gpu-check")
history = next(root.glob("gpu_stats.*.jsonl"))
summary = next(root.glob("gpu_stats_summary.*.json"))
rows = [json.loads(line) for line in history.read_text().splitlines()]
data = json.loads(summary.read_text())

assert rows
assert len(rows) <= 120
assert data["devices"] == [0]
assert data["per_device"]["0"]["physical_index"] == 0
assert data["per_device"]["0"]["n_samples"] >= 1
assert data["failure"] is None, data["failure"]
print(json.dumps(data, indent=2))
PY
```

For a two-GPU runner, repeat with `CUDA_VISIBLE_DEVICES=1,0`,
`--world-size 2`, and assert logical device 0 maps to physical 1 while logical
device 1 maps to physical 0.
