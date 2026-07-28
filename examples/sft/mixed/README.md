# Deterministic weighted SFT dataset mix

This local example mixes two already-normalized SFT sources without network
services or external databases:

```bash
areno train \
  --algo sft \
  --ckpt /path/to/local/model \
  --dataset-mix-config examples/sft/mixed/mix.json \
  --dataset-loader-fn examples/sft/mixed/dataset_loader.py \
  --world-size 1 \
  --tp-size 1 \
  --batch-size 2 \
  --epochs 1 \
  --metrics-log-dir outputs/mixed-sft/metrics
```

For the default seed ``42``, ``renormalize`` policy, and source shuffling, the
same sources can be supplied directly on the command line:

```bash
areno train \
  --algo sft \
  --ckpt /path/to/local/model \
  --dataset-source math=examples/sft/mixed/math.jsonl:0.7 \
  --dataset-source code=examples/sft/mixed/code.jsonl:0.3 \
  --dataset-loader-fn examples/sft/mixed/dataset_loader.py \
  --world-size 1 \
  --tp-size 1 \
  --batch-size 2 \
  --epochs 1 \
  --metrics-log-dir outputs/mixed-sft/metrics
```

Repeat ``--dataset-source NAME=PATH:WEIGHT`` at least twice. Use the JSON
manifest form to choose another seed, exhaustion policy, shuffle setting, or
cycle sample bound.

Before model or worker initialization, AReno validates both sources and prints
the planned source counts. It also writes
`dataset_mix_plan.<pid>.json` under the metrics directory without including
prompt or response contents. During training,
`stage=dataset_mix_progress` logs cumulative per-source counts and observed
proportions for rows whose optimizer step completed.

For a boundary-input example, change a weight to `0`; validation then fails
with `weight must be finite and positive`.
