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

Before model or worker initialization, AReno validates both sources and prints
the planned source counts. It also writes
`dataset_mix_plan.<pid>.json` under the metrics directory without including
prompt or response contents.

For a boundary-input example, change a weight to `0`; validation then fails
with `weight must be finite and positive`.
