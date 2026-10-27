# typed-decisions (evaluation set)

Vendored copy of [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions),
config `all`, used as the out-of-distribution evaluation set for the classify
example. The upstream dataset card is kept verbatim in `UPSTREAM_README.md`.

| | |
| --- | --- |
| Source | Hugging Face `LocalLLaMA/typed-decisions`, revision `f7a2487edd7a043a5441a5e9ccc7fe5ddbd9ebe8` |
| Files | `all/test-00000-of-00001.parquet` → `test.parquet`, `all/train-00000-of-00001.parquet` → `train.parquet` (unmodified) |
| Retrieved | 2026-09-26 |
| License | Apache-2.0 (same as this repository, see the top-level `LICENSE`) |
| sha256 `test.parquet` | `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c` |
| sha256 `train.parquet` | `46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5` |

- 400 test cases / 2,000 decisions and 1,200 train cases across four workflows
  (`agent_trace_observability`, `customer_service`, `invoice_processing`,
  `security_incidents`). Each case asks 5 `noul` / `choice` / `score` questions
  over one JSON state.
- Gold is the mean of three samples from a ~4B-class teacher, so scores measure
  agreement with that teacher; the card puts saturation near 0.75 accuracy.
- Leaderboard reference (zero-shot, `test`): TypeSafe Jev 1.13.0 accuracy 0.727,
  KL 1.442, Brier 0.148. The card does not publish its metric code, so
  `evaluate.py` numbers are an approximate comparison.

Convert and evaluate:

```bash
python examples/classify/jev/convert_datasets.py typed-decisions \
  --src examples/classify/jev/data/typed-decisions --out ~/data/jev-records/typed-decisions
python examples/classify/jev/evaluate.py --checkpoint ~/areno-runs/ling-3.0-tiny-jev \
  --records ~/data/jev-records/typed-decisions --split test
```
