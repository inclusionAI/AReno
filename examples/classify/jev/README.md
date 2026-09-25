# JevForge-style decision scorer

Trains the same model as [jev-forge](https://github.com/zwliJay/jev-forge):
a causal-LM backbone plus a `Linear -> GELU -> Linear(1)` head on the last
token of each candidate path. The candidates of one question share a single
softmax, and that softmax is fitted to the target distribution with
cross-entropy plus `brier_weight` × Brier.

## How it maps onto AReno

| JevForge | AReno |
| --- | --- |
| `JevForgeModel.head` | `score_head` submodule on the actor (`RuntimeConfig.score_head`, `areno/engine/score_head.py`) |
| no LM-head projection | Qwen3 / Qwen3.5 forward with `defer_lm_head=True` |
| `question_loss` | `classify_loss_fn` (`areno/experimental/classify/loss.py`) |
| `batches_by_tokens` | `build_step_rows`: whole questions per DP rank and microbatch, capped at `--microbatch-tokens` real tokens (packed, no padding) |
| head-only warmup, `head_lr` | `score_head_warmup_steps` keeps the backbone LR at 0; `score_head_lr` is a separate LR group |
| `best.safetensors` | `step_XXXXXX/` HF backbone + `score_head.safetensors`; convert with `export_jevforge.py` |

## Run

```bash
# Train. Records are a JevForge split directory holding train.jsonl.
python examples/classify/jev/train.py \
  --records /data/jevforge/web_full --ckpt Qwen/Qwen3.5-0.8B \
  --save-path runs/jev --world-size 8 --tp-size 1 --max-steps 600

# Convert to the JevForge layout, then evaluate or serve with jev-forge.
python examples/classify/jev/export_jevforge.py \
  --ckpt runs/jev/step_000600 --out runs/jev/jevforge --max-length 512
```

## Differences from jev-forge's `train.py`

- **Selecting the best checkpoint and fitting the temperature are not built in.**
  jev-forge picks the checkpoint with the lowest dev CE and fits a temperature
  on the calibration split. Here you save periodically, export the checkpoint,
  and run jev-forge's evaluation. Pass the temperature you fitted to
  `export_jevforge.py --temperature`.
- **The head-only warmup sets the backbone LR to 0 instead of freezing it.**
  AdamW still accumulates backbone moments during those steps.
- **Supported families:** the score head works only with model families whose
  forward accepts `defer_lm_head` (qwen3, qwen3_5, gemma4). It does not work
  with native LoRA.
