# GaLore experiments

Scripts that reproduce the GaLore measurements in
`docs/concepts/galore-results.rst`. They drive the public `Trainer` (directly
or through `areno train`) on one GPU each. The usage guide is
`docs/concepts/galore.rst`.

| Script | Purpose |
| --- | --- |
| `sft_validation.py` | Fixed-budget SFT with a fixed held-out validation split, scored by the live actor. |
| `train_seeded.py` | `areno train` with per-step rollout seeds, optimizer-state checkpoints and resume. |
| `compare_resume.py` | Step-by-step comparison of an uninterrupted run, a resumed run and an uninterrupted control. |
| `check_state_roundtrip.py` | Load a saved optimizer state through the public `Trainer`, save it again and compare every tensor. |
| `microbench.py` | 1024 x 1024 state bytes and step cost, and per-block refresh time and workspace versus a full SVD. |
| `summarize.py` | Final validation loss, final-window reward, timing and memory tables from run directories. |

## Raw results

`results/` holds the logs behind every table: per-step SFT metrics
(`<run>/metrics.jsonl.gz`), RL and resume TensorBoard scalars
(`<run>/scalars.jsonl.gz`), `microbench.json`, the resume comparisons, and the
`*_summary.json` tables. Each group's `runs.json` records, per run, the source
commit, the SFT arguments with model and dataset SHA-256 digests and software
versions, or the configuration printed by `areno train`. All runs used the same
`areno/` tree as this change. `summarize.py` and `compare_resume.py` read the
gzipped logs directly, for example:

```bash
R=examples/galore/results
python examples/galore/summarize.py --output /tmp/rl.json --window 10 \
  --rl adamw=$R/rl/adamw_s1,$R/rl/adamw_s2,$R/rl/adamw_s3 \
  --rl galore=$R/rl/galore_s1,$R/rl/galore_s2,$R/rl/galore_s3
```

## Data and model

- Model: `Qwen/Qwen3-0.6B` (and `Qwen/Qwen3-1.7B` for the memory benchmark and a second SFT comparison), local
  copies whose safetensors SHA-256 digests are recorded in the SFT `runs.json` manifests.
- SFT data: `yahma/alpaca-cleaned` (`alpaca_data_cleaned.json`, 51,760 rows),
  normalized by `examples/sft/alpaca/dataset_loader.py`.
- RL data: GSM8K `main` train split, normalized by `examples/math/dataset_loader.py`
  and scored by `examples/math/math_verify_reward.py` (0/1 reward).

## SFT validation runs

The validation split is the first 1,000 in-budget rows of a fixed permutation
(`--split-seed 1234`); each `--seed` permutes the remaining rows. Every arm uses
400 steps of 32 rows, cosine learning rate 1e-5 to 1e-6, no weight decay, prompt
and response limits of 256 and 512 tokens.

```bash
for seed in 1 2 3; do
  common="--ckpt $MODEL --dataset-path $ALPACA --seed $seed --steps 400"
  python examples/galore/sft_validation.py $common --optimizer adamw --output runs/sft/adamw_s$seed
  python examples/galore/sft_validation.py $common --optimizer galore --galore-rank 128 \
    --galore-update-proj-gap 100 --galore-scale 4 --output runs/sft/galore_s$seed
  python examples/galore/sft_validation.py $common --optimizer adamw8bit --output runs/sft/adamw8bit_s$seed
  python examples/galore/sft_validation.py $common --optimizer adamw4bit --output runs/sft/adamw4bit_s$seed
done
```

The scale was chosen on a separate split (`--split-seed 999 --seed 0`).

## RL runs

```bash
common="--algo gspo --ckpt $MODEL --dataset-path $GSM8K_TRAIN \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --batch-size 32 --n-samples 8 --max-running-prompts 128 \
  --max-prompt-tokens 512 --max-new-tokens 1024 --disable-thinking --max-steps 100"
for seed in 1 2 3; do
  python examples/galore/train_seeded.py --seed $seed -- $common --metrics-log-dir runs/rl/adamw_s$seed/tb
  python examples/galore/train_seeded.py --seed $seed -- $common --metrics-log-dir runs/rl/galore_s$seed/tb \
    --galore-rank 128 --galore-update-proj-gap 50 --galore-scale 4
done
python examples/galore/summarize.py --output rl.json --window 10 \
  --rl adamw=runs/rl/adamw_s1,runs/rl/adamw_s2,runs/rl/adamw_s3 \
  --rl galore=runs/rl/galore_s1,runs/rl/galore_s2,runs/rl/galore_s3
```

## Save and resume through the public Trainer

```bash
e2e="--algo gspo --dataset-path $GSM8K_TRAIN ... --galore-rank 128 --galore-update-proj-gap 4 --galore-scale 4"
python examples/galore/train_seeded.py --seed 7 --save-optimizer-state -- $e2e --ckpt $MODEL \
  --max-steps 12 --save-interval 4 --save-path runs/e2e/cont/ckpt --metrics-log-dir runs/e2e/cont/tb
python examples/galore/train_seeded.py --seed 7 --save-optimizer-state --resume-step 4 --resume-prompts 128 -- \
  $e2e --ckpt runs/e2e/cont/ckpt/step_000004 --max-steps 8 --save-interval 8 \
  --save-path runs/e2e/resumed/ckpt --metrics-log-dir runs/e2e/resumed/tb
# cont2 is a second uninterrupted run with the same seed (the noise floor).
python examples/galore/compare_resume.py --continuous runs/e2e/cont --resumed runs/e2e/resumed \
  --control runs/e2e/cont2
python examples/galore/check_state_roundtrip.py --ckpt runs/e2e/cont/ckpt/step_000004 \
  --output /tmp/roundtrip --galore-rank 128 --galore-update-proj-gap 4 --galore-scale 4
```
