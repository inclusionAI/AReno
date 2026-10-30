# Activation compression experiments

Scripts that reproduce the measurements in
`docs/concepts/activation-compression-results.rst`. They drive the public
`Trainer` (directly or through `areno train`) on one GPU each. The usage guide
is `docs/concepts/activation-compression.rst`.

| Script | Purpose |
| --- | --- |
| `sft_validation.py` | Fixed-budget SFT with a fixed held-out validation split, scored by the live actor; also the memory and step-time benchmark. |
| `train_seeded.py` | `areno train` with per-step rollout seeds and resume from a saved checkpoint. |
| `gradient_error.py` | Gradient error of a checkpoint's saved plan on held-out batches, against the uncompressed gradients. |
| `microbench.py` | Pack and unpack time of the CUDA kernels for one saved tensor. |
| `summarize.py` | Memory, step-time, saved-activation and calibration tables, final validation loss, final-window reward. |

## Raw results

`results/` holds the logs behind every table:

- `bench/<config>_{bf16,actc}/`: per-step benchmark metrics (`metrics.jsonl.gz`) and, for the
  compressed arm, every calibration the run logged (`calibration.json`: plan, per-slot
  sensitivities, closed-loop rounds, errors);
- `sft/<arm>_s<seed>/`: per-step SFT metrics and validation losses, calibrations;
  `sft/gradient_error/`: the `gradient_error.py` reports;
- `rl/<arm>_s<seed>/`: RL TensorBoard scalars (`scalars.jsonl.gz`) and calibrations, including the
  resumed runs (`*_s1_resume`);
- `*_summary.json`: the tables, produced by `summarize.py` from these files.

Each group's `runs.json` records, per run, the source commit and `areno/` tree, the arguments
with model and dataset SHA-256 digests and software versions, or the configuration printed by
`areno train`. SFT arms: `bf16` (baseline), `actc` (adaptive, 1% budget, plan checked every 50
steps), `actc_once` (calibrated once), `actc05` (5% budget), `q17_*` (Qwen3-1.7B). RL arms: `bf16`,
`actc` (calibrated once) and `actcfinal` (plan checked every 50 steps). Paths in these files use
the names of the commands below: `$MODEL`, `$MODEL_1_7B`, `$ALPACA` and `$GSM8K_TRAIN` for the
model and data, and `runs/<group>/<run>` for each run's directory, under the name it has here (the
calibrated-once SFT runs ran as `actc_s<seed>` and were renamed before the checked runs were
added). `summarize.py` reads the gzipped logs directly, for example:

```bash
R=examples/activation_compression/results
python examples/activation_compression/summarize.py --output sft.json \
  --sft bf16=$R/sft/bf16_s1,$R/sft/bf16_s2,$R/sft/bf16_s3 \
  --sft adaptive=$R/sft/actc_s1,$R/sft/actc_s2,$R/sft/actc_s3
```

## Data and model

- Model: `Qwen/Qwen3-0.6B` (ModelScope, `master`), and `Qwen/Qwen3-1.7B` for one benchmark row.
- SFT data: `yahma/alpaca-cleaned` (`alpaca_data_cleaned.json`, 51,760 rows),
  normalized by `examples/sft/alpaca/dataset_loader.py`.
- RL data: GSM8K `main` train split (`train-00000-of-00001.parquet`), normalized
  by `examples/math/dataset_loader.py` and scored by
  `examples/math/math_verify_reward.py` (0/1 reward).

## Memory and step-time benchmark

Both arms of a pair see identical batches (`--seed 1`). Steps 1 and 2 are
warm-up (step 1 of the adaptive arm also calibrates); steps 3 to 14 are
measured. `$MODEL` is Qwen3-0.6B except for the last line.

```bash
common="--dataset-path $ALPACA --seed 1 --steps 14 --val-rows 0"
for arm in bf16 actc; do
  extra=""; [ "$arm" = actc ] && extra="--compression adaptive"
  python examples/activation_compression/sft_validation.py --ckpt $MODEL $common $extra \
    --batch-size 128 --mini-bs 64 --output runs/bench/q06_ckpt_mb64_$arm
  python examples/activation_compression/sft_validation.py --ckpt $MODEL $common $extra \
    --batch-size 512 --mini-bs 256 --output runs/bench/q06_ckpt_mb256_$arm
  python examples/activation_compression/sft_validation.py --ckpt $MODEL $common $extra \
    --batch-size 1024 --mini-bs 512 --output runs/bench/q06_ckpt_mb512_$arm
  python examples/activation_compression/sft_validation.py --ckpt $MODEL $common $extra \
    --batch-size 128 --mini-bs 64 --no-activation-checkpointing --output runs/bench/q06_nockpt_mb64_$arm
  python examples/activation_compression/sft_validation.py --ckpt $MODEL_1_7B $common $extra \
    --batch-size 512 --mini-bs 256 --output runs/bench/q17_ckpt_mb256_$arm
done
# Plan over a longer run: 90 steps, checks every 30 steps.
python examples/activation_compression/sft_validation.py --ckpt $MODEL --dataset-path $ALPACA --seed 1 \
  --steps 90 --val-rows 0 --batch-size 512 --mini-bs 256 --compression adaptive --check-interval 30 \
  --output runs/bench/q06_ckpt_mb256_actc_long
python examples/activation_compression/microbench.py --output microbench.json
```

## SFT validation runs

The validation split is the first 1,000 in-budget rows of a fixed permutation
(`--split-seed 1234`); each `--seed` permutes the remaining rows. Every arm uses
300 steps of 128 rows in two microbatches of 64, cosine learning rate 1e-5 to
1e-6, no weight decay, prompt and response limits of 256 and 512 tokens, and
activation checkpointing.

```bash
for seed in 1 2 3; do
  common="--ckpt $MODEL --dataset-path $ALPACA --seed $seed --steps 300 --batch-size 128 --mini-bs 64"
  python examples/activation_compression/sft_validation.py $common --output runs/sft/bf16_s$seed
  python examples/activation_compression/sft_validation.py $common --compression adaptive \
    --save-at 300 --output runs/sft/actc_s$seed
  python examples/activation_compression/sft_validation.py $common --compression adaptive \
    --check-interval 0 --save-at 300 --output runs/sft/actc_once_s$seed
  python examples/activation_compression/sft_validation.py $common --compression adaptive \
    --error-budget 0.05 --output runs/sft/actc05_s$seed
done
common="--ckpt $MODEL_1_7B --dataset-path $ALPACA --seed 1 --steps 300 --batch-size 128 --mini-bs 64"
python examples/activation_compression/sft_validation.py $common --output runs/sft/q17_bf16_s1
python examples/activation_compression/sft_validation.py $common --compression adaptive \
  --output runs/sft/q17_actc_s1
```

The calibrated-once runs were recorded at an earlier commit, where
`--check-interval 0` was called `--calibration-interval 0` (`runs.json`). The
later changes do not affect them: with recompute no slot is pinned as an alias,
and none of their calibration rounds was over budget, which is when bisection
starts. Gradient errors of the saved plans on held-out batches
(`results/sft/gradient_error/`):

```bash
E="--dataset-path $ALPACA --mini-bs 64"
G=examples/activation_compression/gradient_error.py
python $G --ckpt runs/sft/actc_s1/step_000300 $E --batches 8 --output actc_s1_step300_saved_plan.json
python $G --ckpt runs/sft/actc_once_s1/step_000300 $E --batches 8 --output actc_once_s1_step300_saved_plan.json
python $G --ckpt runs/sft/actc_once_s1/step_000300 $E --batches 8 --plan 8 --output actc_once_s1_step300_uniform8.json
python $G --ckpt runs/sft/actc_once_s1/step_000300 $E --batches 8 --plan 4 --output actc_once_s1_step300_uniform4.json
# The base model with the calibrated-once run's plan, which is its step-1 plan.
mkdir -p runs/sft/base_with_plan && ln -s "$MODEL"/* runs/sft/base_with_plan/
cp runs/sft/actc_once_s1/step_000300/activation_compression.json runs/sft/base_with_plan/
python $G --ckpt runs/sft/base_with_plan $E --batches 4 --output actc_once_s1_plan_on_base_model.json
```

## RL runs

```bash
common="--algo gspo --ckpt $MODEL --dataset-path $GSM8K_TRAIN \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --world-size 1 --tp-size 1 --batch-size 32 --n-samples 8 --max-running-prompts 128 \
  --max-prompt-tokens 512 --max-new-tokens 1024 --disable-thinking --max-steps 100"
for seed in 1 2 3; do
  python examples/activation_compression/train_seeded.py --seed $seed -- $common \
    --metrics-log-dir runs/rl/bf16_s$seed/tb
  python examples/activation_compression/train_seeded.py --seed $seed -- $common \
    --activation-compression adaptive --metrics-log-dir runs/rl/actc_s$seed/tb
done
python examples/activation_compression/summarize.py --output rl.json --window 10 \
  --rl bf16=runs/rl/bf16_s1,runs/rl/bf16_s2,runs/rl/bf16_s3 \
  --rl adaptive=runs/rl/actc_s1,runs/rl/actc_s2,runs/rl/actc_s3
```

## Save and resume through the public Trainer

Seed 1 of each arm also saves every 50 steps (`--save-interval 50 --save-path
runs/rl/<arm>_s1/ckpt`). A second process then resumes from the step-50
checkpoint for the remaining 50 steps; the compressed resume reuses the
checkpoint's plan instead of calibrating again:

```bash
python examples/activation_compression/train_seeded.py --seed 1 --resume-step 50 --resume-prompts 1600 -- \
  $common --activation-compression adaptive --ckpt runs/rl/actc_s1/ckpt/step_000050 --max-steps 50 \
  --metrics-log-dir runs/rl/actc_s1_resume/tb
```
