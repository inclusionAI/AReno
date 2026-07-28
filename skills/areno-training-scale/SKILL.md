---
name: areno-training-scale
description: Use this skill when calculating training scale, batch sizing, estimating training steps or tokens, or solving gradient accumulation for a target global batch in AReno training runs.
---

# areno Training Scale Calculator

Calculate effective global batch, updates per epoch, total updates, and
approximate processed tokens from dataset size, micro batch, accumulation,
and GPU count.  Also solve accumulation for a target global batch.

## When to Use

- Before starting a training run, to estimate how many steps and tokens it
  will consume.
- When choosing `mini_bs`, `gradient_accumulation_steps`, or GPU layout so
  that the global batch matches a desired value.
- When comparing SFT, DPO, and online RL (GSPO/GRPO/PPO) runs to understand
  why they consume data at different rates.

## Counting Rules

Different training modes count "one sample" differently, which directly
affects step and token calculations:

| Mode | Algorithms | 1 row = | Global batch formula |
|------|-----------|---------|---------------------|
| SFT  | `sft`     | 1 sample | `mini_bs × grad_accum × dp_size` |
| DPO  | `dpo`     | 1 chosen/rejected pair | `mini_bs × grad_accum × dp_size` |
| Online RL | `gspo`, `grpo`, `ppo` | 1 prompt + `n_samples` rollouts | `batch_size` (prompt-level) |

**Key difference**: In online RL, `batch_size` is the number of **prompts**
per step (a global quantity), and each prompt generates `n_samples` rollout
sequences.  The actual number of sequences processed per step is
`batch_size × n_samples`.

**dp_size** (data parallel degree) = `world_size / tp_size`.

## Usage

```bash
# SFT: 10k rows, 2 GPUs, tp=1
python skills/areno-training-scale/scripts/calc_training_scale.py \
    --algo sft --dataset-size 10000 \
    --mini-bs 16 --world-size 2 --tp-size 1 --epochs 3

# GSPO: 500 prompts, 8 samples each
python skills/areno-training-scale/scripts/calc_training_scale.py \
    --algo gspo --dataset-size 500 \
    --batch-size 32 --n-samples 8 --mini-bs 16 \
    --world-size 8 --tp-size 4 --epochs 1

# Solve gradient_accumulation for target global batch = 128
python skills/areno-training-scale/scripts/calc_training_scale.py \
    --algo sft --dataset-size 10000 \
    --mini-bs 16 --world-size 8 --tp-size 4 \
    --target-global-batch 128

# Suggest valid mini_bs/dp_size/grad_accum combos for global batch = 64
python skills/areno-training-scale/scripts/calc_training_scale.py \
    --suggest --target-global-batch 64

# JSON output (for piping into recipe generator or other tools)
python skills/areno-training-scale/scripts/calc_training_scale.py \
    --algo gspo --dataset-size 500 \
    --batch-size 32 --n-samples 8 \
    --world-size 8 --tp-size 4 --json
```

## Input Contract

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--algo` | str | `sft` | Training algorithm: `sft`, `dpo`, `gspo`, `grpo`, `ppo` |
| `--dataset-size` | int | required | Number of rows in the training dataset |
| `--mini-bs` | int | `16` | Backend training microbatch size |
| `--gradient-accumulation-steps` | int | auto | Optimizer step interval; auto-calculated if omitted |
| `--world-size` | int | `8` | Total GPU count |
| `--tp-size` | int | `4` | Tensor parallel size |
| `--batch-size` | int | `32` | Prompt batch size per step (RL only) |
| `--n-samples` | int | `8` | Rollout samples per prompt (RL only) |
| `--epochs` | int | `1` | Number of training epochs |
| `--avg-seq-len` | int | `2048` | Average sequence length for token estimation |
| `--target-global-batch` | int | None | Solve for gradient_accumulation to hit this global batch |
| `--suggest` | flag | off | Print valid mini_bs/dp_size/grad_accum combos |
| `--json` | flag | off | Output as JSON instead of human-readable table |

## Output Fields

| Field | Description |
|-------|-------------|
| `algo` | Algorithm name |
| `dp_size` | Data parallel degree (`world_size / tp_size`) |
| `global_batch` | Effective global batch size |
| `gradient_accumulation_steps` | Resolved gradient accumulation steps |
| `samples_per_step` | Actual samples/sequences processed per step |
| `updates_per_epoch` | Number of optimizer steps per epoch |
| `total_updates` | Total optimizer steps across all epochs |
| `approx_tokens` | Approximate total tokens processed |
| `warnings` | List of warnings (uneven division, mismatch, etc.) |

## Defaults

- `mini_bs=16`, `world_size=8`, `tp_size=4` match AReno CLI defaults.
- `batch_size=32`, `n_samples=8` match AReno RL defaults.
- When `gradient_accumulation_steps` is omitted:
  - SFT/DPO: defaults to `1`.
  - Online RL: auto-calculated to cover the full rollout in one optimizer
    step (`ceil(batch_size * n_samples / (mini_bs * dp_size))`).

## Limitations

- `approx_tokens` is a rough estimate based on `avg_seq_len`.  Actual token
  counts depend on dataset content, prompt length, and (for RL) how many
  tokens the model generates per rollout.
- The calculator does not model dropout rows, padding, or multi-epoch
  shuffling effects.
- For RL algorithms, `batch_size` is treated as a global (not per-GPU)
  quantity, matching AReno's CLI semantics.
- The `--suggest` mode uses a fixed range of `mini_bs` and `dp_size` values;
  it may not find combos outside that range.

## One Copyable Example

```bash
python skills/areno-training-scale/scripts/calc_training_scale.py \
    --algo gspo --dataset-size 500 \
    --batch-size 32 --n-samples 8 \
    --mini-bs 16 --world-size 8 --tp-size 4 \
    --epochs 1 --avg-seq-len 2048 --json
```

Output:

```json
{
  "algo": "gspo",
  "dp_size": 2,
  "global_batch": 32,
  "gradient_accumulation_steps": 8,
  "updates_per_epoch": 16,
  "total_updates": 16,
  "approx_tokens": 8388608,
  "samples_per_step": 256,
  "warnings": []
}
```

## Integration with Recipe Generator

The `--json` output is designed to feed into the training-recipe generator
(Issue #273).  Pipe or redirect the JSON result as input for recipe
generation:

```bash
python skills/areno-training-scale/scripts/calc_training_scale.py \
    --algo sft --dataset-size 10000 --mini-bs 16 \
    --world-size 8 --tp-size 4 --json \
    > /tmp/scale_result.json

# Future: feed into recipe generator
# python skills/areno-recipe/scripts/gen_recipe.py \
#     --scale-result /tmp/scale_result.json
```