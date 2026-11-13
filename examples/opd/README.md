# OPD example

On-Policy Distillation (OPD) trains a student on the responses it just sampled,
using a frozen teacher to score every response token. Unlike GRPO/GSPO/PPO it
does not need a reward function: the dense signal is the teacher logprob on the
sampled token.

This example verifies the full wiring (`CLI -> rollout -> teacher scoring ->
loss -> optimizer step`). It is a smoke test, not a quality benchmark. Use the
held-out evaluation flags described below when you need before/after numbers.

## Prerequisites

- CUDA backend (OPD is declared in the CUDA backend capabilities; NPU inherits
  the same capability set).
- `areno check` passes.
- Student and teacher must share the **same tokenizer**. The trainer checks
  vocab size, `pad_token_id` and a probe encoding at start-up and aborts on a
  mismatch. `eos`/`bos` are deliberately not compared: the teacher never
  generates text, it only scores the token ids sampled by the student. This
  matters for Qwen3, where the `-Base` checkpoint uses a different eos id
  from the post-trained checkpoint of the same family.

## Smoke run

```bash
areno check

areno train \
  --algo opd \
  --ckpt Qwen/Qwen3-0.6B \
  --teacher-ckpt Qwen/Qwen3-0.6B-Instruct \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --mini-bs 1 \
  --max-prompt-tokens 256 --max-new-tokens 256 \
  --max-steps 20 \
  --metrics-log-dir ./outputs/opd-smoke
```

Note there is no `--reward-fn-path`. `opd` is registered with
`requires_rollout=True` and `requires_reward=False`, so the CLI does not ask
for a reward function.

Or simply:

```bash
sh examples/opd/run_smoke.sh
```

Expected output: rollout logs, `train_stats=...` lines, and the `opd_loss`,
`opd_advantage_mean`, `opd_teacher_logprob_mean`, `opd_teacher_student_gap` and
`opd_clipfrac` metrics with no exception.

## Held-out evaluation

OPD reuses the existing `--eval-*` options and reports two kinds of metrics on
the same schedule.

### Task accuracy (generation)

Pass a verifier through `--reward-fn-path`. OPD has no reward term during
training, so this callable is used **only** when evaluation fires: the trainer
samples held-out prompts with the current student policy and scores the
generated completions through the standard `reward_fn(record)` contract.
`examples/math/math_verify_reward.py` works as-is for GSM8K-style data.

```bash
areno train \
  --algo opd \
  --ckpt Qwen/Qwen3-0.6B \
  --teacher-ckpt Qwen/Qwen3-0.6B-Instruct \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --validation-dataset-path gsm8k:main \
  --validation-dataset-loader-fn examples/math/dataset_loader.py \
  --eval-interval 5 \
  --eval-batches 1 \
  --eval-before-step \
  --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --mini-bs 1 \
  --max-steps 20
```

This reports `generation_accuracy`, `generation_reward_mean` and
`generation_samples`. With a binary verifier, accuracy is the task-level number
to compare before and after training. Generation uses one greedy completion per
prompt, so repeated evaluations are comparable and `generation_samples` counts
evaluated questions. The training-time `--n-samples` does not inflate evaluation.

### Distillation progress (logprob gap)

Even without a verifier, the trainer scores fixed held-out sequences with both
the student and the frozen teacher and reports `teacher_student_gap`, per-token
NLL and related values. This is a cheap training-progress monitor, not a task
metric.

All metrics go to `validation_metrics.jsonl` under `--metrics-log-dir`,
prefixed with `before_` / `after_` for the evaluation phase. `--eval-batches`
limits the number of batches (each of `--batch-size` rows). The dataset loader
from `examples/math/` is reused as-is; held-out rows need `prompt` plus
`solutions` (or `answer`).

## Uniform checkpoint benchmark

To compare checkpoints from different algorithms on exactly the same held-out
questions, load a saved PEFT adapter through `--lora-adapter-path` and run a
one-step OPD job with `--eval-before-step`. The `before_` row is scored before
any optimizer update, so the adapter weights are used exactly as saved. This
keeps the generation engine, prompt encoding and verifier identical across
OPD-pg, OPD-k3 and a plain SFT baseline.

```bash
sh examples/opd/benchmark_checkpoint.sh ./outputs/opd_ckpt/step_000300 pg300
sh examples/opd/benchmark_checkpoint.sh ./outputs/sft_ckpt/step_000300 sft300
sh examples/opd/benchmark_checkpoint.sh none base
```

The script uses `--batch-size 8 --eval-batches 8`, i.e. the same 64 questions
but generated in parallel batches, which is considerably faster than scoring
them one at a time. Read `before_generation_accuracy` from
`validation_metrics.jsonl` under the output directory. Override
`STUDENT_CKPT`, `TEACHER_CKPT`, `EVAL_BATCH_SIZE`, `EVAL_BATCHES` or `OUT_DIR`
to adapt the run.

## SFT baseline

`examples/opd/sft_baseline_loader.py` converts the same GSM8K rows into the
`prompt` / `response` pairs that `--algo sft` expects. Its prompt is
byte-identical to `examples/math/dataset_loader.py` (pinned by
`tests/test_example_loaders_cpu.py`), and the response keeps the full rationale
rather than only the final value, so the baseline answers in the same style the
verifier rewards.

```bash
areno train \
  --algo sft \
  --ckpt Qwen/Qwen3-0.6B-Base \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/opd/sft_baseline_loader.py \
  --tp-size 1 --world-size 1 \
  --batch-size 1 --mini-bs 1 \
  --lr 1e-5 \
  --max-prompt-tokens 512 --max-new-tokens 512 \
  --lora-rank 8 --lora-alpha 16 \
  --max-steps 300 \
  --save-path ./outputs/sft_ckpt --save-interval 100 \
  --metrics-log-dir ./outputs/sft
```

Score `outputs/sft_ckpt/step_000300` with `benchmark_checkpoint.sh` exactly as
an OPD checkpoint. Supervising a fixed answer set peaks early and then degrades
as the policy keeps fitting those answers, which is the contrast the OPD runs
are meant to show.

## MLX (Apple Silicon)

MLX uses the same `opd` spec, `OPDTrainer` and `teacher` role as CUDA. The
backend registers both, and `areno/api/backend/mlx/losses.py` supplies native
`pg` and `k3` losses. MLX has no role offload, so the student and the frozen
teacher stay resident at the same time.

The two BF16 MLX checkpoints below match the CUDA benchmark models (0.6B Base
student + 1.7B post-trained teacher) and need about 4.6 GB of weights in
total. With LoRA r8 and the small batches in the smoke script, a 16 GB Mac can
run this configuration; check the peak-memory metric on the first step before
increasing the sequence length or sample count.

Install the MLX runtime following `docs/getting-started/mlx.rst`, then run:

```bash
python -m pytest tests/test_opd_mlx_numerics_cpu.py -q
bash examples/opd/run_mlx_smoke.sh
```

The numerical test compares MLX losses and gradients with NumPy closed forms;
it skips on hosts without `mlx.core`. The smoke script runs 5 steps with 2
held-out rows and 128 new tokens. Override with `ARENO_OPD_MLX_STEPS=50` for a
short training run and `ARENO_OPD_MLX_LOSS=k3` for the other loss. Check that
`opd_teacher_student_gap` stays finite across the run.

Two MLX-specific constraints:

- Do not add `--qlora`: the MLX backend rejects QLoRA together with auxiliary
  model roles. Use BF16 LoRA instead.
- If generated text contains `<think>` blocks, confirm `--disable-thinking`
  took effect. `Qwen3-1.7B-MLX-bf16` enables thinking by default, which would
  make prompts and outputs differ from the CUDA benchmark.

The full 300-step GSM8K benchmark belongs on CUDA; MLX verification is about
backend correctness, not reproducing the accuracy numbers.

## OPD flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--teacher-ckpt` | required | Frozen teacher checkpoint or remote repo id |
| `--opd-coef` | 1.0 | Distillation loss scale |
| `--opd-loss-type` | pg | `pg` (clipped dense advantage) or `k3` (low-variance KL) |
| `--opd-clip` | 5.0 | Advantage clipping range for `pg` |
| `--opd-teacher-micro-bs` | `--score-micro-bs` | Teacher scoring microbatch size |
| `--reward-fn-path` | none | Verifier used only by the held-out generation benchmark; OPD training has no reward term |

## Troubleshooting

| Symptom | Cause |
| --- | --- |
| `OPD teacher and student vocab sizes differ` | Teacher is from a different model family; use a matching tokenizer |
| `OPD requires teacher logprobs in packed_ref_logprobs` | The teacher role did not score this batch (MLX words the same condition as `ref_logprobs`); check that `--teacher-ckpt` was set and the backend supports the role |
| `teacher logprobs contain non-finite values` | Teacher checkpoint is broken or the sequence contains unsupported tokens |
| Loss stays exactly zero | Student and teacher produce identical logprobs; this is expected with an identical checkpoint and no LoRA |
| No `generation_accuracy` in the metrics | `--reward-fn-path` was not supplied; only the teacher/student logprob gap is reported |
| `generation benchmark produced no scored samples` | The held-out set has no usable `prompt` / `solutions` rows, or all prompts exceeded `--max-prompt-tokens` |
| Out of memory during scoring | Lower `--opd-teacher-micro-bs`; CUDA offloads the teacher between passes, while MLX keeps it resident and also needs smaller `--max-new-tokens` / `--n-samples` |

On CUDA the frozen teacher is loaded lazily by the framework's role manager and
offloaded after each scoring pass, so no manual offload flag is needed. MLX has
no role offload; both models stay resident for the whole run.
