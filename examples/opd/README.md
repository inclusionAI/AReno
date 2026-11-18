# OPD example

On-Policy Distillation (OPD) trains a student on the responses it just sampled,
using a frozen teacher to score every response token. Unlike GRPO/GSPO/PPO it
does not need a reward function: the dense signal is the teacher logprob on the
sampled token.

The smoke run verifies the full wiring (`CLI -> rollout -> teacher scoring ->
loss -> optimizer step`). The benchmark sections below reproduce the 300-step
GSM8K numbers on a single A100.

## Prerequisites

`areno check` must pass. Student and teacher must share the **same tokenizer**;
the trainer compares vocab size, `pad_token_id` and probe encodings at start-up.
`eos`/`bos` are not compared, because the teacher only scores token ids and never
generates. Backend requirements are in the
[OPD cookbook](../../docs/cookbook/opd-training.rst).

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

Two metrics share the `--eval-*` schedule. `generation_accuracy` samples greedy
completions and scores them through `--reward-fn-path`, which is used only here
and never in the training loss; `teacher_student_gap` teacher-forces the same
prompts through the student and the frozen teacher and needs no verifier.

Commands, flag semantics, the `validation_metrics.jsonl` layout and the
`gsm8k:main:test` requirement are in the
[OPD cookbook](../../docs/cookbook/opd-training.rst). The gap is a training
monitor, not a task result; the benchmark below uses accuracy.

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
STUDENT_CKPT=Qwen/Qwen3-1.7B sh examples/opd/benchmark_checkpoint.sh none teacher
```

The script uses `--batch-size 8 --eval-batches 8`, i.e. the same 64 questions
but generated in parallel batches, which is considerably faster than scoring
them one at a time. Read `before_generation_accuracy` from
`validation_metrics.jsonl` under the output directory. Override
`STUDENT_CKPT`, `TEACHER_CKPT`, `EVAL_BATCH_SIZE`, `EVAL_BATCHES` or `OUT_DIR`
to adapt the run.

## Reproducing the 300-step results

The table below comes from one run per loss type plus the SFT control. All three
share the student, the teacher, the data order, the learning rate and the
evaluation points.

| Step | OPD-pg | OPD-k3 | SFT |
| ---: | -----: | -----: | --: |
| 0 | 1.56% | 1.56% | 1.56% |
| 100 | 62.50% | 62.50% | 64.06% |
| 200 | 67.19% | 64.06% | 46.88% |
| 300 | 60.94% | 57.81% | 35.94% |

Every cell of the table is a `benchmark_checkpoint.sh` pass over the same 64
GSM8K test prompts at `--batch-size 8`. One question is worth 1.56 points, and
the 95% Wilson interval at this sample size is about ±11 points.

Training-side metrics, averaged over 60-step windows (SFT has no teacher, so no
`opd_loss` to report):

| Window | pg `opd_loss` | pg gap | k3 `opd_loss` | k3 gap |
| ---: | ---: | ---: | ---: | ---: |
| 0-59 | -8.17 | -3.03 | 2.53 | -3.18 |
| 60-119 | -2.66 | -1.36 | 1.97 | -2.40 |
| 120-179 | -1.01 | -1.02 | 1.00 | -1.07 |
| 180-239 | -0.48 | -0.70 | 0.77 | -0.81 |
| 240-299 | -0.30 | -0.62 | 0.76 | -0.80 |

Gap is the mean of `teacher logprob - student logprob` over each window's
training rollout. `opd_loss` uses a different scale in each objective, so read it
down a column rather than across a row. Gradient norm falls from 12.0 to 3.9 (pg)
and 3.8 to 3.5 (k3).

Two more numbers from the same windows: pg `opd_clipfrac` falls from 16.7% to
4.4%, so the default `--opd-clip 5` binds mostly in the first 60 steps (k3 has no
clipping and reports zero); mean response length stays between 230 and 310
tokens, so the accuracy gain is not a length effect.

The teacher itself, scored by the same script on the same 64 prompts, reaches
76.56% (49/64): the reference the student is distilling toward. OPD-pg peaks at
67.19%, so it recovers most of the headroom from the 1.56% base without ever
seeing a gold answer.

Hardware and settings: one A100-SXM4-80GB, BF16, LoRA r8 / alpha 16, `native`
attention, 300 steps at `--batch-size 1 --n-samples 2`, lr 1e-5. Teacher scoring
accounted for about 21% of step time, and a 300-step run took roughly 100
minutes on a card that was also serving other jobs.

```bash
areno train \
  --algo opd \
  --ckpt Qwen/Qwen3-0.6B-Base \
  --teacher-ckpt Qwen/Qwen3-1.7B \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --validation-dataset-path gsm8k:main:test \
  --validation-dataset-loader-fn examples/math/dataset_loader.py \
  --eval-interval 50 --eval-batches 64 \
  --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --mini-bs 1 \
  --lr 1e-5 \
  --max-prompt-tokens 512 --max-new-tokens 512 \
  --max-running-prompts 2 \
  --lora-rank 8 --lora-alpha 16 \
  --drop-rollout-state --disable-thinking \
  --attn-backend native \
  --max-steps 300 \
  --save-path ./outputs/opd_ckpt --save-interval 100 \
  --metrics-log-dir ./outputs/opd
```

For the `k3` arm add `--opd-loss-type k3` and point `--save-path` at
`./outputs/opd_k3_ckpt`. The SFT control uses the command in
[SFT baseline](#sft-baseline) unchanged.

The run above evaluates every 50 steps with `--batch-size 1`; those points are a
training monitor. The table re-scores each saved adapter with
`benchmark_checkpoint.sh`, which puts every algorithm on the same generation
batch, so an intermediate step can differ from the in-training number by a
question or two. On a shared GPU the runs also capped the process with
`--mem-frac` (0.2-0.3); leave it out on an exclusive card.

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

The two BF16 checkpoints used by `run_mlx_smoke.sh`
(`mlx-community/Qwen3-0.6B-Base` and `Qwen/Qwen3-1.7B-MLX-bf16`) match the CUDA
benchmark models and need about 4.6 GB of weights in total. With LoRA r8 and the
small batches in the smoke script, a 16 GB Mac can run this configuration; check
the peak-memory metric on the first step before increasing the sequence length or
sample count.

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

## See also

- [OPD concept guide](../../docs/concepts/opd.rst): objective, tokenizer
  contract, scope.
- [OPD cookbook](../../docs/cookbook/opd-training.rst): flags, held-out
  evaluation commands, troubleshooting.
- [Training CLI reference](../../docs/cli/training.rst): every flag and default.
