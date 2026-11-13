#!/bin/sh
# Uniform held-out generation benchmark for any saved LoRA adapter.
#
# Usage:
#   sh examples/opd/benchmark_checkpoint.sh <adapter_dir|none> <tag>
#
# The adapter is loaded read-only through --lora-adapter-path. Evaluation fires
# before the first optimizer step, so the `before_*` row describes the adapter
# exactly as saved. --max-steps 1 keeps the run short; the single update is
# harmless because only the `before_` metrics are used.
#
# Override with environment variables when needed:
#   STUDENT_CKPT, TEACHER_CKPT, EVAL_BATCH_SIZE, EVAL_BATCHES, OUT_DIR
# EVAL_BATCH_SIZE * EVAL_BATCHES is the number of held-out questions (64 here).

set -eu

ADAPTER=${1:-none}
TAG=${2:-benchmark}
STUDENT_CKPT=${STUDENT_CKPT:-Qwen/Qwen3-0.6B-Base}
TEACHER_CKPT=${TEACHER_CKPT:-Qwen/Qwen3-1.7B}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-8}
EVAL_BATCHES=${EVAL_BATCHES:-8}
OUT_DIR=${OUT_DIR:-./outputs/opd_benchmark_$TAG}

set -- \
  --algo opd \
  --ckpt "$STUDENT_CKPT" \
  --lora-rank 8 --lora-alpha 16 \
  --teacher-ckpt "$TEACHER_CKPT" \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --reward-fn-path examples/math/math_verify_reward.py \
  --validation-dataset-path gsm8k:main:test \
  --validation-dataset-loader-fn examples/math/dataset_loader.py \
  --eval-interval 1 --eval-batches "$EVAL_BATCHES" --eval-before-step \
  --tp-size 1 --world-size 1 \
  --batch-size "$EVAL_BATCH_SIZE" --n-samples 2 --mini-bs 1 \
  --max-prompt-tokens 512 --max-new-tokens 512 \
  --max-running-prompts "$EVAL_BATCH_SIZE" \
  --drop-rollout-state --disable-thinking \
  --attn-backend native \
  --max-steps 1 \
  --metrics-log-dir "$OUT_DIR"

if [ "$ADAPTER" != "none" ]; then
  set -- "$@" --lora-adapter-path "$ADAPTER"
fi

areno train "$@"

echo "Result: $OUT_DIR/validation_metrics.jsonl (read before_generation_accuracy)"
