#!/usr/bin/env bash
# Short MLX smoke for OPD on Apple Silicon.
# Checks teacher-role loading, per-step teacher scoring, the native MLX
# pg/k3 losses and the held-out logprob-gap benchmark.
set -euo pipefail

cd "$(dirname "$0")/../.."
OUT="${ARENO_OPD_MLX_OUT:-./opd_mlx_smoke}"
mkdir -p "$OUT"

areno train \
  --backend mlx \
  --algo opd \
  --ckpt mlx-community/Qwen3-0.6B-Base \
  --teacher-ckpt Qwen/Qwen3-1.7B-MLX-bf16 \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --validation-dataset-path gsm8k:main:test \
  --validation-dataset-loader-fn examples/math/dataset_loader.py \
  --eval-interval 5 --eval-batches 2 \
  --batch-size 1 --n-samples 1 --mini-bs 1 \
  --score-micro-bs 1 --opd-teacher-micro-bs 1 \
  --max-prompt-tokens 256 --max-new-tokens 128 \
  --max-running-prompts 1 \
  --lr 1e-5 \
  --lora-rank 8 --lora-alpha 16 \
  --opd-loss-type "${ARENO_OPD_MLX_LOSS:-pg}" \
  --disable-thinking \
  --drop-rollout-state \
  --max-steps "${ARENO_OPD_MLX_STEPS:-5}" \
  --save-interval 5 \
  --save-path "$OUT/ckpt" \
  --metrics-log-dir "$OUT/logs"

echo "Metrics and peak-memory logs: $OUT/logs"
