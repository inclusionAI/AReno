#!/usr/bin/env bash
# Minimal end-to-end OPD smoke run. See README.md for prerequisites.
set -euo pipefail

cd "$(dirname "$0")/../.."

areno check

areno train \
  --algo opd \
  --ckpt Qwen/Qwen3-0.6B \
  --teacher-ckpt Qwen/Qwen3-0.6B-Instruct \
  --dataset-path gsm8k:main \
  --dataset-loader-fn examples/math/dataset_loader.py \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 2 \
  --mini-bs 1 \
  --max-prompt-tokens 256 \
  --max-new-tokens 256 \
  --max-steps 20 \
  --metrics-log-dir ./outputs/opd-smoke
