#!/usr/bin/env bash
# SFT training script for Tic-Tac-Toe.
#
# Prerequisites:
#   1. Generate the dataset:
#        python examples/agentic/tictactoe/gen_sft_data.py \
#          --output examples/agentic/tictactoe/dataset.jsonl \
#          --num-samples 5000
#   2. CUDA GPU available (AReno requires CUDA for training).
#
# Usage:
#   bash examples/agentic/tictactoe/train_sft.sh [MODEL_CKPT] [SAVE_PATH]
#
# Defaults:
#   MODEL_CKPT = Qwen/Qwen3-0.6B
#   SAVE_PATH  = ./tictactoe-sft

set -euo pipefail

MODEL_CKPT="${1:-Qwen/Qwen3-0.6B}"
SAVE_PATH="${2:-./tictactoe-sft}"
DATA_PATH="examples/agentic/tictactoe/dataset.jsonl"
LOADER_FN="examples/agentic/tictactoe/dataset_loader.py"

echo "=== Tic-Tac-Toe SFT Training ==="
echo "Model:       $MODEL_CKPT"
echo "Dataset:     $DATA_PATH"
echo "Save path:   $SAVE_PATH"
echo ""

areno train \
  --algo sft \
  --ckpt "$MODEL_CKPT" \
  --dataset-path "$DATA_PATH" \
  --dataset-loader-fn "$LOADER_FN" \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --mini-bs 1 \
  --gradient-accumulation-steps 4 \
  --activation-checkpointing \
  --adam-8bit \
  --max-prompt-tokens 128 \
  --max-new-tokens 16 \
  --epochs 3 \
  --lr 1e-5 \
  --save-path "$SAVE_PATH" \
  --save-interval 100
