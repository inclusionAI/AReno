# Qwen3.5-VL Tic-Tac-Toe Image Example

This example trains Qwen3.5-VL on synthetic tic-tac-toe board images. It keeps
the dataset loader model-agnostic: the loader returns text fields and
`image_base64`; AReno encodes the images with the processor from `--ckpt`.

## Files

- `dataset_generator.py` draws board images with Python and Pillow and writes a
  JSONL manifest.
- `dataset_loader.py` loads JSONL rows and returns `image_base64`, `prompt`,
  `response`, and reward metadata.
- `game.py` contains board validation, BFS best moves, and scoring.
- `run_agent.py` sends image requests through the OpenAI-compatible rollout
  proxy for agentic GSPO/PPO.
- `reward.py` scores RL completions with the same best-move logic as the
  Tic-Tac-Toe agentic example.

## Generate Data

```bash
python examples/vl/tictactoe_image/dataset_generator.py \
  --output /tmp/areno-vl-tictactoe/dataset.jsonl \
  --count 128
```

Each JSONL row stores the PNG as `image_base64`, so the dataset is self-contained.
You can also pass `--image-dir /tmp/areno-vl-tictactoe/images` to write preview
PNGs; the JSONL will still include `image_base64`. If the dataset path does not
exist, the loader generates a tiny in-memory demo dataset.

## SFT

```bash
areno train \
  --ckpt Qwen/Qwen3.5-0.8B \
  --dataset-path /tmp/areno-vl-tictactoe/dataset.jsonl \
  --dataset-loader-fn examples/vl/tictactoe_image/dataset_loader.py \
  --algo sft \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --mini-bs 1 \
  --epochs 1 \
  --max-new-tokens 128 \
  --max-prompt-tokens 8192
```

## GSPO / GRPO

```bash
areno train \
  --ckpt Qwen/Qwen3.5-0.8B \
  --dataset-path /tmp/areno-vl-tictactoe/dataset.jsonl \
  --dataset-loader-fn examples/vl/tictactoe_image/dataset_loader.py \
  --reward-fn-path examples/vl/tictactoe_image/reward.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 4 \
  --mini-bs 1 \
  --epochs 1 \
  --max-running-prompts 4 \
  --max-new-tokens 128 \
  --max-prompt-tokens 8192 \
  --drop-rollout-state
```

## PPO

```bash
areno train \
  --ckpt Qwen/Qwen3.5-0.8B \
  --dataset-path /tmp/areno-vl-tictactoe/dataset.jsonl \
  --dataset-loader-fn examples/vl/tictactoe_image/dataset_loader.py \
  --reward-fn-path examples/vl/tictactoe_image/reward.py \
  --algo ppo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 4 \
  --mini-bs 1 \
  --score-micro-bs 1 \
  --epochs 1 \
  --max-running-prompts 4 \
  --max-new-tokens 128 \
  --max-prompt-tokens 8192 \
  --drop-rollout-state
```

## Agentic GSPO

The agentic variant sends the board as an OpenAI `image_url` data URL. It
supports rows with either `image_base64` or `image_path`; `run_agent.py`
normalizes `image_path` into a data URL before calling the rollout proxy.

```bash
areno train \
  --ckpt Qwen/Qwen3.5-0.8B \
  --dataset-path /tmp/areno-vl-tictactoe/dataset.jsonl \
  --dataset-loader-fn examples/vl/tictactoe_image/dataset_loader.py \
  --reward-fn-path examples/vl/tictactoe_image/reward.py \
  --agent-fn examples/vl/tictactoe_image/run_agent.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 4 \
  --mini-bs 1 \
  --epochs 1 \
  --max-running-prompts 4 \
  --max-new-tokens 128 \
  --max-prompt-tokens 8192 \
  --drop-rollout-state
```
