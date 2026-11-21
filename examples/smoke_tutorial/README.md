# Smoke Tutorial

A minimal, self-contained example for verifying that AReno's training pipeline
works correctly on your machine. This tutorial uses a tiny synthetic dataset and
a simple heuristic-based reward function.

## Quick Start

### 1. Generate the dataset

Create a small JSONL file with sample prompts:

```bash
cat > /tmp/smoke_data.jsonl << 'EOF'
{"prompt": "Explain what is machine learning in simple terms.", "reference": "Machine learning is a subset of AI where systems learn from data."}
{"prompt": "What are the benefits of exercise?", "reference": "Exercise improves physical health, mental well-being, and longevity."}
{"prompt": "How does photosynthesis work?", "reference": "Plants convert sunlight, water, and CO2 into glucose and oxygen."}
{"prompt": "Explain the concept of recursion.", "reference": "Recursion is when a function calls itself to solve smaller subproblems."}
{"prompt": "What is the water cycle?", "reference": "Water evaporates, forms clouds, precipitates, and collects back to bodies of water."}
EOF
```

### 2. Run the smoke test

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/smoke_data.jsonl \
  --dataset-loader-fn examples/smoke_tutorial/dataset_loader.py \
  --reward-fn-path examples/smoke_tutorial/reward.py \
  --algo gspo \
  --tp-size 1 \
  --world-size 1 \
  --batch-size 1 \
  --n-samples 2 \
  --max-prompt-tokens 128 \
  --max-new-tokens 256
```

### 3. Expected output

A successful run will show:
- Dataset loading with 5 samples
- Rollout generation for each prompt
- Reward scores based on response length/quality
- Training statistics (loss, gradients, etc.)

## What This Tests

- **Dataset loading**: JSONL parsing and field normalization
- **Reward function**: Basic scoring without external dependencies
- **Training loop**: Single-step RL training with GSPO
- **CLI integration**: End-to-end command execution

## Files

- `dataset_loader.py`: Minimal dataset loader supporting JSONL and plain text
- `reward.py`: Heuristic-based reward function (length + reasoning markers)
- `README.md`: This file

## Customization

To adapt this tutorial:
1. Add more samples to the JSONL file
2. Modify reward logic in `reward.py`
3. Try different algorithms (`--algo grpo`, `--algo ppo`)
4. Adjust sampling parameters (`--n-samples`, `--temperature`)
