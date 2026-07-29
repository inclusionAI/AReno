# Countdown Arithmetic Agent

An agentic RL demo for the Countdown number game using AReno.

## Game Rules

Given 6 numbers and a target, use basic arithmetic (`+`, `-`, `*`, `/`) to
reach the target. Each number can only be used once. Division must result in
integers. The model solves puzzles by calling tools step by step and
submitting its final answer via the `finish` tool.

## Files

| File | Role |
|------|------|
| `dataset_loader.py` | Input conversion: reads `countdown.jsonl` and formats each row into an agent prompt |
| `reward.py` | Reward signal: scores a trajectory by how close the `finish` answer is to the target |
| `run_agent.py` | Agent environment: bounded multi-turn loop with 5 tools (add, subtract, multiply, divide, finish) |
| `data/countdown.jsonl` | 10 sample problems |
| `UNDERSTANDING.md` | Design notes and the author's understanding of this PR |

## Training

```bash
cd AReno
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path examples/agentic/countdown/data/countdown.jsonl \
  --dataset-loader-fn examples/agentic/countdown/dataset_loader.py \
  --reward-fn-path examples/agentic/countdown/reward.py \
  --agent-fn examples/agentic/countdown/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 512 \
  --max-steps 10 \
  --model-hub modelscope
```

For GPU-constrained environments (e.g. Kaggle T4, 14.5 GB VRAM), add memory
flags such as `--adam-8bit`, `--drop-rollout-state`, and
`--attn-backend native`.

## Sample Data

Example problem:
```json
{"numbers": [25, 50, 75, 100, 3, 6], "target": 952, "id": "1"}
```

Solution: `(100 + 6) * (75 - 50) - 3 * 25 = 952`