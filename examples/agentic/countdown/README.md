# Countdown Arithmetic Agent

An agentic RL demo for the Countdown number game using AReno.

## Game Rules

Given 6 numbers and a target, use basic arithmetic (+, -, *, /) to reach the target.
Each number can only be used once. Division must result in integers.

## Files

- `dataset_loader.py` - Loads and formats countdown.jsonl data
- `reward.py` - Calculates reward based on how close answer is to target
- `run_agent.py` - Defines 5 tools (add, subtract, multiply, divide, finish)
- `data/countdown.jsonl` - Training data with 10 sample problems

## Training

```bash
cd AReno
areno train \
  --ckpt Qwen/Qwen2.5-0.5B-Instruct \
  --dataset-path examples/agentic/countdown/data/countdown.jsonl \
  --dataset-loader-fn examples/agentic/countdown/dataset_loader.py \
  --reward-fn-path examples/agentic/countdown/reward.py \
  --agent-fn examples/agentic/countdown/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 512 \
  --max-steps 10
```

## Sample Data

Example problem:
```json
{"numbers": [25, 50, 75, 100, 3, 6], "target": 952, "id": "1"}
```

Solution: (100 + 6) * (75 - 50) - 3 * 25 = 952
