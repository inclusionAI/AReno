# Countdown Arithmetic Agentic RL Demo

This demo implements a single-step Countdown numbers game as an agentic RL
example, following the same structure as the Tic-Tac-Toe demo.

## How the game works

The model receives a set of numbers and a target value. It picks exactly two
numbers and one arithmetic operation (`+`, `-`, `*`, `/`) by calling the
`calculate` tool. The reward is based on how close the result is to the target.

## Files

| File | Purpose |
|---|---|
| `game.py` | Arithmetic rules, proximity-based scoring, prompt formatting, random baseline, trace replay, evaluation metrics |
| `dataset_generator.py` | Generates random puzzles with guaranteed solvable targets |
| `dataset_loader.py` | Loads JSONL puzzles and converts to AReno prompt records |
| `run_agent.py` | Single-turn agent that calls the `calculate(a, b, op)` tool |
| `reward.py` | Extracts tool-call arguments and scores the move |
| `fixtures/easy.jsonl` | Deterministic easy puzzles (2 numbers, simple operations) |
| `fixtures/medium.jsonl` | Deterministic medium puzzles (4-5 numbers) |
| `fixtures/hard.jsonl` | Deterministic hard puzzles (6 numbers, large targets) |

## Quick start

### Generate a dataset

```bash
python examples/agentic/countdown/dataset_generator.py --output countdown.json --count 1024
```

### Run GSPO training

```bash
areno train \
  --ckpt Qwen/Qwen3.5-0.8B \
  --world-size 2 \
  --algo gspo \
  --tp-size 2 \
  --dataset-path countdown.json \
  --dataset-loader-fn examples/agentic/countdown/dataset_loader.py \
  --reward-fn-path examples/agentic/countdown/reward.py \
  --agent-fn examples/agentic/countdown/run_agent.py \
  --batch-size 2 \
  --n-samples 4 \
  --max-running-prompts 8 \
  --max-new-tokens 256 \
  --mini-bs 2
```

### Use deterministic fixtures

```bash
# Easy
areno train --dataset-path examples/agentic/countdown/fixtures/easy.jsonl ...

# Medium
areno train --dataset-path examples/agentic/countdown/fixtures/medium.jsonl ...

# Hard
areno train --dataset-path examples/agentic/countdown/fixtures/hard.jsonl ...
```

## Observable output

During training, the logs report:

- `tool_calls`: number of successful `calculate` tool calls per batch
- `reward_mean`: average reward across all samples in the step
- `rollout_logprob_mean`: average log-probability of generated tokens

A reward of `1.0` means the model exactly hit the target. A reward of `-1.0`
means an invalid operation (e.g., division by zero, number not in the list).

## Scoring

| Outcome | Score |
|---|---|
| Result exactly equals target | 1.0 |
| Result close to target | 0.0 – 0.8 (scaled by proximity) |
| Invalid operation | -1.0 |

## Evaluation metrics

The `game.py` module provides:

- `random_baseline_score()`: average reward of a random policy
- `evaluate_moves()`: aggregate metrics (exact-solve rate, invalid-action rate, mean reward)
- `oracle_solve()`: best achievable score for a puzzle
- `format_trace()`: human-readable trace of a single move

## Tests

```bash
pytest tests/test_countdown_cpu.py -k cpu
```

Covers: arithmetic, scoring, input validation, random baseline determinism,
trace replay, evaluation metrics, oracle solver, and all three fixtures.