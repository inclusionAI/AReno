# Agentic Balance-Scale Example

This example trains a policy to solve an odd-ball balance-scale puzzle: given a
set of visually identical balls where one is heavier or lighter, the agent uses
a balance scale to identify the odd ball and its weight direction within a
limited weighing budget.

## Files

- `game.py` — Balance-scale game logic: `BalanceGame` class, `generate_game`,
  and `format_prompt`.
- `dataset_generator.py` — Generates reproducible JSONL puzzle records.
- `dataset_loader.py` — Loads JSONL into Areno prompt records.
- `run_agent.py` — Multi-turn agent entrypoint with `weigh` and `answer` tools.
- `reward.py` — Rewards full-answer (1.0), identity-only (0.5), or wrong (0.0).

## Generate Puzzle Data

```bash
python examples/agentic/balance_scale/dataset_generator.py \
  --output /tmp/areno-balance-scale-puzzles.jsonl \
  --count 2048 \
  --seed 2026 \
  --num-balls 9 \
  --max-weighings 3
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-balance-scale-puzzles.jsonl \
  --dataset-loader-fn examples/agentic/balance_scale/dataset_loader.py \
  --reward-fn-path examples/agentic/balance_scale/reward.py \
  --agent-fn examples/agentic/balance_scale/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 256
```

## Reward Structure

| Outcome | Reward |
|---------|--------|
| Correct ball identity + correct direction | 1.0 |
| Correct ball identity, wrong direction | 0.5 |
| Wrong ball or no answer | 0.0 |

## Tools

### weigh

Compares two disjoint equal-size groups of balls on a balance scale.

Parameters:
- `left_group`: list of ball indices (integers)
- `right_group`: list of ball indices (integers)

Returns: `left_heavy`, `right_heavy`, or `balanced`.

### answer

Submits the final answer.

Parameters:
- `ball_index`: the index of the odd ball (integer)
- `direction`: `"heavier"` or `"lighter"`

## Limitations

- The weighing budget is enforced by the game; exceeding it raises an error
  that is surfaced to the model as a tool-response message.
- Each puzzle has exactly one odd ball; there are no "all balls equal" puzzles.
- The multi-turn loop is capped at 20 model calls per puzzle as a safety net.
