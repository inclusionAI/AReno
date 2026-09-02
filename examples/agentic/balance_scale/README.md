# Agentic Balance-Scale Example

This example trains a policy to solve an odd-ball balance-scale puzzle: given a
set of visually identical balls where one is heavier or lighter, the agent uses
a balance scale to identify the odd ball and its weight direction within a
limited weighing budget.

## Files

- `game.py` — Balance-scale game logic: `BalanceGame` class, `generate_game`,
  `format_prompt`, `format_xml_prompt`, and XML parsing helpers.
- `dataset_generator.py` — Generates reproducible JSONL puzzle records.
- `dataset_loader.py` — Loads JSONL into Areno prompt records (tool-call variant).
- `dataset_loader_no_tool.py` — Same but uses XML-tag prompts for models that
  cannot produce OpenAI tool calls.
- `run_agent.py` — Multi-turn agent entrypoint with `weigh` and `answer` tools.
- `run_agent_no_tool.py` — Multi-turn agent using XML tags (`<weigh>` / `<answer>`)
  instead of OpenAI tool calls. Suitable for smaller models.
- `reward.py` — Rewards full-answer (1.0), identity-only (0.5), or wrong (0.0).
- `reward_no_tool.py` — Same reward logic but extracts the answer from XML tags
  in the completion text.

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

## No-Tool XML Variant

For models that do not reliably produce OpenAI tool calls (e.g. smaller than
1.7B parameters), use the XML no-tool variant. The model outputs plain text
containing `<weigh left="0,1" right="2,3"/>` and
`<answer ball="3" direction="heavier"/>` tags instead of structured tool calls.

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/areno-balance-scale-puzzles.jsonl \
  --dataset-loader-fn examples/agentic/balance_scale/dataset_loader_no_tool.py \
  --reward-fn-path examples/agentic/balance_scale/reward_no_tool.py \
  --agent-fn examples/agentic/balance_scale/run_agent_no_tool.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 4 \
  --max-new-tokens 64 \
  --adam-8bit \
  --drop-rollout-state
```
