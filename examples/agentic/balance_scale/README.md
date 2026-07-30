# Odd-Ball Balance-Scale Agentic Example

This example trains a policy to solve the classic odd-ball balance-scale
puzzle: among *N* visually identical balls, exactly one is heavier or lighter
than the rest. The agent uses a balance-scale tool (`weigh`) to compare two
equal-size disjoint groups and a final-answer action (`submit_answer`) to
identify the odd ball and its weight direction.

The environment is deterministic and self-contained — no network services,
no sandbox, no external dependencies beyond the Python standard library.

## Files

- `game.py` — core engine: ball-set creation, weighing simulation, answer
  verification, and prompt formatting.
- `dataset_generator.py` — generates reproducible JSONL puzzles with seeded
  random odd-ball positions and weight directions. Supports random ball
  counts and train/test split.
- `dataset_loader.py` — loads JSONL puzzles and converts them to Areno prompt
  records.
- `reward.py` — reward function: information-gain-aware continuous scoring
  with repetition/invalid penalties, auto-scales to any number of balls.
- `run_agent.py` — multi-turn agent entrypoint: loops weigh/submit_answer
  tool calls with budget enforcement. Includes text-based fallback parser
  for AReno rollout proxy compatibility.
- `verify_ui.py` — Gradio verification UI: configure puzzles and observe the
  model's reasoning trace and final verdict.

## Generate Puzzles

### Fixed ball count

```bash
python examples/agentic/balance_scale/dataset_generator.py \
  --output /tmp/areno-balance-scale-puzzles.jsonl \
  --count 2048 \
  --seed 2026 \
  --num-balls 12
```

### Random ball count (recommended for generalisation)

```bash
python examples/agentic/balance_scale/dataset_generator.py \
  --output /tmp/areno-balance-scale-puzzles.jsonl \
  --count 2048 \
  --seed 2026 \
  --num-balls-range 3 12
```

Each puzzle will have a random number of balls between 3 and 12, preventing
the model from memorising a single ball count.

### With train/test split

```bash
python examples/agentic/balance_scale/dataset_generator.py \
  --output /tmp/areno-balance-scale-puzzles.jsonl \
  --count 2048 \
  --seed 2026 \
  --num-balls-range 3 12 \
  --split 0.33
```

This generates two files:
- `puzzles.jsonl` — training set (67% of records)
- `puzzles_test.jsonl` — test set (33% of records)

### Generator options

| Option | Default | Description |
| --- | --- | --- |
| `--output` | stdout | Output JSONL path |
| `--count` | 128 | Number of puzzles to generate |
| `--seed` | 2026 | Random seed for reproducibility |
| `--num-balls` | 12 | Fixed number of balls per puzzle |
| `--num-balls-range MIN MAX` | none | Random ball count per puzzle in [MIN, MAX] |
| `--max-weighings` | 0 (auto) | Max weighings (0 = 2×ceil(log3(num_balls*2))) |
| `--split RATIO` | 0 (no split) | Test set fraction (e.g. 0.33) |

When `--max-weighings` is omitted (or 0), it auto-computes as 2× the
information-theoretic minimum: `ceil(log3(num_balls * 2))`.

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/areno-balance-scale-puzzles.jsonl \
  --dataset-loader-fn examples/agentic/balance_scale/dataset_loader.py \
  --reward-fn-path examples/agentic/balance_scale/reward.py \
  --agent-fn examples/agentic/balance_scale/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 64 \
  --world-size 1 \
  --tp-size 1 \
  --disable-thinking \
  --temperature 1.5
```

### Training parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--ckpt` | required | Model checkpoint (e.g. `Qwen/Qwen3-0.6B`) |
| `--dataset-path` | required | Path to JSONL puzzles |
| `--dataset-loader-fn` | required | Path to `dataset_loader.py` |
| `--reward-fn-path` | required | Path to `reward.py` |
| `--agent-fn` | required | Path to `run_agent.py` |
| `--algo` | required | Algorithm: `gspo`, `grpo`, `ppo`, `sft`, `dpo` |
| `--batch-size` | 2 | Prompts per rollout batch |
| `--n-samples` | 4 | Samples per prompt (for advantage computation) |
| `--max-new-tokens` | 64 | Max generated tokens per turn |
| `--world-size` | 1 | Number of GPU workers |
| `--tp-size` | 1 | Tensor parallel size |
| `--mini-bs` | 2 | Mini-batch size for training |
| `--score-micro-bs` | 2 | Micro-batch for reward scoring |
| `--max-steps` | none | Max training steps (none = full dataset) |
| `--disable-thinking` | off | **Required for Qwen3** — disables think mode |
| `--temperature` | 1.0 | Rollout sampling temperature (1.5 recommended) |
| `--save-path` | none | Checkpoint output directory |
| `--save-interval` | 100 | Save checkpoint every N steps |

### Key training tips

- `--disable-thinking` is **required** for Qwen3 models — without it, the
  model enters think mode and consumes all tokens on `...` tags.
- `--temperature 1.5` encourages exploration across samples, producing
  reward variance for GSPO advantage computation. Without it, samples tend
  to produce identical outputs (reward variance = 0, no gradient signal).
- `--n-samples 4+` is recommended to ensure reward diversity within a group.
- `--save-interval` must be <= `--max-steps` for checkpoints to be saved.
- On T4 (15GB), use `--batch-size 1 --n-samples 2 --max-new-tokens 32` to
  avoid OOM. Dual T4 or A100 allows larger configurations.

## Multi-Turn Agent Loop

The agent (`run_agent.py`) runs a multi-turn loop for each puzzle:

1. **Turn 1**: Force `weigh` tool call to bootstrap tool usage.
2. **Turns 2..N**: Allow `weigh` or `submit_answer` (tool_choice=auto).
3. **Budget exhausted**: Force `submit_answer` with a hint message.
4. **Termination**: Loop ends when `submit_answer` is called, model returns
   no tool call, or max turns (`max_weighings * 2 + 1`) is reached.

Each weighing result (`left_heavy` / `right_heavy` / `balanced`) is appended
as a tool-result message so the model can reason about the next step.

### Tool interaction protocol

| Tool | Arguments | Returns | Notes |
| --- | --- | --- | --- |
| `weigh` | `{"left": [0,1], "right": [2,3]}` | `{"result": "left_heavy", "weighings_used": N}` | Groups must be equal-size and disjoint |
| `submit_answer` | `{"ball_index": 5, "direction": "heavier"}` | `{"submitted": true, "ball_index": 5, "direction": "heavier"}` | Direction must be `"heavier"` or `"lighter"` |

## Input Contract

Each JSONL record:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Unique record identifier |
| `num_balls` | int | Number of balls (varies per record when using `--num-balls-range`) |
| `odd_ball_index` | int | Index of the odd ball (0-based) |
| `direction` | string | `"heavier"` or `"lighter"` |
| `max_weighings` | int | Soft upper bound on weighings (auto = 2× ceil(log3(num_balls*2))) |

## Output & Metrics

The reward function populates `record.metadata` with:

| Field | Type | Description |
| --- | --- | --- |
| `weighings_used` | int | Total weigh tool calls (valid + repeated + invalid) |
| `valid_weighings` | int | Unique valid weighings |
| `repeated_weighings` | int | Duplicate weighings (same groups) |
| `invalid_weighings` | int | Invalid weighings (bad size/overlap/range) |
| `min_weighings` | int | Information-theoretic minimum: ceil(log3(num_balls*2)) |
| `base_reward` | int | Base reward = min_weighings |
| `full_answer_accuracy` | float | 1.0 if ball + direction both correct |
| `identity_only_accuracy` | float | 1.0 if ball index correct (regardless of direction) |
| `reward_components` | dict | Breakdown: k, t_cost, repeat_cost, invalid_cost |

## Reward Formula

```
R_end = K - T·alpha - P_repeat - P_invalid
```

| Component | Value | Description |
| --- | --- | --- |
| K (answer reward) | `base` / `base/2` / `0` / `-1` | Full correct / identity only / wrong / no submit |
| base | `ceil(log3(num_balls * 2))` | Information-theoretic minimum, auto-scales |
| T (weighing cost) | `(valid + repeated) * alpha` | Each weighing costs `alpha` (default 0.15) |
| P_repeat | `repeated * 0.3` | Penalty for identical repeated weighings |
| P_invalid | `invalid * 0.2` | Penalty for malformed weighings |

### Example rewards (12 balls, base=3)

| Scenario | Weighings | Reward |
| --- | --- | --- |
| 0 weighings, full correct | 0 | 3.0 (lucky guess) |
| 3 weighings, full correct | 3 | 3.0 - 0.45 = 2.55 |
| 3 weighings, identity only | 3 | 1.5 - 0.45 = 1.05 |
| 3 weighings, wrong | 3 | 0.0 - 0.45 = -0.45 |
| No submit | 3 | -1.0 |

## Fallback Tool Call Parser

AReno's rollout proxy does not parse structured `tool_calls` from model text
output — `message.tool_calls` is always `None` even when the model writes
valid tool call JSON in `content`. The `run_agent.py` includes a fallback
parser (`_parse_tool_call_from_text`) that extracts tool calls from the
model's text using regex patterns:

1. `{"name": "weigh", "arguments": {"left": [0,1], "right": [2,3]}}`
2. `{"left": [0,1], "right": [2,3]}`
3. `{"ball_index": 5, "direction": "heavier"}`
4. Array extraction when `tool_choice` forces `weigh`
5. Natural language "ball 5 is heavier" when `tool_choice` forces `submit_answer`

Parsed tool calls are injected back into the response object and
`AgentTrajectoryTurn.parsed_tool_calls` so the AReno framework can track
them in metrics and reward records.

## Verification UI

After training, launch the Gradio UI to interactively test the model:

```bash
python examples/agentic/balance_scale/verify_ui.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key EMPTY \
  --model policy \
  --agent-mode model
```

Or without an LLM (random agent for demo):

```bash
python examples/agentic/balance_scale/verify_ui.py --agent-mode random
```

The UI lets you:
- Set the number of balls (2–200)
- Set the odd ball index and direction (or randomize)
- Watch the model's multi-turn weighing trace
- See the final verdict: correct/wrong, weighings used, efficiency vs optimal

In Colab/Kaggle, the UI appears inline with `share=True`.

### Serve configuration for T4

On T4 (15GB), serve requires `--max-running-prompts 1` to reduce KV cache
pre-allocation:

```bash
areno serve --model-path <checkpoint> --tp-size 1 --world-size 1 \
  --port 8000 --eager-decode --default-max-tokens 64 \
  --max-running-prompts 1 --disable-thinking
```

## Limitations

- The weighing budget is a soft upper bound (auto = 2× theoretical minimum);
  the agent loop enforces it by forcing `submit_answer` when exhausted.
  The loop allows up to `max_weighings * 2 + 1` turns to accommodate invalid
  weighings that don't consume budget.
- The reward auto-scales via `ceil(log3(num_balls * 2))`, so larger ball
  counts produce higher base rewards and proportionally higher weighing costs.
- GPU training requires NVIDIA GPU with sufficient VRAM; T4 (15GB) can run
  rollout + 1-2 training steps with `--batch-size 1 --max-new-tokens 32`.
  Dual T4 or A100 allows multi-step training.
- AReno's rollout proxy does not return structured `tool_calls`; the fallback
  text parser is required for tool-call extraction (see above).
- Qwen3 models require `--disable-thinking` to prevent think mode from
  consuming all generation tokens.
