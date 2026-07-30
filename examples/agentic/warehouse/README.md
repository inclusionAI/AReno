# Agentic Warehouse-Picking Example

This example trains a policy on a deterministic, multi-turn warehouse task.
Each sample contains a connected shelf grid, reproducible stock, and a
satisfiable order. Every prompt/sample rollout owns an independent mutable
environment. The prompt tells the agent which shelf holds the ordered item;
the agent must navigate there and submit the order.

On each bounded action turn, the model must call exactly one tool:

1. `move_to` moves one step to a directly adjacent shelf.
2. `submit_order` validates that the agent is at the target shelf and
   completes the order.

The turn limit equals the shortest-route distance plus one submit turn. This
keeps every generated task achievable while bounding rollout length. Missing
calls and malformed or multiple calls remain visible as failures; the agent
does not fabricate a successful call.

## Generate Tasks

```bash
python examples/agentic/warehouse/dataset_generator.py \
  --output /tmp/areno-warehouse.jsonl \
  --count 2048 \
  --seed 2026
```

The generator prints a JSON summary and writes local JSONL records. Generation
cycles through small (2x2), medium (3x3), and hard (4x3) layouts, so any count
of at least three contains every difficulty. Reusing the same seed produces
the same records. Every generated order quantity fits on at least one shelf.

Inputs are validated before agent requests begin. For example, this boundary
input fails with `count must be a positive integer`:

```bash
python examples/agentic/warehouse/dataset_generator.py \
  --output /tmp/invalid.jsonl \
  --count 0
```

Dataset validation reports the zero-based row and affected field, such as
`warehouse dataset row 2: start_shelf must identify a shelf in the 3x3 layout`.

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-warehouse.jsonl \
  --dataset-loader-fn examples/agentic/warehouse/dataset_loader.py \
  --reward-fn-path examples/agentic/warehouse/reward.py \
  --agent-fn examples/agentic/warehouse/run_agent.py \
  --algo gspo \
  --batch-size 8 \
  --n-samples 4 \
  --max-context-len 8192 \
  --max-new-tokens 128
```

The reward replays exact assistant tool calls against a fresh deterministic
environment. Exact order submission is the primary outcome. Partial navigation
progress earns a graded reward based on remaining distance to the target.
Invalid actions (malformed tools, unreachable moves, invalid submissions) are
penalized.

## Observable Output

Every tool result contains a human-readable message and structured `metrics`:

- `complete_orders`: `1` after a successful submission, otherwise `0`
- `invalid_actions`: malformed tools, unreachable moves, and invalid submissions
- `distance` and `baseline_distance`
- `remaining_distance`
- `progress`: fraction of baseline distance covered

The agent logs the same completion, invalid-action, and distance fields per
action without logging the full training sample. JSONL records keep the
deterministic seed needed to reconstruct layouts and replay trajectories.

## Limitations

- The environment is local and uses no external database or sandbox.
- Only existing AReno dependencies are required.
- Shelf layouts are connected rectangular grids, not arbitrary warehouse maps.
- Inventory locations are deterministic and static during each episode.
- Generated tasks are satisfiable. Invalid action behavior is exercised
  through invalid action paths and focused CPU tests.