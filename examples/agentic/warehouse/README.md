# Agentic Warehouse-Picking Example

This example trains a policy on a deterministic, multi-turn warehouse task.
Each sample contains a connected shelf grid, reproducible stock, and a
satisfiable order. Every prompt/sample rollout owns an independent mutable
environment.

On each of at most six action turns, the model must call exactly one tool:

1. `check_shelf` inspects stock on the current shelf.
2. `move_to` moves one step to an adjacent shelf.
3. `pick_item` picks stock from the current shelf.
4. `submit_order` validates the cart against the exact order.

After a completed, rejected, or turn-limited episode, one final model turn
summarizes the real tool results without calling another tool. Missing calls
and malformed or multiple calls remain visible as failures; the agent does not
fabricate a successful call.

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
the same records. Generated order quantities never exceed total stock.

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
  --max-new-tokens 128
```

The reward replays exact assistant tool calls against a fresh deterministic
environment. It rewards exact completion and route efficiency while
penalizing picking mistakes, invalid actions, and repeated checks. Incomplete
cart progress remains non-positive, so repeated observation or movement cannot
earn a positive reward by itself.

## Observable Output

Every tool result contains a human-readable message and structured `metrics`:

- `complete_orders`: `1` after an exact successful submission, otherwise `0`
- `picking_mistakes`: wrong-item, excess-quantity, and stock errors
- `invalid_actions`: malformed tools, unreachable moves, and invalid submissions
- `distance` and `baseline_distance`
- `distance_ratio` and `distance_efficiency`
- `cart_progress`

The agent logs the same completion, mistake, invalid-action, and distance
fields per action without logging the full training sample. JSONL records keep
the deterministic seed needed to reconstruct layouts and replay trajectories.

## Limitations

- The environment is local and uses no external database or sandbox.
- Only existing AReno dependencies are required.
- Shelf layouts are connected rectangular grids, not arbitrary warehouse maps.
- Generated tasks are satisfiable. Out-of-stock and wrong-item behavior is
  exercised through invalid action paths and focused CPU tests.
