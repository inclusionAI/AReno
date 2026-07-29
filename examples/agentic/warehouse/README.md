# Agentic Warehouse-Picking Example

This example trains a policy on a deterministic, multi-turn warehouse task.
Each sample contains a connected shelf grid, reproducible stock, and a
satisfiable order. Every prompt/sample rollout owns an independent mutable
environment. The robot locates requested SKUs through the inventory service
before navigating to and inspecting their shelves.

On each bounded action turn, the model must call exactly one tool:

1. `query_inventory` returns every current shelf location for one SKU.
2. `move_to` moves one step to an adjacent shelf.
3. `check_shelf` verifies stock on the current shelf. Inspecting a shelf with
   no still-requested stock is recorded as an empty check.
4. `pick_item` picks verified stock from the current shelf.
5. `submit_order` validates the cart against the exact order.

The turn limit is the shortest-route reference action count plus two recovery
turns. This keeps every generated task achievable while bounding rollout
length. Picking before inspecting the current shelf is rejected.

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
the same records. Every generated order quantity fits on at least one shelf;
the environment and route baseline continue to support split stock in custom
fixtures.

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
environment. Exact order submission is the primary outcome. Completed paths
receive an additional efficiency component based on the exact minimum movement
distance needed to reach sufficient stock. Empty or repeated inspections,
picking mistakes, and invalid actions are penalized. Incomplete cart progress
remains negative, so merely reaching a shelf cannot earn a completion reward.

## Observable Output

Every tool result contains a human-readable message and structured `metrics`:

- `complete_orders`: `1` after an exact successful submission, otherwise `0`
- `picking_mistakes`: wrong-item, excess-quantity, and stock errors
- `invalid_actions`: malformed tools, unreachable moves, and invalid submissions
- `empty_shelf_checks`: inspections with no still-requested stock
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
- Inventory locations are deterministic and static during each episode.
- Generated tasks are satisfiable. Out-of-stock and wrong-item behavior is
  exercised through invalid action paths and focused CPU tests.
