# Warehouse-Picking Agentic RL Demo

A grid-based warehouse environment where an Agent navigates aisles, queries
shelf inventory, picks items into a cart, and submits completed orders at a
deposit point.

## Environment

```
S . . # .    S = start (agent)
. # . . .    # = shelf
. . . . .    . = aisle
# . . . D    D = deposit point
```

The Agent receives a warehouse layout and an order (item + quantity pairs).
It must find the items on shelves, pick them, and submit at the deposit point.

## Actions

| Action | Parameters | Description |
|---|---|---|
| `query_inventory` | `shelf_id: str` | Inspect a shelf's current stock |
| `move` | `direction: str` | Move up/down/left/right (one cell) |
| `pick` | `item: str, quantity: int` | Pick item from adjacent shelf into cart |
| `submit` | (none) | Submit cart at deposit point |

## Quick Example

```python
from examples.agentic.warehouse.game import (
    generate_small, query_inventory, move, pick, submit,
    solve_baseline, compute_metrics, make_prompt,
)

# Generate a small warehouse
state = generate_small(seed=42)
print(make_prompt(state))

# Query a shelf
result = query_inventory(state, "shelf_1")
print(result)  # {"ok": True, "shelf_id": "shelf_1", "stock": {"item_A": 3}}

# Move the agent
move(state, "right")
move(state, "down")

# Pick an item (must be adjacent to a shelf)
pick(state, "item_A", 1)

# Navigate to deposit and submit
# ... move to bottom-right corner ...
result = submit(state)

# Compute metrics
baseline = solve_baseline(state)
metrics = compute_metrics(state, baseline)
print(metrics)
# {"complete_orders": 1, "picking_mistakes": 0, "invalid_actions": 0, ...}
```

## Difficulty Levels

| Level | Grid | Shelves | Item Types | Order Size |
|---|---|---|---|---|
| Small | 4x4 | 3 | 2 | 1 |
| Medium | 6x6 | 6 | 4 | 2 |
| Hard | 8x8 | 10 | 6 | 3 |

## Metrics

| Metric | Description |
|---|---|
| `complete_orders` | 1 if order completed, 0 otherwise |
| `picking_mistakes` | Wrong item or out-of-stock pick attempts |
| `invalid_actions` | Invalid moves (out of bounds, blocked, wrong direction) |
| `agent_distance` | Total steps the agent took |
| `baseline_distance` | BFS shortest path visiting all required shelves |
| `distance_ratio` | agent_distance / baseline_distance (1.0 = optimal) |

## Validation

Every action is validated:
- **Move**: checks bounds and shelf collision
- **Pick**: checks adjacent shelf, item existence, stock availability
- **Submit**: checks deposit position, cart completeness, wrong items
- **Query**: checks shelf ID exists

## Determinism

Given the same random seed, the warehouse layout, shelf stock, and order
are identical, enabling reproducible training and testing.