# Agentic Warehouse-Picking Example

This example trains a policy on a multi-turn tool-calling task. Each sample
generates a warehouse with a grid of shelves, stock, and an order. The agent
must navigate shelves, query inventory, pick the required items, and submit the
completed order. The agent runs four model turns:

1. `query_inventory`
2. `move`
3. `pick`
4. `submit_order`

Every step is validated by the environment: shelf existence, reachability
(adjacency), stock quantity, and cart completeness. The reward function scores
order completion, picking mistakes, invalid actions, and distance efficiency
relative to a greedy BFS baseline.

## Generate Tasks

```bash
python examples/agentic/warehouse/dataset_generator.py \
  --output /tmp/areno-warehouse.jsonl \
  --count 2048 \
  --seed 2026
```

Three difficulty levels are generated: small (2x2 grid), medium (3x3), and
hard (4x3 with possible stockouts).

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

## Observable Output

- **Logs**: per-turn tool calls, environment validation results, distances.
- **Metrics**: reward (completion + penalties + distance efficiency), picking
  errors, invalid actions, baseline vs actual distance.
- **Artifacts**: JSONL task records with deterministic seeds for replay.

## Limitations

- Pure local environment, no external database or sandbox required.
- Only standard library + existing AReno dependencies used.
- The agent sees the order in the prompt but must explore shelf inventory via
  `query_inventory` to find where items are located.