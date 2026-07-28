# Agentic Water-Jug Example

This example trains a policy to solve water-jug puzzles by calling
fill, empty, and pour actions via an OpenAI-compatible tool call.

## Files

- `game.py` --- Water-jug game logic, BFS solver, and prompt formatting.
- `dataset_generator.py` --- Generate reproducible water-jug puzzle JSONL.
- `dataset_loader.py` --- Load and format puzzles for training.
- `run_agent.py` --- Agent function: model calls water_jug_action tool.
- `reward.py` --- Reward function: +1 for solving, partial reward for proximity.

## Quick Start

### 1. Generate dataset

    python examples/agentic/water_jug/dataset_generator.py \
        --output /tmp/water_jug_puzzles.jsonl \
        --count 1024 --seed 2026

### 2. Train with Agentic GSPO

    areno train \
        --ckpt Qwen/Qwen3.5-0.8B \
        --dataset-path /tmp/water_jug_puzzles.jsonl \
        --dataset-loader-fn examples/agentic/water_jug/dataset_loader.py \
        --reward-fn-path examples/agentic/water_jug/reward.py \
        --agent-fn examples/agentic/water_jug/run_agent.py \
        --algo gspo \
        --batch-size 2 \
        --n-samples 4 \
        --max-new-tokens 128

### 3. Unit test

    python -c "
    import sys; sys.path.insert(0, 'examples/agentic/water_jug')
    import game
    print(game.shortest_path((3, 5), (0, 0), 4))
    "

## Puzzle Format

Each puzzle is a JSONL line:

    {"id":"generated-000000","capacities":[3,5],"initial_state":[0,0],"target":4,"oracle_steps":6}

- `capacities`: max litres each jug can hold
- `initial_state`: starting water in each jug (always all zeros)
- `target`: desired litres in any single jug
- `oracle_steps`: BFS-optimal action count (for reward shaping)

## Actions

The model calls `water_jug_action(action="...")` with one of:

- `fill(i)` --- fill jug i to its capacity
- `empty(i)` --- empty jug i
- `pour(i,j)` --- pour jug i into jug j until j is full or i is empty

## Reward Design

| Outcome | Reward |
|---------|--------|
| Solved in optimal steps | 1.1 |
| Solved with extra steps | 1.0 - 0.1 * (extra steps), floor 0.1 |
| Not solved but closer | 0.5 * proximity (0 to 0.5) |
| Not solved, no progress | 0.0 |
