# Maze Agentic Example

A partially observable maze for agentic RLVR training. The agent sees only a
local window around its position and must navigate to the goal, optionally
picking up keys to open doors along the way.

## Overview

Unlike the tictactoe and duelgrid examples which are full-information, the
maze demo is **partially observable**: the agent only sees a small window
(e.g. 3x3 or 5x5) around its current position. This makes exploration and
multi-step planning essential.

Each episode is a multi-turn tool-call trajectory:

1. The agent receives its local view as a text prompt.
2. It calls the `act` tool with one action (`move`, `pickup`, or `use_key`).
3. The action is executed and the new view is sent back as a tool result.
4. Repeat until the goal is reached or `max_steps` is exhausted.

### Tiles

| Symbol | Meaning |
|--------|---------|
| `#` | Wall (impassable) |
| `.` | Open floor |
| `A` | Agent (you) |
| `K` | Key (pick up) |
| `D` | Locked door (needs a key to open) |
| `G` | Goal (reach to win) |
| `?` | Unknown (outside view range) |

### Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| `move` | `direction`: UP/DOWN/LEFT/RIGHT | Move one tile |
| `pickup` | — | Pick up a key if standing on one |
| `use_key` | `direction`: UP/DOWN/LEFT/RIGHT | Open an adjacent door if holding a key |

### Rewards

- Reach goal: **+1.0**
- Pick up key: **+0.2**
- Open door: **+0.1**
- Move (step cost): **-0.01**
- Invalid action (wall, no key, etc.): **-0.1**

## Quick start

Generate maze states:

```bash
python examples/agentic/maze/dataset_generator.py \
  --count 256 \
  --rows 4 --cols 4 \
  --num-keys 1 \
  --output /tmp/maze_states.jsonl
```

Train with GSPO:

```bash
areno train --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/maze_states.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo --tp-size 1
```

## Dataset generator options

```
--count         Number of maze states to generate (default: 128)
--seed          Master random seed (default: 2026)
--rows          Maze cell rows before wall expansion (default: 4)
--cols          Maze cell cols before wall expansion (default: 4)
--num-keys      Maximum key-door pairs per maze (default: 1)
--max-steps     Maximum steps per episode (default: 30)
--view-radius   Local view radius (default: 1, i.e. 3x3 window)
```

## Testing

CPU tests run without a GPU:

```bash
pytest tests/test_agentic_maze_example_cpu.py -v
```

## File structure

```
examples/agentic/maze/
├── game.py              # Maze generation, state, local view, actions
├── run_agent.py         # Multi-turn tool-call agent entrypoint
├── reward.py            # reward_fn(record) — replay trajectory for reward
├── dataset_loader.py    # Load JSONL into Areno prompt records
├── dataset_generator.py # Seed-solvable maze JSONL generator
└── README.md
```
