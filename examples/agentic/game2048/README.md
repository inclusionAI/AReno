# Agentic 2048 Example

This example trains a policy to play 2048 via multi-turn tool calls. The model
calls a `move` tool with a direction (UP, DOWN, LEFT, RIGHT) each turn. After
each move the environment spawns a random tile (2 or 4) using a seeded RNG,
reports the new board and merge score, and continues until the board is full
or the move cap is reached.

The game engine is pure Python with no external dependencies. Tile placement
is deterministic given the seed, so every episode is reproducible.

## Files

- `game.py` — 4x4 board engine: slide/merge, spawn, terminal detection, scoring.
- `dataset_generator.py` — generates reproducible initial board JSONL.
- `dataset_loader.py` — loads JSONL into AReno prompt records.
- `run_agent.py` — multi-turn agent loop (move tool call per turn).
- `reward.py` — replays the agent's moves and computes episode reward.

## Generate Boards

```bash
python examples/agentic/game2048/dataset_generator.py \
  --output /tmp/game2048-boards.jsonl \
  --count 256 \
  --seed 2026
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/game2048-boards.jsonl \
  --dataset-loader-fn examples/agentic/game2048/dataset_loader.py \
  --reward-fn-path examples/agentic/game2048/reward.py \
  --agent-fn examples/agentic/game2048/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 64
```

## Reward

The reward function replays the agent's move sequence on the seeded board and
combines three signals into a scalar in [-1, 1]:

| Component | Weight | Description |
|-----------|--------|-------------|
| Merge score | 50% | `min(total_score / 2000, 1)` |
| Max tile | 30% | `min(log2(max_tile) / 13, 1)` (8192 = 1.0) |
| Valid-move base | 20% | Fixed bonus, reduced by invalid-move penalty |

Invalid moves (moves that do not change the board) do not consume an RNG draw
and incur a penalty of `0.3 * invalid_rate`.

## Episode Cap

Each episode is capped at `DEFAULT_MAX_MOVES = 200` turns. The cap is
configurable via the `max_moves` field in the dataset record.