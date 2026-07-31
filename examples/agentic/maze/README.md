# Agentic Maze Example

This example trains a policy to navigate a **partially observable maze** — the
agent sees only a bounded local view around its position and must find a key to
unlock doors before reaching the goal.  The environment is deterministic and
self-contained (no external services).

## Key Features

- **POMDP environment**: walls, keys, doors, and a goal.
- **Bounded local view**: the agent only sees cells within a configurable vision
  radius (default 3×3).  The full maze is never exposed through any observation.
- **Multi-turn tool calls**: the agent calls `move(direction)` one step at a time
  in a loop until it reaches the goal or exhausts its step budget.
- **Configurable size**: maze width/height, vision radius, and max steps are all
  configurable.

## Files

- `game.py` — pure maze environment: generation, state transitions, local view,
  scoring, and serialisation.  Zero AReno dependencies.
- `dataset_generator.py` — generates reproducible solvable maze JSONL datasets.
- `dataset_loader.py` — loads JSONL mazes into AReno prompt records.
- `reward.py` — replays move sequences against the maze and scores outcomes.
- `run_agent.py` — multi-turn agent entrypoint using OpenAI-compatible tool calls.

## Generate Mazes

Use 5×5 mazes with `--max-steps 10` for best training results (shortest path
is 6-8 steps; larger mazes cause multi-turn context to exceed limits):

```bash
python examples/agentic/maze/dataset_generator.py \
  --output /tmp/areno-maze.jsonl \
  --count 512 \
  --seed 2026 \
  --width 5 \
  --height 5 \
  --vision-radius 1 \
  --max-steps 10
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--count` | 128 | Number of mazes to generate |
| `--seed` | 2026 | Random seed for reproducibility |
| `--width` | 7 | Maze width (min 5, auto-rounded to odd) |
| `--height` | 7 | Maze height (min 5, auto-rounded to odd) |
| `--vision-radius` | 1 | Agent's vision radius (1 = 3×3 view) |
| `--max-steps` | width×height | Maximum steps per episode (recommended: 10 for 5×5) |

## Run Training

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/areno-maze.jsonl \
  --dataset-loader-fn examples/agentic/maze/dataset_loader.py \
  --reward-fn-path examples/agentic/maze/reward.py \
  --agent-fn examples/agentic/maze/run_agent.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 2 \
  --max-new-tokens 64 \
  --disable-thinking \
  --attn-backend native \
  --max-context-len 16384 \
  --tp-size 2 \
  --world-size 2 \
  --max-steps 500 \
  --save-interval 100 \
  --mini-bs 1 \
  --adam-8bit \
  --activation-checkpointing
```

> **Tips** (verified on 2×A10 24GB):
> * `--disable-thinking`: Qwen3 thinking mode wastes tokens for single-step
>   direction choices; disabling it speeds up training 2.5× with better reward
> * `--max-context-len 16384`: multi-turn episodes accumulate context; 10-step
>   episodes produce ~8K tokens per trajectory
> * `--attn-backend native`: avoids Triton/FLA version conflicts
> * `--adam-8bit --activation-checkpointing`: required for 24GB GPUs

## How It Works

1. **Dataset generation**: each maze is carved with randomized DFS, then one key
   and one door are placed.  The door sits on the shortest path so the key is
   required; the key is placed in a region reachable before the door.

2. **Local view**: the agent sees a `(2r+1)×(2r+1)` grid centered on itself.
   Cells outside are shown as `?`.  Glyphs: `@` = agent, `#` = wall, `.` = empty,
   `k` = key, `D` = locked door, `G` = goal.

3. **Multi-turn episode**: `run_agent` deserializes the initial maze state, then
   loops: call the model → parse `move(direction)` → execute on the local maze
   → append tool result with new observation → repeat until goal or max steps.

4. **Reward**: `reward_fn` extracts the move sequence from `tool_calls`, replays
   it against the initial state, and scores via BFS closest-approach shaping:
   goal reached → `1.0 − penalty × excess_steps`; not reached → `−0.5 + 0.3 ×
   (1 − min_dist / maze_size)` (distance-based gradient); invalid moves →
   `−0.1` each. An optional PBRS mode is available via `source_record["reward_mode"]
   = "pbrs"`.

## Testing

```bash
pytest tests/test_agentic_maze_example_cpu.py -v
```

All tests are CPU-only and require no GPU.