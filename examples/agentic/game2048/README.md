# Agentic 2048 Example

This example trains a policy to play 2048 via **single-turn** LLM calls. Each
step, the model sees only the system prompt and the current board state — no
conversation history is accumulated. The model outputs brief reasoning followed
by a direction keyword (UP/DOWN/LEFT/RIGHT), which is parsed from the response
text.

This design matches academic RL practice for 2048: the game is Markovian, so
the current board is the complete state. Single-turn calls avoid context
length growth (OOM) and eliminate tool-call parsing failures.

## Files

- `game.py` — 4x4 board engine: slide/merge, spawn, terminal detection, scoring.
- `dataset_generator.py` — generates reproducible initial board JSONL.
- `dataset_loader.py` — loads JSONL into AReno prompt records.
- `run_agent.py` — single-turn agent loop (one LLM call per move, no history).
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

## How It Works

Each episode plays up to `max_moves` steps. At each step:

1. `run_agent.py` sends the system prompt + current board to the model as a
   fresh chat (no prior turns appended).
2. The model outputs brief reasoning, ending with `MOVE: <DIRECTION>`.
3. `game.parse_action()` extracts the direction from the text.
4. If no direction is found, a random legal move is used as fallback.
5. The board is updated and the next step repeats.

All steps from one episode are collected as `AgentTrajectoryTurn` entries and
merged by the AReno framework into a single training row. The reward function
parses all directions from the concatenated response text and replays the
episode on the seeded board.

## Reward

The reward function replays the agent's move sequence on the seeded board and
combines:

| Component | Weight | Description |
|-----------|--------|-------------|
| Merge score | 1.0 | Total merge score across all valid moves |
| Monotonicity bonus | 0.01 | Per-step bonus for monotonic board structure |
| Empty-cell bonus | 0.02 | Per-step bonus × empty cell count |
| Invalid move penalty | -1.0 | Per invalid move |

## Episode Cap

Each episode is capped at `DEFAULT_MAX_MOVES = 50` turns. The cap is
configurable via the `max_moves` field in the dataset record.