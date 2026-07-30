# Battleship Agentic RL Demo

A multi-turn agentic RL example where a model learns to play Battleship using tool calls.

## Game Rules

- **Board**: 8×8 grid (rows A-H, columns 1-8)
- **Fleet**: 4 ships of lengths [4, 3, 2, 2] = 11 total cells
- **Goal**: Sink all ships in as few shots as possible
- **Max turns**: 40 shots per game (< 64 board cells, so exhaustive search cannot win — the agent must play strategically)
- **Feedback**:
  - `miss`: No ship at that coordinate
  - `hit`: Ship hit but not sunk
  - `sunk`: Last cell of a ship was hit

The agent sees only its own view: hits (X), misses (o), and unknown cells (.). The hidden ship cells are never revealed.

## Files

| File | Description |
|------|-------------|
| `game.py` | Core game logic (no areno imports) |
| `dataset_generator.py` | Generate reproducible fleet JSONL |
| `dataset_loader.py` | Load fleets for training |
| `reward.py` | Reward function for RL |
| `run_agent.py` | Multi-turn agent loop |
| `evaluate.py` | Baseline comparison harness |
| `play_llm.py` | Batch-evaluate any OpenAI-compatible LLM (or served trained model) |
| `web_ui.py` | Cartoon browser game backed by an OpenAI-compatible tool-call model |

## Quick Start

### 1. Generate training data

```bash
python examples/agentic/battleship/dataset_generator.py \
  --output /tmp/battleship.jsonl \
  --count 128 \
  --seed 2026
```

### 2. Run training

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/battleship.jsonl \
  --dataset-loader-fn examples/agentic/battleship/dataset_loader.py \
  --agent-fn examples/agentic/battleship/run_agent.py \
  --reward-fn-path examples/agentic/battleship/reward.py \
  --algo gspo \
  --n_samples 4 \
  --metrics-log-dir ./runs/battleship
```

### 3. Evaluate baselines

```bash
# Random baseline
python examples/agentic/battleship/evaluate.py \
  --fleets /tmp/battleship.jsonl \
  --player random

# Fake deterministic player (for testing)
python examples/agentic/battleship/evaluate.py \
  --fleets /tmp/battleship.jsonl \
  --player fake
```

### 4. Play in the Web UI

Start a policy server, then point the UI at its OpenAI-compatible endpoint:

```bash
areno serve --model-path /path/to/model --port 8000 --world-size 1
python examples/agentic/battleship/web_ui.py \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key token \
  --model policy
```

Open `http://127.0.0.1:8768`. The UI supports clicking cells to fire, an
"Agent Fires Once" button, "Auto-play" to run the agent to completion, a
seed input to replay a fixed fleet, and switching the agent between **LLM**
mode (uses the `fire` tool against your server) and **Heuristic** mode
(hunt/target strategy, no server needed).

To run without an LLM server:

```bash
python examples/agentic/battleship/web_ui.py --agent-mode heuristic
```

### 5. Batch-evaluate an LLM

`play_llm.py` plays `--games` seeded fleets against any OpenAI-compatible
endpoint — a trained checkpoint served by `areno serve`, or an external LLM —
and reports win rate, completion, shots to win, and invalid-shot rate.

```bash
# Trained checkpoint served locally
areno serve --model-path ./runs/battleship/.../step_100 --port 8000 --world-size 1
python examples/agentic/battleship/play_llm.py \
  --base-url http://127.0.0.1:8000/v1 \
  --games 50 --output /tmp/battleship_llm.json

# External LLM (OpenAI-compatible)
python examples/agentic/battleship/play_llm.py \
  --base-url https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --model gpt-4o-mini \
  --games 50
```

## Observable Output

Training produces artifacts in `--metrics-log-dir`:

- **TensorBoard** (`rollout/accuracy`): Win rate (fraction of games won)
- **TensorBoard** (`rollout/rewards_mean`): Mean reward per batch
- **JSONL transcripts** (`rollout_samples.<pid>.jsonl`): Full game transcripts for inspection

```bash
tensorboard --logdir ./runs/battleship
```

## Reward Shape

The reward function provides dense shaping:

- **+1.0**: Win (all ships sunk within turn cap)
- **+0.05**: Per hit cell
- **+0.15**: Per sunk ship
- **−0.02**: Per invalid/repeated shot
- **−0.05 × shots_used**: Efficiency penalty (strong enough that exhaustive/slow wins score near 0 while efficient wins score high)

This creates a learnable gradient: a random/exhaustive player scores ~0 reward (it cannot sink the fleet within 40 shots), while an efficient policy that wins quickly scores up to ~1.6. Eval confirms the random baseline's win rate drops from 100% (at 64 turns) to 0% (at 40 turns).

## Limitations

- **GPU required for real training**: The agent loop runs on CPU, but the model requires a GPU.
- **Single-player**: The agent plays solo against a fixed, randomly-placed fleet (no opponent).
- **Compact board**: Uses 8×8 / 11-cell fleet for shorter trajectories. The standard 10×10 / 17-cell board requires more context tokens.
- **No two-player Battleship**: This is a solo search problem, not the full two-player game.

## Testing

Run the CPU test suite:

```bash
python -m pytest tests/test_agentic_battleship_example_cpu.py -k cpu -v
```

Run the evaluation harness:

```bash
python examples/agentic/battleship/evaluate.py --fleets /tmp/battleship.jsonl --player random
```