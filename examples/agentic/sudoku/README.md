# Agentic Sudoku Example

Sudoku is a deterministic multi-turn constraint-satisfaction puzzle. The policy
receives a 9x9 grid with some cells filled and must place digits 1-9 in every
empty cell so that each row, column, and 3x3 box contains 1-9 with no repeats.
Three tools are available: `inspect_candidates`, `place_digit`, and `undo`.
The environment validates constraints after every tool call but never reveals
the solution.

## Files

- `game.py` — puzzle generator, board validation, three tool implementations,
  episode state, and scoring.
- `dataset_generator.py` — generates reproducible uniquely-solvable puzzles as
  JSONL.
- `dataset_loader.py` — loads JSONL into AReno prompt records.
- `run_agent.py` — multi-turn agent entrypoint for agentic rollout.
- `reward.py` — replays the tool-call trajectory and scores the episode.
- `tui.py` — terminal UI for human or LLM play (Kaggle-compatible).
- `web_ui.py` — browser UI for human or LLM play (local).

## Generate Puzzles

```bash
python examples/agentic/sudoku/dataset_generator.py \
  --output /tmp/sudoku.jsonl --count 64 --difficulty easy --seed 2026
```

Difficulties: `easy` (~36 empty), `medium` (~46 empty), `hard` (~54 empty).

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/sudoku.jsonl \
  --dataset-loader-fn examples/agentic/sudoku/dataset_loader.py \
  --reward-fn-path examples/agentic/sudoku/reward.py \
  --agent-fn examples/agentic/sudoku/run_agent.py \
  --algo gspo --tp-size 1 --world-size 1 \
  --batch-size 1 --n-samples 2 --max-new-tokens 64
```

## Play in the Terminal

Human mode:

```bash
python examples/agentic/sudoku/tui.py --seed 7 --difficulty easy
```

LLM mode (requires a running `areno serve` endpoint):

```bash
python examples/agentic/sudoku/tui.py --agent --seed 7 \
  --base-url http://127.0.0.1:8000/v1 --model policy --api-key token
```

Actions: `inspect (row,col)`, `place (row,col,digit)`, `undo`.

## Play in the Browser

```bash
python examples/agentic/sudoku/web_ui.py --seed 7 --difficulty easy
```

Open `http://127.0.0.1:8769`. Click cells to select, then use the tool panel
to inspect, place, or undo. Enable `--agent` to let an LLM solve step-by-step.

## Reward Logic

- Solved efficiently: `0.8 + 0.2 * efficiency` (efficiency = remaining actions / max).
- Partial fill: `0.3 * fill_ratio`.
- No progress: `-1.0`.
- Invalid actions reduce the reward proportionally.

## Tests

```bash
pytest tests/test_sudoku_cpu.py
```