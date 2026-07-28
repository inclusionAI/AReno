# Agentic Sudoku Example

A focused, self-contained agentic-RL environment: a policy solves a uniquely
solvable Sudoku puzzle by calling three tools (`inspect_candidates`,
`place_digit`, `undo`) over multiple turns under a bounded action budget. The
environment is pure Python, CPU-only, and has no network or sandbox
requirements. The solution is **never** exposed to the policy — grading relies
on the visible-board invariant (a uniquely solvable board is solved iff it is
filled with no row/column/box conflict).

## Files

- `sudoku.py` — the environment: unique-solution generator, the three tools,
  per-call validation, and the no-solution terminal check.
- `dataset_generator.py` — reproducible JSONL puzzle generator (difficulty bands).
- `dataset_loader.py` — JSONL → AReno prompt records.
- `run_agent.py` — multi-turn agent loop (lockstep turns, in-memory env).
- `reward.py` — episode grading from `place_digit` tool results.

## Why multi-turn

Unlike the Tic-Tac-Toe example (one tool call per board), Sudoku is a sequence
of moves. `run_agent.py` runs a turn-by-turn loop that mirrors the coding-agent
loop: each turn it sends the current messages + tool schemas, parses the first
tool call, executes it on an in-memory `SudokuEnv`, and appends the JSON result
as a `role: tool` message. An episode ends when the board is solved or the
action budget is exhausted.

## Input contract

Each JSONL record (produced by `dataset_generator.py`):

| field | type | meaning |
|---|---|---|
| `id` | str | stable record id |
| `difficulty` | str | one of `easy`, `medium`, `hard`, `extreme` |
| `seed` | int | seed used to generate the puzzle (reproducibility) |
| `action_budget` | int | max tool calls per episode |
| `puzzle` | int[9][9] | visible board, `0` = empty |

The solution is deliberately **not** stored.

## Defaults and backward compatibility

- `difficulty` defaults to `medium`; `action_budget` defaults to `81`.
- When the dataset loader cannot find the JSONL path, it falls back to
  in-memory generation, so the example runs with no external files.
- This example only adds files under `examples/agentic/sudoku/`; it does not
  modify AReno's core API, trainer, or registry, so behavior is unchanged when
  the example is not selected.

## Generate puzzles

```bash
python examples/agentic/sudoku/dataset_generator.py \
  --output /tmp/areno-sudoku-puzzles.jsonl \
  --count 256 \
  --seed 2026 \
  --difficulties easy,medium,hard,extreme
```

## Train (GPU)

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-sudoku-puzzles.jsonl \
  --dataset-loader-fn examples/agentic/sudoku/dataset_loader.py \
  --reward-fn-path examples/agentic/sudoku/reward.py \
  --agent-fn examples/agentic/sudoku/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 256
```

`areno train` serves its own OpenAI-compatible policy server; `run_agent.py`
connects to it, so no external model API is required.

## Observable output

- Per episode, tool results carry: `action`, `placed`/`undone`,
  `invalid_action`, `reason` (on rejection), `solved`, `board`, `board_text`,
  `actions_remaining`, `is_terminal`, `difficulty`.
- Reward (from `reward.py`): `1.0` solved, `0.0` no solve but ≥1 legal move,
  `-0.1` no legal move at all.
- Metrics to wire into the trainer config, grouped by `difficulty`:
  `solve_rate` and `invalid_action_rate`. Both are computable from the same
  `place_digit` tool results used by the reward function.

## Design decisions / TODO

- **Illegal placement semantics (confirm):** `place_digit` on a cell that is
  filled, a given, or where the digit conflicts with its row/column/box is
  *rejected* — the board is unchanged, the call still costs one action, and the
  result is flagged `invalid_action=True`. This keeps the board clean and gives
  the policy a learnable negative signal. To switch to "accept + penalize"
  instead, change `SudokuEnv.place_digit` / `_reject`.
- **Difficulty bands** are defined by retained clue counts in
  `sudoku.DIFFICULTY_CLUES`. Swap to a reasoning-skill taxonomy if preferred.
- **Metrics:** `reward.py` returns a scalar; per-difficulty `solve_rate` /
  `invalid_action_rate` should be registered in the trainer's metric config
  (reuse AReno's existing metric fields rather than introducing new ones).

## Limitations

- Generation cost rises with `extreme` (more digging + uniqueness re-checks);
  for tests use small counts or pre-generated fixtures.
- The uniqueness solver is exponential in the worst case but trivial for 9×9
  in practice; tests should rely on fixed seeds, not random generation.
- No GPU, network, sandbox, or external database is required for any of the
  example code; only training (`areno train`) needs a GPU.