# Agentic 6x6 Othello Example

This example trains a policy to choose one 6x6 Othello move for the side to
move from a rendered board. The environment is deterministic, self-contained,
and reusable: legal-move enumeration with 8-direction line flipping, placement,
forced-pass handling, two-consecutive-pass terminal detection, and terminal
scoring.

The example **adds no code to `areno/`** and registers nothing. AReno discovers
it purely by file paths passed to the existing `areno train` flags.

## Files

- `game.py` — pure 6x6 Othello rules (board validation, `legal_moves`,
  `apply_move`, `flips_for`, `score_board`, terminal detection, move parsing,
  and the `score_move` reward kernel). No AReno imports.
- `dataset_generator.py` — generates reproducible, *reachable* opening
  positions by playing only legal moves from the standard opening.
- `dataset_loader.py` — `load_training_dataset(...) -> list[dict]`, converting
  JSONL boards into Areno prompt records.
- `run_agent.py` — `async def run_agent(ctx, batch) -> AgentTrajectory` issuing
  one OpenAI-compatible tool-call request per board (the `choose_move` tool
  takes `row` and `col`, each bounded to `0..5`).
- `reward.py` — `def reward_fn(record) -> float`, parsing the `choose_move`
  tool call and scoring it with `game.score_move`.
- `opponent.py` — seeded random opponent + self-play evaluation harness. Runs
  fully offline (no LLM, no network, no database).

## Input contract

- Board: a 6x6 list of rows of `B`, `W`, `.`.
- `choose_move` tool: `{"row": int, "col": int}` with `0 <= row,col <= 5`,
  `required=["row","col"]`, `additionalProperties=False`.
- XML no-tool action (optional): `<move>r,c</move>`.

## Defaults and reward convention

`score_move(board, move, player)` uses a **tiered** reward so a GSPO group with
mixed outcomes always has a non-zero gradient:

- No `choose_move` tool call, or its arguments could not be parsed → `-1.0`
  (worst tier; the model produced no actionable call). The reward function
  never raises.
- A `choose_move` call with an **out-of-range** coordinate → `-0.5`. Illegal,
  but the model did emit a call with integers, so it beats doing nothing.
- A `choose_move` call with an **in-range but illegal** cell (occupied, or no
  flanked line) → `-0.3`. Better than out-of-range: a real cell was targeted.
- A **legal, non-terminal** move → `+0.4`.
- A legal move that **ends the game** leaving `player` ahead → `+1.0`.

The illegal tiers are deliberately separated from the no-call tier (`-1.0`).
With a flat `-1.0` for every illegal action and every absent call, the
group-relative advantage for "called the tool with a bad cell" equals that of
"called no tool", so the gradient that suppresses a bad cell also suppresses
tool calling itself — driving `tool_calls` to zero and freezing reward at
`-1.0` with no gradient to recover (a cold-start collapse observed on a 0.6B
policy). The tiers make "keep calling the tool" strictly better than "stop",
so the policy is structurally pulled toward continued tool use. Defaults
(count=128, seed=2026, max-plies=8) produce Black-to-move, non-terminal,
reachable openings.

## Generate Opening Boards

```bash
python examples/agentic/othello/dataset_generator.py \
  --output /tmp/areno-othello-boards.jsonl \
  --count 2048 \
  --seed 2026 \
  --max-plies 8
```

Invalid inputs (`--count <= 0`, `--seed < 0`, `--max-plies < 0`) raise a clear
`ValueError` before any boards are generated.

## Run with Tool Calls (training)

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-othello-boards.jsonl \
  --dataset-loader-fn examples/agentic/othello/dataset_loader.py \
  --reward-fn-path examples/agentic/othello/reward.py \
  --agent-fn examples/agentic/othello/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 64 \
  --max-context-len 1024
```

If `--dataset-path` is missing or points at a directory without `boards.jsonl`,
the loader falls back to generating boards inline with the default seed.

## Offline Self-Play Evaluation (no LLM / no network)

The acceptance harness plays the policy against a seeded random opponent and
reports **win rate** and **invalid-move rate** over reproducible matches:

```bash
python examples/agentic/othello/opponent.py \
  --n-games 20 \
  --seed 2026 \
  --policy greedy \
  --policy-side B \
  --max-steps 80
```

Example output (deterministic for a given seed):

```
6x6 Othello self-play evaluation
  games          : 20
  policy side    : B
  win rate       : 0.800
  draw rate      : 0.100
  invalid rate   : 0.000
```

Structured JSON (full per-match breakdown) is also printed (or written with
`--output`). The policy choices are `random` (the demo) and `greedy`
(most-discs-flipped). Both use the same seeded RNG as the random opponent, so
every match is fully reproducible.

## Observable outputs

- During `areno train`: `rollout/rewards_mean|std|max|min`,
  `rollout/accuracy` (proxy win rate), `rollout/response_len_mean`, and the
  other auto metrics Areno records, plus the standard run-config and rollout
  artifacts under `metrics_log_dir`.
- Offline evaluation: a human-readable summary plus a structured JSON report
  with `win_rate`, `draw_rate`, `invalid_move_rate`, and per-match
  `{winner, black, white, steps, passes, invalid, terminal}`.

## Limitations

- 6x6 only; the board size is fixed and validated.
- The reward is terminal-led but uses tiered shaping (legal `+0.4`, in-range
  illegal `-0.3`, out-of-range `-0.5`, no-call `-1.0`) to break the cold-start
  zero-gradient lock; it is not a dense per-flip or disc-count reward.
- The offline harness ships `random`/`greedy` policies; wiring an LLM policy is
  done by providing a `policy_fn(board, player, rng) -> (row,col)|None` that
  calls your served model.
- No external database, sandbox, or heavyweight dependency is required or added.