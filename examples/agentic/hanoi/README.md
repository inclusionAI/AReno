# Hanoi Agentic Example

A small Towers of Hanoi agentic RL demo for AReno. The model is given a start
board and must solve it by calling the `move_disk` tool with an ordered move
sequence. Rewards come from the rules engine, so the same fixtures work for
warmup, rollout collection, or GSPO/RLVR training — with **no external
database, sandbox, or network service**.

## Rules

- Three pegs (`0`, `1`, `2`). Disks are numbered `1..n` (larger number = larger
  disk). At the start all disks sit on peg `0`, largest at the bottom.
- A move is `(source, target)` with both in `{0, 1, 2}`. Only the **top disk**
  of the source peg may be moved, and it may land only on an empty peg or on a
  **larger** disk.
- The puzzle is solved when all disks sit on peg `2` (largest at the bottom).

Illegal moves (empty source, larger-on-smaller, out of range, no-op) are
rejected by the rules engine: by default the move is **penalized** and the
state is left unchanged so the agent can keep learning
(`illegal_action_policy="penalize"`); set `"terminate"` to end the episode on
the first illegal move instead.

## Oracle and metrics

The minimum number of moves for `n` disks is:

```
optimal_steps(n) = 2 ** n - 1     # n=3 -> 7, n=6 -> 63
```

Evaluation reports two metrics:

- **completion_rate** — fraction of traces that solve the board;
- **excess_moves_over_optimum** — for completed traces, `actual_steps - 2**n-1`
  (relative to the known optimum, per the issue).

The completion reward is `1.0` minus a small efficiency penalty
(`0.02 * excess_moves`); incomplete traces score a hybrid partial credit:
**progress** (0.02 per disk correctly stacked on peg 2 from the bottom)
plus a **tiny legal-move floor** (0.005 per legal move, capped at 0.005)
to keep gradient signals alive during cold start — without being worth
freezing on. This "hybrid" design is the result of three reward iterations
that progressively fixed cold-start stalls and mode collapses.

## Input contract

The agent produces a single `move_disk` tool call per board:

```json
{"moves": [[0, 2], [0, 1], [2, 1], [0, 2], [1, 0], [1, 2], [0, 2]]}
```

Each element is `[source, target]`, both integers in `{0, 1, 2}`. The tool name
is always `move_disk`; `source`/`target` are arguments, not tool names. `n` is
restricted to `[3, 6]`.

## Defaults

- Disk range: `n ∈ [3, 6]`.
- `illegal_action_policy`: `"penalize"` (illegal → small penalty, state
  unchanged, episode continues).
- The feature is **opt-in** and defaults to disabled; existing AReno behavior
  is unchanged when it is not enabled.

## Output fields

`replay(trace, n)` returns a `ReplayResult` with both human-readable and
structured forms:

```python
{
    "steps": 7,
    "legal_count": 7,
    "illegal_count": 0,
    "completed": true,
    "excess_moves": 0,
    "final_pegs": [[], [], [3, 2, 1]],
}
```

`evaluate(traces, n)` returns `{n, sample_count, completion_rate,
avg_excess_moves, oracle_steps, results}`.

When exposed through the CLI, both human-readable text and structured JSON are
emitted, and invalid input produces a clear validation error identifying the
affected move and reason.

## Copyable example

Generate deterministic fixtures (16 records: 4 scenarios × `n=3..6`):

```bash
python examples/agentic/hanoi/dataset_generator.py --output /tmp/hanoi_fixtures.jsonl
```

Inspect one fixture:

```bash
python examples/agentic/hanoi/dataset_generator.py --count 1
```

Run the focused CPU tests:

```bash
python -m pytest tests/test_agentic_hanoi_example_cpu.py -q
```

Replay an optimal trace from Python (no model, no training run):

```python
from examples.agentic.hanoi import game

trace = game.serialize_trace(game.optimal_solution(3))  # "0->2,0->1,..."
print(game.replay(trace, 3).as_text())
# -> completed, excess_moves_over_optimum=0
```

A boundary/invalid input — attempting to move from an empty peg (peg `1` is
empty at the start) is rejected, then the board is still solved. The illegal
attempt is counted but, under the default `penalize` policy, does not end the
episode:

```python
boundary = "1->2," + game.serialize_trace(game.optimal_solution(3))
print(game.replay(boundary, 3).as_text())
# -> replay: 8 steps (7 legal, 1 illegal) -> completed
#    excess_moves_over_optimum=0
```

## Training effect (observed)

A 100-step GSPO run (Kaggle 2×T4, Qwen3.5-0.8B, n_samples=4) shows:

- **reward_mean** activates from step 0 (0.00125), trending up to the floor cap
  of 0.005 by step 5, with multi-turn trajectory lengths up to 17k tokens.
- **Multi-turn agent** enables 40-65 tool calls per item (vs ≤8 single-turn),
  with tool_results non-zero (board state is fed back each turn).
- **grad_zero_ratio** ≈ 0.25 on active steps (~75% of parameters update), with
  collapse-to-cap self-healing in 1-2 steps.
- The hybrid floor (0.005 cap) prevents stable reward hacking, but the 0.8B
  model's capacity limits convergence to completion on n≥4. This is a training
  experiment question, not a demo-correctness one.

Further improvement to convergence on harder board sizes (n ≥ 4) would require
switching `run_agent.py` to a multi-turn design or warmup via SFT; this is a
training experiment question rather than a demo-correctness one.

## Limitations

- Only `n = 3..6` disks (per the issue). Larger `n` is rejected by `make_state`.
- Runs entirely on CPU; no GPU is required to exercise the rules engine,
  fixtures, replay, or evaluation.
- No external database, hosted control plane, sandbox, or network service.
- Adopts AReno's existing agentic contracts (`areno/api/agentic.py`); it does
  not replace the trainer, rollout engine, or SDK.
- Purely opt-in: the example adds no AReno CLI flag, config default, or public
  API change, so existing AReno behavior is unchanged when the example is not
  invoked.

## Files

| File | Role |
| --- | --- |
| `game.py` | Rules engine: state, `move`, legality, reward, oracle, trace replay, evaluation |
| `dataset_generator.py` | Deterministic scripted fixtures (optimal / contains_illegal / boundary / failure) |
| `dataset_loader.py` | Load JSONL fixtures and build AReno prompt records |
| `reward.py` | `reward_fn(record)` — score a `move_disk` completion via the rules engine |
| `run_agent.py` | `async run_agent(ctx, batch)` — issue `move_disk` tool calls via the rollout proxy |
