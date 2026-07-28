# Agentic Towers of Hanoi Example

This example trains a policy to solve the Towers of Hanoi puzzle for 3 to 6
disks. The policy repeatedly calls a single tool, `move(source, target)`, to
transfer every disk from peg `A` to peg `C`. The environment is fully
deterministic, CPU-only, and self-contained: no network services, sandboxes, or
external databases are required.

## Files

- `game.py` -- state, legality, the optimal-move oracle, rendering, and
  `score_episode` (completion + excess-over-optimum efficiency).
- `dataset_generator.py` -- reproducible `n in {3..6}` tasks written as JSONL.
- `dataset_loader.py` -- validates each task and attaches the per-task prompt.
- `run_agent.py` -- the bounded multi-turn episode loop (one `move` call/turn).
- `reward.py` -- replays emitted `move` calls through `score_episode`.

## Task contract

Each task record stores the disk count `n` and a generous move cap `max_moves`
(twice the optimum plus slack). The optimum `2**n - 1` is **not** stored on the
record; the reward path recomputes it via `game.optimal_steps`, so a record
cannot leak the answer to the model.

The only tool is:

```python
move(source: "A" | "B" | "C", target: "A" | "B" | "C")
```

An episode ends when the puzzle is solved, the move cap is reached, or the model
emits an **illegal move** (empty source peg, source equals target, or a larger
disk placed on a smaller disk). Illegal moves terminate the rollout.

## Scoring

`score_episode(n, moves)` replays the move sequence against a fresh start state
and returns:

| Field         | Meaning                                                    |
| ------------- | ---------------------------------------------------------- |
| `completed`   | All disks reached peg `C`.                                 |
| `illegal`     | An illegal move was encountered before completion.         |
| `num_moves`   | Moves replayed until termination.                          |
| `excess`      | `max(0, num_moves - optimal)` when completed, else `None`. |
| `efficiency`  | `max(0, 1 - excess / optimal)` when completed, else `0`.   |
| `reward`      | `0.5 * completed + 0.5 * efficiency`; illegal/incomplete = `0.0`. |

`COMPLETION_WEIGHT` and `EFFICIENCY_WEIGHT` in `game.py` tune the split without
changing the scoring contract. A solved optimal run scores `1.0`; a solved run
with extra moves scores less but still rewards completion; any illegal or
incomplete run scores `0.0`.

## Generate tasks

```bash
python examples/agentic/hanoi/dataset_generator.py \
  --output /tmp/areno-hanoi-tasks.jsonl \
  --count 128 --seed 2026 --min-disks 3 --max-disks 6
```

## Train

```bash
areno train \
  --ckpt Qwen/Qwen3-0.6B \
  --dataset-path /tmp/areno-hanoi-tasks.jsonl \
  --dataset-loader-fn examples/agentic/hanoi/dataset_loader.py \
  --reward-fn-path examples/agentic/hanoi/reward.py \
  --agent-fn examples/agentic/hanoi/run_agent.py \
  --algo gspo \
  --batch-size 1 \
  --n-samples 2 \
  --max-new-tokens 64
```

## Observable output

During rollout the trainer logs per-step agentic diagnostics: number of
samples, total tokens, message/tool-call/tool-result counts, and the share of
trajectories filtered for exceeding the context length. The reward records
expose `completed`, `excess`, and `efficiency`, so a completed-but-inefficient
rollout is distinguishable from a failed one.

## Reproducible oracle and trace replay

`game.optimal_moves(n)` returns the recursive shortest solution and
`game.optimal_steps(n)` returns `2**n - 1`. Replaying `optimal_moves(n)` against
`score_episode(n, ...)` yields `completed=True`, `excess=0`, `reward=1.0` for
every supported `n`; this is covered by the CPU tests and gives a deterministic
fixture independent of any model.
