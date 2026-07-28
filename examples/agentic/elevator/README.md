# Agentic Elevator-Dispatch Example

This example trains a policy to dispatch an elevator: it plans a sequence of
move/open/close actions that picks up arriving passengers, delivers them to
their destinations, and minimizes wait while never exceeding car capacity.
Each prompt describes the building; the model calls a `dispatch` tool with an
action string. AReno replays the episode deterministically and scores it. The
environment is deterministic and self-contained.

## Files

- `game.py` is the pure-Python elevator engine: building validation, a door
  state machine, capacity-bounded boarding (overload prevention), a deterministic
  event queue, episode replay, terminal detection, prompt formatting, and the
  FCFS baseline policy.
- `dataset_generator.py` generates reproducible building JSONL (seeded arrival
  schedules).
- `dataset_loader.py` converts JSONL records into Areno prompt records.
- `run_agent.py` is the agent entrypoint: one `dispatch` tool call per building.
- `reward.py` replays the model's action string with `game.play`, writes the
  episode metrics onto the source record, and returns a scalar shaped by
  delivered passengers, mean wait, and invalid-action rate.
- `baseline.py` reports a first-come-first-served (FCFS) baseline so
  trained-vs-baseline improvement is measurable.

## Action Letters

- `U` move the car up one floor; `D` move down one floor.
- `O` open the door and exchange passengers (drop off matches, then board up to
  capacity from the floor's waiting queue).
- `C` close the door.

Invalid actions (wrong door state, moving past the top/bottom floor) are skipped
but counted in `n_invalid`. Each action advances the clock by one tick, so
arrivals land deterministically.

## Generate Buildings

```bash
python examples/agentic/elevator/dataset_generator.py \
  --output /tmp/areno-elevator.jsonl \
  --count 2048 \
  --seed 2026 \
  --arrivals 6
```

## Run with Tool Calls

```bash
areno train \
  --ckpt Qwen/Qwen3-1.7B \
  --dataset-path /tmp/areno-elevator.jsonl \
  --dataset-loader-fn examples/agentic/elevator/dataset_loader.py \
  --reward-fn-path examples/agentic/elevator/reward.py \
  --agent-fn examples/agentic/elevator/run_agent.py \
  --algo gspo \
  --batch-size 2 \
  --n-samples 4 \
  --max-new-tokens 96
```

(This command trains on GPU. The engine, generator, loader, reward, and
baseline run on CPU with no network access.)

## Report a FCFS Baseline

```bash
python examples/agentic/elevator/baseline.py \
  --count 256 \
  --seed 2026 \
  --arrivals 6
```

The output prints mean delivered passengers, mean wait, and mean invalid-action
rate. Compare a trained run against this baseline to report improvement.

## Episode Replay Contract

`game.play(building, actions, *, max_steps)` returns:

- `delivered_passengers`: passengers dropped at their destination.
- `mean_wait` / `max_wait` / `total_wait`: wait time in ticks.
- `n_steps` / `n_attempts` / `n_invalid` / `invalid_rate`: action accounting.
- `remaining_passengers`: aboard + waiting + pending-arrival passengers.
- `terminal`: whether nothing remains to dispatch (game over).

The same `building + actions + max_steps` is bit-identical across runs.
