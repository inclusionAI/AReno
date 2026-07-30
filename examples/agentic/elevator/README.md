# Elevator Dispatch — Agentic RL Demo

This example builds a focused, independently reviewable **elevator-dispatch
agentic RL** demo on top of AReno's existing public contracts. It models one
elevator cab serving N floors under a discrete time-step clock, exposes
`move` / `open_door` / `close_door` actions, and advances a local event queue
deterministically — no external services, sandboxes, or databases are used.

The demo targets the acceptance areas from issue #195:

- **Overload protection** — a cab capacity limit refuses extra boarding.
- **Empty-door invalid actions** — moving with the door open is rejected.
- **Concurrent requests** — passengers waiting on multiple floors at `t0`.
- **Peak traffic** — high arrival density over a long horizon.
- **Termination** — short horizons force early episode cutoff.

A First-Come-First-Served (FCFS) baseline is included to compare a trained
agent against a hand-coded dispatcher on delivered passengers, mean waiting
time, and illegal actions.

---

## Files

| File                       | Role                                                                 |
| -------------------------- | -------------------------------------------------------------------- |
| `game.py`                  | Deterministic single-cab environment: state, actions, `step`, prompt |
| `dataset_generator.py`     | Deterministic JSONL scenarios for the five acceptance areas          |
| `dataset_loader.py`        | `load_training_dataset(dataset_path, *, default_loader, **_)` loader |
| `reward.py`                | `reward_fn(record)` replays the trajectory and scores the outcome    |
| `run_agent.py`             | Bounded multi-turn `run_agent(ctx, batch)` agent loop                |
| `fcfs_baseline.py`         | FCFS baseline runner with human-readable + JSON output               |

CPU tests live under `tests/`:

- `test_elevator_game_cpu.py`
- `test_elevator_reward_cpu.py`
- `test_elevator_agentic_cpu.py`
- `test_elevator_fcfs_baseline_cpu.py`
- `test_elevator_scenarios_cpu.py`

---

## Input contract

Each record is a JSON object with the following fields. The loader validates
every field before any model or worker starts, so malformed inputs fail fast.

| Field        | Type            | Default   | Notes                                                    |
| ------------ | --------------- | --------- | -------------------------------------------------------- |
| `floors`     | int             | `6`       | `>= 2`; floor ids run `0 .. floors-1`                    |
| `capacity`   | int             | `4`       | `>= 1`; maximum passengers inside the cab                |
| `horizon`    | int             | `64`      | `>= 1`; maximum time steps per episode                   |
| `scenario`   | str             | `"mixed"` | one of the `SCENARIOS` below                             |
| `door_open`  | bool            | `False`   | initial door state (the `empty_door` scenario forces `True`) |
| `id`         | str             | auto      | record id; auto-assigned if missing                      |
| `passengers` | list[object]    | required  | one object per passenger                                 |

Each passenger object:

| Field         | Type | Default | Notes                                              |
| ------------- | ---- | ------- | -------------------------------------------------- |
| `pid`         | int  | auto    | passenger id; auto-assigned if missing             |
| `origin`      | int  | —       | floor where the passenger arrives (`0 .. floors-1`)|
| `dest`        | int  | —       | destination floor (`0 .. floors-1`, `!= origin`)   |
| `arrive_time` | int  | `0`     | time step at which the passenger starts waiting    |

Validation rejects: empty passenger lists, out-of-range floors, `origin == dest`,
non-positive `capacity`/`horizon`, and records missing required passenger fields.

---

## Scenarios

`dataset_generator.py` exposes `SCENARIOS` and emits deterministic records for
each (same seed → same records):

| Scenario     | Characteristics                                              |
| ------------ | ------------------------------------------------------------- |
| `overload`   | `capacity = 1`, `>= 4` passengers at `t0` on random floors   |
| `empty_door` | `door_open = True` at `t0` so the first move is invalid       |
| `concurrent` | `>= 2` distinct origins at `arrive_time = 0`                 |
| `peak`       | `horizon >= 64`, `>= 8` passengers                            |
| `terminate`  | `horizon <= 4` so most passengers cannot be delivered         |
| `mixed`      | default blend drawing from the above                          |

---

## Action tools

The environment exposes three deterministic tools to the model. Each tool call
advances the clock by exactly one step and returns an observation.

| Tool         | Arguments                 | Effect                                                            |
| ------------ | ------------------------- | ----------------------------------------------------------------- |
| `move`       | `direction: {+1, -1}`     | Move the cab one floor up/down. Invalid if the door is open.      |
| `open_door`  | (none)                    | Open the door; alight delivered passengers then board waiting ones up to capacity. Refused boardings are counted. Invalid if already open. |
| `close_door` | (none)                    | Close the door. Invalid if already closed.                       |
| `done`       | (none)                    | End the episode early.                                            |

An episode also ends naturally when the horizon is reached or when `is_terminal`
returns `True` (all passengers delivered).

---

## Reward

`reward_fn(record)` replays the trajectory recorded in `record.tool_calls` against
`game.build_state(record.source_record)` and combines three signals:

```text
reward = +DELIVER_WEIGHT  * delivered_passengers / total_passengers
         - WAIT_WEIGHT    * normalized_total_wait
         - INVALID_WEIGHT * invalid_actions / total_passengers
```

- `delivered_passengers / total_passengers` — coverage in `[0, 1]`.
- `normalized_total_wait` — sum of waiting steps across delivered passengers,
  divided by an upper bound, in `[0, 1]`.
- `invalid_actions / total_passengers` — per-passenger illegal-action rate.

Weights are module-level constants (`DELIVER_WEIGHT = 1.0`,
`WAIT_WEIGHT = 0.3`, `INVALID_WEIGHT = 0.5`) so tuning is centralized. The
reward is a plain `float`; malformed arguments are caught and treated as an
invalid action with a `0.5 / total_passengers` penalty, so a bad trajectory
never crashes training. When `total_passengers == 0` the reward is `0.0` by
guard (no division by zero).

---

## Quick start

### 1. Generate a dataset

```bash
python examples/agentic/elevator/dataset_generator.py \
  --count 256 --seed 2026 --scenario mixed \
  --out /tmp/elevator_train.jsonl
```

Any of the scenarios listed above can be passed via `--scenario`. The same
`--seed` reproduces the exact same records.

### 2. Score the FCFS baseline

```bash
python examples/agentic/elevator/fcfs_baseline.py \
  --dataset /tmp/elevator_train.jsonl
```

Add `--json` for machine-readable output, or omit `--dataset` to run the default
deterministic dataset.

### 3. Train an agent

Point AReno at the example directory. The exact flags follow AReno's agentic
training recipe; the example-specific pieces are stable:

- `--dataset-path /tmp/elevator_train.jsonl`
- `--dataset-loader examples/agentic/elevator/dataset_loader.py`
- `--reward-fn-path examples/agentic/elevator/reward.py`
- `--agentic-runner examples/agentic/elevator/run_agent.py`

The agent calls `move` / `open_door` / `close_door` each turn; the environment
runs in-process and the new state is fed back as a tool result. The loop ends
on `done`, a terminal state, or the horizon cap.

### 4. Compare against FCFS

After training, re-run `fcfs_baseline.py` on the evaluation split and compare
the emitted fields (see below) with the trained agent's metrics.

---

## Output fields

`fcfs_baseline.py` and the reward replay expose these fields per episode and
aggregated:

| Field                     | Level  | Meaning                                          |
| ------------------------- | ------ | ------------------------------------------------ |
| `delivered`               | both   | passengers delivered this episode                |
| `total_passengers`        | both   | initial passenger count                          |
| `mean_wait`               | ep.    | per-episode mean waiting steps over delivered passengers |
| `mean_wait_per_passenger` | agg.   | global `sum(total_wait) / sum(total_passengers)` |
| `invalid_actions`         | both   | count of rejected actions this episode           |
| `overload_refused`        | both   | boardings refused due to capacity limit          |
| `delivery_rate`           | agg.   | `sum(delivered) / sum(total_passengers)`         |
| `by_scenario`             | agg.   | per-scenario breakdown of the above              |

The FCFS baseline intentionally scores low on `terminate` (horizon too short to
deliver anyone) and `overload` (capacity 1) — this leaves headroom for a
trained agent to learn a better dispatch policy.

---

## Defaults

Defaults are chosen so the example runs out of the box without any flags:

- `DEFAULT_FLOORS = 6`, `DEFAULT_CAPACITY = 4`, `DEFAULT_HORIZON = 64`.
- Default scenario is `mixed`.
- `door_open` defaults to `False` unless the scenario overrides it.
- Reward weights: `DELIVER_WEIGHT = 1.0`, `WAIT_WEIGHT = 0.3`,
  `INVALID_WEIGHT = 0.5`.

These follow AReno's default-value guidelines: they are safe, predictable, and
aligned with the issue's acceptance points.

---

## Limitations

- **Single cab only.** The environment models one elevator; multi-cab central
  dispatch is out of scope for this demo.
- **Discrete time steps.** The clock advances one step per action; continuous
  event queues are not modeled.
- **No external traffic model.** Passenger arrivals come from the generator's
  deterministic fixtures, not a real traffic trace.
- **CPU-only tests.** The five test files run without a GPU; they verify
  environment semantics, reward monotonicity, the agent loop, the FCFS
  baseline, and the five scenario fixtures.

---

## Reproducing the tests

```bash
pytest tests/ -k elevator
```

All elevator tests are CPU-only and deterministic. Each test loads the example
modules via `importlib` without importing the `areno` package, so they run in a
plain Python environment without CUDA or PyTorch.
