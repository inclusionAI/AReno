# AReno Watch — Terminal Training Monitor

`areno watch` is a lightweight, in-terminal monitor for AReno training runs. It
reads the low-latency status file that trainers already write on every stage
transition and renders **live progress, loss, reward, throughput, and ETA**
without a browser, TensorBoard, or any extra backend. Pair it with
`areno runs` to list every run stored on the machine.

It is intentionally local and read-only: watching a run does **not** touch the
training process, and stopping the watcher (Ctrl+C) leaves training running.

## What it shows

The TTY panel renders one framed status block per refresh:

```
╔════════════════════════════════════════════════════════════════╗
║  AReno Watch - Run status                                       ║
╠════════════════════════════════════════════════════════════════╣
║  Step: 150/1000  ████░░░░░░░░░░░░░░░░  15.0%                   ║
║  Loss: 0.4321    Reward: 0.2187                                  ║
║  Throughput: 612 tok/s                                          ║
║  Stage: train    ETA: 12m 4s                                    ║
║  Updated: 3s ago                                                 ║
╚════════════════════════════════════════════════════════════════╝
```

* **Step progress bar** — `step` vs `total_steps`, with a filled bar and
  percentage.
* **Loss / Reward** — the loss and reward mean recorded at the most recent
  `train_end` stage (shown as `N/A` until the first training step writes them).
* **Throughput** — tokens-per-second when the trainer reports it.
* **Stage + ETA** — the current lifecycle stage, color-coded, plus a
  linear-projection estimate of remaining time.
* **Updated** — how many seconds since the status file was last written.

Color is applied per stage (`rollout`=cyan, `reward`=yellow,
`advantage`=magenta, `train`=green) and auto-stripped when output is not a TTY.

## Quick start

```bash
# Watch the most recent run (auto-discovers the status file)
areno watch --latest

# Watch a specific run by its run ID
areno watch --run-id 20240115_143022

# List every run on this machine
areno runs
areno runs --verbose     # adds stage and last-updated columns
```

A typical workflow is to start training in one terminal and tail it from
another:

```bash
# terminal 1
areno train --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
  --reward-fn-path examples/math/math_verify_reward.py --algo gspo --tp-size 1

# terminal 2
areno watch --latest
```

## Output modes

`areno watch` picks a renderer based on the flag and whether stdout is a TTY.

| Mode | When | Shape |
| --- | --- | --- |
| **TTY panel** | stdout is a terminal, not `--quiet` | Framed, colorized block, screen cleared each refresh |
| **Single line** | non-TTY or `--quiet` | `[timestamp] Step 150/1000 \| Loss 0.4321 \| ...` |
| **JSON Lines** | `--json` | One JSON object per refresh, ideal for piping into a log |

Example line mode:

```text
[2026-08-21 14:30:05] | Step 150/1000 | Loss 0.4321 | Reward 0.2187 | Stage=train | Status=running
```

Example JSON mode:

```json
{"step": 150, "total_steps": 1000, "loss": 0.4321, "reward": 0.2187,
 "throughput": 612, "eta_seconds": 724, "stage": "train", "status": "running",
 "pid": 12345, "updated_at": 1724254205.1}
```

Pipe JSON output for structured logging:

```bash
areno watch --latest --json >> run.jsonl
```

## Options

```text
areno watch --help
```

| Option | Default | Description |
| --- | --- | --- |
| `--run-id <id>` | — | Run ID to watch (directory under `~/.areno/runs/`). |
| `--latest` | off | Watch the most recent run (mutually convenient with `--run-id`). |
| `--interval <s>` | `1` | Refresh interval in seconds (must be ≥ 1). |
| `--json` | off | Emit JSON Lines instead of the text panel. |
| `--quiet` | off | Suppress header/footer messages; forces line output. |
| `--no-header` | off | Same as `--quiet` for header suppression. |
| `--timeout <s>` | `0` | Exit after N seconds (`0` = unlimited). |
| `--tail <N>` | — | Show only the last N lines (line mode, for log tailing). |
| `--fields <list>` | all | Comma-separated subset: `step,loss,reward,throughput,eta,stage,status`. |

Field filtering is useful for compact logs:

```bash
areno watch --latest --quiet --fields step,loss,reward --interval 2
```

## Where the data comes from

`areno watch` reads a single status file; it performs no IPC with the training
process. The producer is `areno.api.metrics.MetricsRecorder`, which every
trainer calls through `areno.api.dashboard.record_dashboard_state(...)` at each
stage transition. The recorder writes the file **atomically**
(write-to-`.tmp`, then `replace`) so the watcher never reads a half-written
payload.

**Default status file location:**

```
<metrics-log-dir>/dashboard_state.<pid>.json      # default dir: /tmp/areno/tfevent
```

`--metrics-log-dir` on `areno train` controls where the file is written.

**Discovery order used by `areno watch`:**

1. `~/.areno/runs/<run-id>/dashboard_state.*.json` (newest first)
2. `/tmp/areno/tfevent/dashboard_state.*.json` (fallback, newest first)

`areno watch --latest` walks the same order to pick the most recent run ID
(run directories first, then tfevent files keyed by PID).

> Note: `areno runs` lists directories under `~/.areno/runs/`. Runs that only
> wrote a tfevent status file are discoverable by `areno watch` (via the
> fallback) but will not appear in `areno runs` until they have a run
> directory.

**Status file format** (fields the watcher consumes):

| Field | Type | Source |
| --- | --- | --- |
| `pid` | int | writer process — used to detect whether training is still alive |
| `stage` | str | lifecycle stage at write time |
| `status` | str | run state (`running`, `completed`, `error`, `stopped`) |
| `updated_at` | float | epoch seconds of the write |
| `step` | int? | current global step |
| `epoch` | int? | current epoch |
| `role` | str? | model role, e.g. `policy` |
| `loss` | float? | last training-step loss (written at `train_end`) |
| `reward_mean` | float? | mean reward of the last materialized batch (written at `train_end`) |
| `total_steps` | int? | configured `max_steps`, used by the progress bar |
| `throughput` | int? | tokens-per-second when reported |

## Stage reference

The trainer writes a status record at each transition, so the `Stage:` field
walks the training lifecycle in real time:

```text
epoch_start
  └─ rollout_start → rollout_end      (sampling / agentic execution)
       └─ train_start → train_end     (optimizer step; loss & reward_mean land here)
            └─ (train_skip on critic-only warmup steps)
epoch_end
max_steps_reached                     (terminal)
```

The watcher exits on its own when the training process is no longer alive
(`pid` check) or `status` reports `completed` / `error` / `stopped`.

## Behavior notes

* **Graceful exit** — SIGINT/SIGTERM are trapped; the loop stops cleanly and
  training is unaffected. The watcher never sends signals to the trainer.
* **Duplicate suppression** — in TTY/JSON modes, unchanged status is not
  re-rendered; pass `--tail` to always emit a fresh line.
* **Transient read failures** — if the status file is momentarily unavailable
  (it is being atomically replaced), the watcher retries on the next interval.
* **ETA** — linear projection from `(total_steps - step) / rate`, where `rate`
  is `step / elapsed`. Returns `N/A` before the first step and `done` once the
  step count reaches `total_steps`.

## Module layout

```
areno/cli/watch.py     # watch_command, runs_command, renderers, ETA, discovery
areno/api/metrics.py   # MetricsRecorder.record_dashboard_state (the writer)
areno/api/dashboard.py # record_dashboard_state() dispatcher used by trainers
areno/cli/main.py      # registers `areno watch` and `areno runs`
tests/test_watch_cpu.py # CPU test suite (renderers, ETA, discovery, active check)
test_watch_demo.py     # standalone demo that mocks the status data flow
```

Run the CPU tests without a GPU:

```bash
pytest tests/test_watch_cpu.py
```
