# Issue #257 — Track GPU memory and utilization history for a run

Design + implementation plan. Drafted by the contributor picking up the issue;
not yet reviewed/merged.

## Goal (one line)

Sample per-device GPU memory, utilization, and temperature at a bounded
interval during a training run, retain a bounded local history linked to the
run, print a CLI summary at the end, and degrade cleanly when NVIDIA tooling is
unavailable — without touching the training hot path.

## Constraints from the issue (and how this design honors each)

| Issue requirement | Design response |
|---|---|
| Periodic, bounded-interval sampling, bounded local history linked to the run | Daemon thread polls every `interval_s`; samples held in a bounded `deque(maxlen=max_history)`; artifacts written next to the run's other per-pid artifacts in `metrics_log_dir` |
| Degrade cleanly when NVIDIA tooling is unavailable | `shutil.which("nvidia-smi")` missing → sampler is a no-op (started, emits nothing, summary reports `n_samples=0`, `"reason": "nvidia-smi not found"`). Reuses the exact degrade pattern from `diagnostics.py:_nvidia_smi_driver_info` and `dashboard/server.py:runtime_env` |
| Use existing public contracts / local artifact formats; no parallel subsystem | Reuses `metrics_log_dir` (same dir AReno already writes TB events, `dashboard_state.*.json`, `rollout_samples.*.jsonl`, `areno_run_config.{pid}.{txt,json}` into); per-pid JSONL+JSON filename convention mirrors `_write_dashboard_run_config`. No new dependency, no new process, no new DB |
| New public option must have a safe default (preserve current behavior), clear validation error, human-readable + structured output | `--gpu-stats` flag defaults **off** → zero behavior change. `--gpu-stats-interval-s` and `--gpu-stats-history` default `5.0`/`1000`, validated `> 0` via the same positive-field path as `--epochs` etc. CLI summary is human-readable text; `gpu_stats_summary.{pid}.json` is structured |
| `areno/cli/`, existing local run artifacts/metrics, CPU tests; narrow change; reuse existing data/metric/lifecycle/registry contracts | All code under `areno/cli/`; one new module `gpu_stats.py` + flag group in `train.py`; CPU tests under `tests/`. Lifecycle hangs off the existing `run()` → `fit()` boundary |
| Don't block the training hot path | A single daemon thread runs `subprocess.run(nvidia-smi, ...)` off the main loop; sampling never enters `engine/`, `trainer.fit()`, or the loss path. The main thread only touches `GPUSampler` at start/stop |
| Multi-GPU device mapping, missing fields, sampler shutdown, history bounds, CLI summaries — all testable without blocking the hot path | `GPUSampler` takes an injectable `sample_fn` so tests fake `nvidia-smi` output (incl. missing columns and N-device rows) without a GPU; shutdown tested via `stop()`+`.join`; history bounds via over-filling the deque |

## Non-goals (explicit, to keep the diff reviewable)

- No sampling inside `engine/` or any loss / rollout path.
- No new runtime dependency. `shutil`/`subprocess`/`threading`/`json`/`collections`
  are stdlib and already used elsewhere in the CLI.
- No dashboard UI work beyond emitting the JSON summary the dashboard *could*
  later read; the dashboard rewrite is a separate issue.
- No automatic config mutation, artifact deletion, or process control.

## Architecture

### New module: `areno/cli/gpu_stats.py` (core; engine-agnostic)

```python
@dataclass(frozen=True, slots=True)
class GPUSample:
    timestamp_s: float          # time.perf_counter() of the tick
    index: int                  # device index from nvidia-smi
    name: str | None
    mem_used_mb: int | None
    mem_total_mb: int | None
    util_pct: int | None
    temp_c: int | None

class GPUSampler:
    def __init__(self, *, interval_s: float, max_history: int,
                 devices: list[int] | None = None,
                 sample_fn: Callable[[], list[GPUSample]] | None = None): ...
    def start(self) -> None: ...                  # daemon thread; no-op if nvidia-smi absent
    def stop(self) -> None: ...                   # idempotent; joins thread, flushes one final tick
    def sample_once(self) -> list[GPUSample]: ...  # subprocess nvidia-smi -> parse -> list[GPUSample]
    def history(self) -> list[GPUSample]: ...      # snapshot of bounded deque
    def dump_jsonl(self, path) -> int: ...         # append one JSON line per (tick, device)
    def write_summary(self, path) -> dict: ...     # structured summary JSON
    def summary_text(self) -> str: ...             # human-readable CLI summary block

# nvidia-smi parsing helper, separable for unit testing
def parse_nvidia_smi_csv(stdout: str) -> list[GPUSample]: ...
```

Design notes:

- `sample_fn` injection is the **CPU-test seam**: tests pass a fake that
  returns canned `GPUSample` lists (success / missing columns / multi-device /
  empty), so no GPU and no real `subprocess` is needed for the core-logic suite.
- Default `sample_fn` calls `subprocess.run([nvidia-smi,
  --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,
  --format=csv,noheader,nounits], ...)`, wrapped in try/except that yields `[]`
  on any failure — `parse_nvidia_smi_csv` is pure and tested separately.
- `temp_c` may be absent on some drivers/boards → keep it optional; missing
  field test asserts the sample still parses with `temp_c=None` (covers the
  "missing fields" acceptance item).
- Bounded history: `collections.deque(maxlen=max_history)`; the "history
  bounds" test over-pushes and asserts exactly `max_history` retained and the
  oldest samples dropped.
- Thread safety: the worker thread is the sole writer after `start()`; `stop()`
  sets an `Event`, joins, and the main thread is then the only accessor. No lock
  needed on the hot path; a lock guards only the deque swap in `history()`/
  `stop()` for the snapshot.
- Shutdown: forward the existing graceful-shutdown intent — `stop()` is
  idempotent and safe to call from a `finally` block; the daemon flag means a
  crashed `fit()` never leaves a hanging thread.

### Lifecycle hook: `areno/cli/train.py` (`run()`)

`run()` already defines the run boundary (`_write_dashboard_run_config` →
construct `Trainer` → `trainer.fit()`). The sampler rides that boundary:

```python
def run(trainer_config: TrainerConfig):
    ...
    _write_dashboard_run_config(trainer_config)
    sampler = _maybe_start_gpu_sampler(trainer_config)   # None when --gpu-stats off
    try:
        ...build trainer...
        trainer.fit()
    finally:
        if sampler is not None:
            sampler.stop()
            _flush_gpu_stats_artifacts(sampler, trainer_config)  # jsonl + summary json
            click.echo(sampler.summary_text())
```

When `--gpu-stats` is off (the default), `_maybe_start_gpu_sampler` returns
`None` and `run()` is byte-for-byte the current code path → backward compatible.

### CLI surface (new public options — flagged as needing maintainer confirmation per AGENTS.md)

Add to `train.py`'s `@click.option` block and register under the
**Observability** group in `TRAIN_OPTION_GROUPS` (alongside `metrics_log_dir`):

| Flag | Type | Default | Group | Validation |
|---|---|---|---|---|
| `--gpu-stats` | flag | `False` | Observability | — |
| `--gpu-stats-interval-s` | float | `5.0` | Observability | `> 0` |
| `--gpu-stats-history` | int | `1000` | Observability | `> 0` |

Add three fields with these defaults to `TrainerConfig` in
`areno/api/trainer_config.py` (base config — SFT/DPO also get observability):

```python
gpu_stats: bool = False
gpu_stats_interval_s: float = 5.0
gpu_stats_history: int = 1000
```

> AGENTS.md requires asking before altering CLI option surfaces or config
> dataclasses. The flag shape (off-by-default, safe defaults) was chosen with
> the issue author's constraints; final flag naming is open to maintainer
> review. These are **additive** (new fields with defaults), so existing
> constructors and the `_options()` test fixture stay valid.

### Artifacts (reuse the existing per-run-per-pid convention)

Written into `metrics_log_dir` (the same dir AReno already uses), named to
mirror `areno_run_config.{pid}.json`:

- `gpu_stats.{pid}.jsonl` — one JSON object per (tick, device), append-only.
- `gpu_stats_summary.{pid}.json` — one structured summary per run:

```json
{
  "pid": 12345,
  "interval_s": 5.0,
  "max_history": 1000,
  "n_samples": 87,
  "duration_s": 432.1,
  "devices": [0, 1],
  "reason": null,
  "per_device": {
    "0": {"peak_mem_used_mb": 71234, "mean_util_pct": 63, "max_temp_c": 71, "n_samples": 87},
    "1": {"peak_mem_used_mb": 70988, "mean_util_pct": 61, "max_temp_c": 70, "n_samples": 87}
  }
}
```

When NVIDIA tooling is unavailable: `reason="nvidia-smi not found"`,
`n_samples=0`, `per_device={}`.

### CLI summary (human-readable, printed after `fit()`)

A compact block in the same style as `_print_training_config_summary`, e.g.:

```
AReno GPU stats
  Devices            2
  Samples            87  (interval=5.0s, history_cap=1000, duration=432.1s)
  device 0  peak_mem 71234/81920 MB  mean_util 63%  max_temp 71C
  device 1  peak_mem 70988/81920 MB  mean_util 61%  max_temp 70C
  Wrote gpu_stats_summary.12345.json, gpu_stats.12345.jsonl
```

or, when unavailable:

```
AReno GPU stats
  nvidia-smi not found — GPU sampling disabled for this run.
```

## Testing plan (issue: CPU tests for core logic, malformed input, boundary,
disabled/default, deterministic output; integration with tiny fixtures;
GPU-only orchestration isolated behind fakes; assert fields+error strings, not
just exit status; default behavior unchanged)

`tests/test_gpu_stats_cpu.py` — core, no GPU:

1. `parse_nvidia_smi_csv` happy path (multi-device rows → `list[GPUSample]`).
2. `parse_nvidia_smi_csv` missing-column / malformed rows → missing fields are
   `None`, valid fields still parsed (covers "missing fields").
3. `GPUSampler` with injected `sample_fn` returning N devices → `history()`
   reflects them (covers "multi-GPU device mapping").
4. Over-push beyond `max_history` → exactly `max_history` retained, oldest
   dropped (covers "history bounds").
5. `stop()` then `history()` is a stable snapshot; calling `stop()` twice is
   safe (covers "sampler shutdown").
6. `sample_fn` raising → sampler yields `[]`, no exception escapes (graceful
   degrade in-process, parallel to the nvidia-smi-absent path).
7. `summary_text()` / `write_summary()` deterministic for a fixed sample set —
   assert exact fields and a stable string prefix (covers "deterministic
   output" + "CLI summaries" + "assert emitted fields ... not only exit
   status").
8. nvidia-smi-absent path: monkeypatch `shutil.which("nvidia-smi")` → `None` →
   `start()` is a no-op, `summary_text` reports the absent reason (covers
   "disabled/default behavior").

`tests/test_gpu_stats_cli_cpu.py` — integration across `train.py` (tiny fixture,
no real training): assert `run()` with `--gpu-stats` off produces no
`gpu_stats.*` artifacts and the help output lists the new flags under
Observability.

Existing `tests/test_train_cli_config_cpu.py` is augmented: add the three new
fields to `_options()` and a positive-validation case for the two new numeric
flags; confirm the default-off config still builds identically (guards
backward compatibility).

GPU-only validation that remains (documented, not in the CPU suite): a real
run with `--gpu-stats --gpu-stats-interval-s 1 --max-steps 2` on a box with
NVIDIA GPUs; assert `gpu_stats.{pid}.jsonl` has ≥1 non-empty line per visible
device. This is the single GPU box checked manually; the orchestration logic it
exercises is otherwise covered by the injected-fake CPU tests.

## Acceptance criteria checklist (from the issue)

- [x] multi-GPU device mapping — CPU test #3
- [x] missing fields — CPU test #2
- [x] sampler shutdown — CPU test #5
- [x] history bounds — CPU test #4
- [x] CLI summaries — CPU test #7
- [x] without blocking the training hot path — daemon thread, no engine/loss
      touch (design invariant)
- [x] uses existing AReno contracts; no external DB / mandatory sandbox —
      `metrics_log_dir` artifacts; stdlib only
- [x] default behavior backward compatible — flag off by default,
      `run()` unchanged when off
- [x] focused automated tests cover success / invalid input / one
      boundary/failure — CPU tests + CLI positive-validation
- [x] docs include a minimal runnable example + observable output — docs step
      (G4)

## Implementation gates (user approves each before the next)

1. **G1 — this design doc.** ← you are here
2. **G2 — `gpu_stats.py` core + `test_gpu_stats_cpu.py`, green locally.**
3. **G3 — `train.py` flag group + `run()` lifecycle +
      `test_train_cli_config_cpu.py` updates, green locally.**
4. **G4 — docs + claim-the-issue comment + PR description.**

## Open questions for the maintainer

- Final flag naming (`--gpu-stats` vs `--track-gpu-stats` vs a subcommand) —
  kept as `--gpu-stats` in this draft; happy to rename.
- Should the JSONL path also feed the dashboard's existing GPU panel, or stay a
  standalone artifact for now? (Default: standalone; dashboard integration is a
  separate issue per the non-goals.)
- Whether `serve` should get the same observability — out of scope here unless
  the maintainer wants it folded in.
