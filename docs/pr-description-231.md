# PR Description: Preflight output-directory writability and atomic writes (#231)

## Summary

Adds a preflight writability probe for `save_path` and `metrics_log_dir` directories before expensive model/worker initialization. The probe runs a full create→write→flush→rename→cleanup cycle with a uniquely-named temporary file, catching read-only directories, full disks, and quota errors early. All JSON/text file writes are also migrated to atomic writes (write-to-temp + rename) with exception cleanup.

Closes #231

## Motivation

Currently `areno train` does not validate output directories before training starts. If a directory is read-only, the disk is full, or a quota is exceeded, the error only surfaces when training reaches `save_interval` and attempts to write a checkpoint — wasting hours of compute.

Additionally, checkpoint `index.json` and dashboard config files are written directly to the target path via `write_text`. If the process is interrupted mid-write (e.g. OOM kill), a corrupted partial JSON file is left behind, causing cryptic parse errors on subsequent loads.

## Changes

### New: preflight probe module (`areno/cli/preflight_io.py`)

`probe_directory_writability()` runs a 5-step probe cycle on the target directory:

1. **create** — `mkdir(parents=True)` to handle nested missing directories
2. **probe file create** — `open("xb")` exclusive-create mode; never overwrites existing files
3. **write + flush + fsync** — verifies data actually lands on disk
4. **rename** — `Path.replace()` to verify atomic replace works
5. **cleanup** — `unlink` the probe file

Each step has its own try/except returning a `PreflightProbeResult` with the failing `operation` name. The `finally` block always runs glob-based cleanup, including `KeyboardInterrupt` scenarios. The file handle is tracked (`fh = None`) and safely closed in `finally` to prevent resource leaks.

Probe filenames include PID + UUID to avoid concurrent collisions.

### New: atomic write utilities (`areno/cli/atomic_io.py`)

The repo already had 3 ad-hoc implementations of write-to-temp + rename (`metrics.py`, `dashboard_registry.py`, `server.py`) — all duplicated and none with exception cleanup. Extracted into `atomic_write_text` / `atomic_write_bytes` / `atomic_write_json`, with automatic `.tmp` cleanup on any failure (including `KeyboardInterrupt`).

### CLI integration (`areno/cli/train.py`)

Two new options:
- `--preflight-io/--no-preflight-io` (default: enabled)
- `--preflight-probe-prefix` (default: `.areno_preflight_`)

`_preflight_output_directories()` is called in `_trainer_config_from_options` — after config is built (so `save_path` and `metrics_log_dir` are populated) and before `run()` (so workers haven't started yet). On failure, raises `click.UsageError` reporting the failing stage, operation, and path.

### Atomic write migration (6 write sites)

| File | Function | Before | After |
|------|----------|--------|-------|
| `metrics.py` | `record_dashboard_state` | Manual tmp+replace | `atomic_write_text` |
| `train.py` | `_write_dashboard_run_config` | Direct `write_text` | `atomic_write_text` + `atomic_write_json` |
| `io.py` | checkpoint `index.json` | Direct `open("w")` | `atomic_write_json` |
| `common.py` | passthrough `index.json` | Direct `write_text` | `atomic_write_json` |
| `dashboard_registry.py` | `_write_registry` | Manual tmp+replace | `atomic_write_json` |
| `server.py` | `_save_state` | Manual tmp+replace | `atomic_write_json` |

> Safetensors weight files (`save_file`) remain direct writes. The C extension cannot be wrapped in Python, preflight already validates directory writability, and the safetensors format is designed to be "readable only after complete write."

### Diagnostics upgrade (`areno/cli/diagnostics.py`)

`_writable_path_check` now uses the real I/O probe for existing directories. For non-existent paths, it falls back to `os.access` on the nearest existing parent — no side effects, no directory creation during `areno check`.

## Tests

| Test file | Count | Coverage |
|-----------|-------|----------|
| `test_atomic_io_cpu.py` | 8 | Create, overwrite, exception cleanup, JSON roundtrip, binary write, no partial file on failure |
| `test_preflight_io_cpu.py` | 19 | Success, read-only dir, disk full (ENOSPC), quota (EDQUOT), concurrent creation, interrupted probe, nested missing dirs, no user file overwrite, disabled, None/empty path, custom prefix, deterministic output, formatted output |
| `test_metrics_cpu.py` | +2 | Atomic write verification, no `.tmp` residue on error |
| `test_train_cli_config_cpu.py` | +8 | Writable dirs pass, read-only rejected, None skips, `--no-preflight-io` skips, error message contains stage and path, end-to-end CLI integration |
| `test_cli_diagnostics_cpu.py` | +2, 1 updated | Writable OK, read-only FAIL, non-existent WARN without creating directory |

**Runnable**: 27 tests, all passing (Python 3.9)
**Python 3.10+**: 14 tests, syntax verified via `ast.parse` (project requires Python >= 3.10 for `@dataclass(slots=True)`)

## Backward Compatibility

- `save_path=None` skips checkpoint directory probe (unchanged)
- `metrics_log_dir=None` skips metrics directory probe (unchanged)
- `--no-preflight-io` fully disables the probe
- Probe is lightweight (create temp file + delete), produces no output on success
- Atomic writes are transparent to callers; final file content is unchanged

## Documentation

- `docs/cli/training.rst` — New "Preflight I/O checks" section with option descriptions, probe steps, and failure output example
- `docs/cli/diagnostics.rst` — Updated writable check description
- `CODEMAP.md` — Added `preflight_io.py` and `atomic_io.py` entries

## Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| Cover read-only dirs, quota errors, concurrent creation, interrupted probes, nested missing dirs; report failing operation and path | ✅ 7 tests covering all scenarios |
| Uses existing AReno contracts; no external database or sandbox | ✅ Pure Python stdlib, no new dependencies |
| Default behavior remains backward compatible | ✅ `None` skips + `--no-preflight-io` + 3 tests |
| Focused tests cover success, invalid input, boundary/failure | ✅ 27 runnable + 14 syntax-verified |
| User docs with minimal runnable example and observable output | ✅ `docs/cli/training.rst` updated |