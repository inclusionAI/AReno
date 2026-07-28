# areno_skill_sdk — Shared runtime SDK for skill scripts

A lightweight scaffolding library that extracts the repeated boilerplate
(argument parsing, JSON output, error handling) duplicated across the 24
scripts under `.agents/skills/`. It is **not** a new framework and does not
replace any existing component — it only centralizes the scaffolding so scripts
keep just their business logic.

See `docs/sdk/skill-sdk.rst` for the full design proposal (issue #275).

## Quick start

A skill script imports the SDK via a `sys.path` insert, since scripts run
standalone and are not part of the `areno` package:

```python
#!/usr/bin/env python3
"""My skill script."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
from areno_skill_sdk import build_parser, Result, SkillError, skill_main


@skill_main
def main() -> Result:
    parser = build_parser("Do something useful.")
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    if args.count <= 0:
        raise SkillError("count must be positive", stage="validate")

    return Result(ok=True, data={"count": args.count})


if __name__ == "__main__":
    raise SystemExit(main())
```

`@skill_main` takes over exception envelope + JSON output + exit code. The
decorated function returns a `Result` (or a `dict`); the decorator emits it
and returns the exit code.

## Modules

### `skill_main(func)` — unified entry decorator

Takes over exception envelope + JSON output + exit code. The wrapped function
returns a `Result` or `dict`.

| Situation | Emitted JSON | Exit code |
|-----------|--------------|:---------:|
| Returns `Result(ok=True, ...)` | `{"ok": true, ...}` | 0 |
| Returns `Result(ok=False, errors=[...])` | `{"ok": false, "errors": [...]}` | 1 |
| Raises `SkillError("msg", stage="x")` | `{"ok": false, "error": "msg", "stage": "x"}` | 1 |
| Raises any other `Exception` | `{"ok": false, "error": "Type: msg", "stage": "execute"}` | 1 |

### `build_parser(description="", **kwargs)` — argument handling

A thin wrapper over `argparse.ArgumentParser` with `allow_abbrev=False` so flag
behavior is predictable (no prefix-stripping). Pass any extra `ArgumentParser`
kwargs through. Keep argparse (do not switch to click) to preserve every
existing flag with zero migration risk.

### `validate_positive(args, *, exclude=())` — argument validation

Raises `SkillError(stage="validate")` for any non-positive numeric arg, before
expensive model/worker initialization. Keys in `exclude` (e.g.
`memory_fraction`, which has its own range check) are skipped.

### `Result(ok, data={}, errors=<unset>, stage=None)` — result objects

A dataclass whose `to_dict()` produces the `{"ok": ..., ...}` contract used by
all scripts:

- `ok`: success flag.
- `data`: business fields, spread into the top-level dict.
- `errors`: multi-error list. Emitted **only when the caller passes it
  explicitly** (including an empty list), so scripts like `check_capacity`
  keep their exact pre-migration shape (`errors: []` on success), while scripts
  that never report multi-errors do not gain a spurious field.
- `stage`: failed-stage label; emitted only when set.

`exit_code(result_dict)` maps `{"ok": true}` -> `0`, otherwise `1`, preserving
the legacy `0 if result["ok"] else 1` semantics.

### `emit(result, *, json_mode=True, indent=2, stream=None)` — rendering

- `json_mode=True` (default): writes pure JSON to stdout (machine-clean).
- `json_mode=False`: writes human-readable rich tables to **stderr** so stdout
  stays machine-clean. Falls back to plain text when `rich` is absent, so the
  SDK imports in minimal CPU environments.
- `stream`: override the JSON output destination (for JSON-Lines-style scripts
  and tests).

### `SkillError(message, *, stage="execute")` / `envelope(exc, *, stage)` — exceptions

`SkillError` carries the failed `stage`. `envelope(exc)` wraps any exception
into `{"ok": False, "error": "Type: msg", "stage": ...}`, matching the legacy
`f"{type(exc).__name__}: {exc}"` convention and adding a `stage` field.

### `ProgressEvent` / `ProgressSink` / `JsonLinesSink` — progress protocol

`ProgressEvent(stage, fraction, message)` is the protocol. `JsonLinesSink`
writes one JSON object per line (deterministic, testable). TTY in-place refresh,
non-TTY line output, cancellation, and last-completed-stage reporting belong to
issue #276, which builds on this protocol.

## Input contract

| Input | Rule |
|-------|------|
| Args | argparse; flags are not abbreviations (`allow_abbrev=False`). |
| Validation | `validate_positive` raises `SkillError(stage="validate")` before expensive init. |
| Defaults | Every new option has a safe default preserving current behavior. |

## Output fields

| Field | When present | Type |
|-------|--------------|------|
| `ok` | always | bool |
| `error` | single-error abort (via `SkillError`) | str |
| `errors` | multi-error aggregation (via `Result(errors=[...])`) | list[str] |
| `stage` | on failure | str |
| business fields | from `Result.data` | any |

## Limitations

- Human-readable mode (`json_mode=False`) writes to **stderr**, not stdout, to
  keep stdout machine-clean. Consumers reading human output must read stderr.
- `rich` is optional. When absent, human mode degrades to plain aligned text.
- The SDK serves only `.agents/skills/` scripts; it does not change AReno's
  public SDK (`areno/api/`).

## Copyable example

```bash
# Invocation is unchanged after migration:
python .agents/skills/areno-tune-capacity/scripts/check_capacity.py \
  --batch-size 4 --n-samples 16 --max-running-prompts 8 \
  --mini-bs 2 --world-size 4 --tp-size 2
```

Output (unchanged):

```json
{
  "data_parallel_size": 2,
  "errors": [],
  "minimum_admission_waves": 8,
  "ok": true,
  "rollout_demand": 64,
  "settings": {"batch_size": 4, "max_running_prompts": 8, "...": "..."}
}
```

## Extension patterns

- **New script, JSON output**: use `@skill_main` + `Result`. Return `Result`
  on success; raise `SkillError(stage=...)` to abort with a labeled stage.
- **New script, multi-error validation**: collect errors in a list and return
  `Result(ok=not errors, errors=errors, data={...})` — keeps the `errors` field
  name for compatibility with existing consumers.
- **Streaming script**: do not use `@skill_main` (it emits once at the end). Use
  `build_parser` for args and keep the streaming loop; call `emit(...,
  stream=...)` per record if you want JSON-Lines output.
- **Human-readable-only script**: use `build_parser` for args only; keep plain
  `print()` output. The SDK never forces JSON on scripts that intentionally
  emit human-readable diffs.
- **Long-running script with progress**: construct a `JsonLinesSink` and call
  `sink.emit(ProgressEvent(...))` between stages. TTY refresh arrives in #276.