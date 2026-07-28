Skill SDK reference (design proposal)
=====================================

This page documents the proposed shared runtime SDK for repository skill
scripts, tracking :github:`issue #275 <inclusionAI/AReno/issues/275>`. The SDK
is a lightweight scaffolding library that extracts the repeated boilerplate
(argument parsing, JSON output, error handling) currently duplicated across the
24 scripts under ``.agents/skills/``. It is **not** a new framework and does not
replace any existing component — it only centralizes the scaffolding so scripts
keep just their business logic.

.. note::

   This is a design proposal. The SDK is not yet implemented. Sections below
   describe the target API, migration plan, and test strategy.

Motivation
----------

AReno ships 10 skills under ``.agents/skills/`` totalling 24 Python scripts
(~1572 lines). Reading all 24 reveals they are near-instantiations of one
template:

.. list-table::
   :header-rows: 1
   :widths: 25 55 20

   * - Dimension
     - Current pattern
     - Frequency
   * - Header
     - ``#!/usr/bin/env python3`` + ``from __future__ import annotations``
     - 24/24
   * - Args
     - Bare ``argparse.ArgumentParser()`` + ``parse_args()``
     - 24/24
   * - Output
     - ``print(json.dumps(result, indent=2, sort_keys=True))``
     - 22/24
   * - Result shape
     - ``{"ok": bool, "error"/"errors": ..., ...fields}``
     - 22/24
   * - Error envelope
     - ``except Exception as exc: result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}``
     - ~15/24
   * - Entry
     - ``def main() -> int`` + ``raise SystemExit(main())``
     - 23/24
   * - Exit code
     - ``return 0 if result["ok"] else 1``
     - 22/24

Problems:

1. ~300 lines of duplicated boilerplate across 24 scripts.
2. Inconsistent behavior — some use ``error``, some ``errors``;
   ``compare_ckpt_diff.py`` has no JSON output and no exit code.
3. Adding a cross-cutting capability (e.g. progress output, timestamped logs)
   requires editing 24 files.
4. Issue #276 (live progress) needs a unified mount point that does not exist.

Goals
~~~~~

Build a lightweight internal SDK that extracts the repeated scaffolding into a
shared library, so scripts keep only business logic. Per issue #275:

   Create a lightweight internal SDK for argument handling, stable result
   objects/exit codes, table and JSON rendering, progress events, and exception
   envelopes, then migrate representative scripts without breaking old flags.

Non-goals (from the issue):

* Replacing AReno's trainer, rollout engine, dashboard, or public SDK.
* Adding an external database, hosted control plane, or heavyweight dependency.
* Automatically changing user configuration, deleting artifacts, or terminating
  unrelated processes.
* Migrating all 24 scripts — only representative scripts.

Constraints
-----------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Constraint
     - Effect
   * - Reuse existing deps
     - ``rich>=13``, ``click>=8.1``, ``tqdm>=4.66`` are already AReno runtime
       deps; the SDK may use them without counting as a "new heavyweight
       dependency."
   * - Backward compatible
     - Old flags must not break; default behavior unchanged; unmigrated scripts
       stay as-is.
   * - Machine-clean stdout
     - In JSON mode, stdout contains only JSON — no human-readable text mixed in.
   * - Validate before init
     - Argument validation must complete before expensive model/worker
       initialization.
   * - Test contract unchanged
     - ``tests/test_agent_skills_cpu.py`` runs scripts via subprocess and
       asserts stdout JSON, the ``ok`` field, and exit code. The SDK must
       preserve this contract.
   * - CPU-testable
     - All tests run on CPU without GPU or network.

SDK layout
----------

The issue says: "Start with the named ``.agents/skills/`` package, shared
``.agents/scripts/`` code when justified, and script tests." The SDK is shared
across all skills, so it lives in the shared code area ``.agents/scripts/``:

.. code-block:: text

   .agents/scripts/
   ├── validate_skills.py              # existing: validates skill metadata
   └── areno_skill_sdk/                # new: shared SDK
       ├── __init__.py                 # public surface: skill_main, Result, ...
       ├── args.py                     # argument handling
       ├── result.py                   # result objects + exit codes
       ├── render.py                   # table + JSON rendering
       ├── progress.py                 # progress event protocol (for #276)
       └── errors.py                   # exception envelopes

Scripts import via a relative path insert — the lightest, zero-dependency way,
since skill scripts run standalone (``python .agents/skills/xxx/scripts/yyy.py``)
and are not part of the ``areno`` package:

.. code-block:: python

   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
   from areno_skill_sdk import skill_main, Result, SkillError

Modules
-------

Argument handling (args.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A thin wrapper over argparse providing a parser builder plus validation
helpers. **argparse is kept** (not switched to click) to avoid flag-compat risk.

.. code-block:: python

   import argparse

   def build_parser(description: str = "", **kwargs) -> argparse.ArgumentParser:
       """Build an ArgumentParser with unified behavior (disables abbrev)."""
       return argparse.ArgumentParser(
           description=description,
           allow_abbrev=False,  # avoid ambiguous prefix matching for flags
           **kwargs,
       )

   def validate_positive(args, *, exclude: tuple[str, ...] = ()) -> None:
       """Validate all int/float args are positive, else raise SkillError(stage='validate')."""
       for key, value in vars(args).items():
           if key in exclude:
               continue
           if isinstance(value, (int, float)) and value <= 0:
               raise SkillError(f"{key} must be positive", stage="validate")

Design notes:

* Keeping argparse preserves every existing flag (``--batch-size``, etc.) with
  zero migration risk.
* ``allow_abbrev=False`` disables argparse's default prefix-stripping (``--batch``
  matching ``--batch-size``) so flag behavior is predictable during migration.
* Validators raise ``SkillError(stage="validate")`` to satisfy the issue's
  "validate before init + identify affected stage on failure" requirement.

Result objects + exit codes (result.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unifies the ``{"ok": bool, ...}`` shape and stabilizes exit-code semantics.

.. code-block:: python

   from dataclasses import dataclass, field
   from typing import Any

   @dataclass
   class Result:
       ok: bool
       data: dict[str, Any] = field(default_factory=dict)
       errors: list[str] = field(default_factory=list)
       stage: str | None = None       # identifies stage on failure

       def to_dict(self) -> dict[str, Any]:
           out: dict[str, Any] = {"ok": self.ok}
           if self.stage:
               out["stage"] = self.stage
           if self.errors:
               out["errors"] = self.errors
           out.update(self.data)
           return out

   def exit_code(result: dict[str, Any]) -> int:
       """ok=True -> 0, otherwise -> 1. Preserves existing 0-if-ok semantics."""
       return 0 if result.get("ok") else 1

``to_dict()`` outputs the same ``{"ok":..., "errors":..., ...}`` shape used
today, so existing tests asserting stdout JSON are unaffected.

Table + JSON rendering (render.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dual-mode output: JSON mode writes pure JSON to stdout (machine-clean); human
mode renders rich tables.

.. code-block:: python

   import json
   import sys

   def emit(result: dict, *, json_mode: bool = True, indent: int = 2) -> None:
       """Emit result. json_mode=True writes only JSON to stdout (machine-clean);
       json_mode=False writes human-readable rich text to stderr."""
       if json_mode:
           sys.stdout.write(json.dumps(result, indent=indent, sort_keys=True) + "\n")
       else:
           _emit_human(result)

   def _emit_human(result: dict) -> None:
       from rich.console import Console
       console = Console(file=sys.stderr)  # stderr so stdout stays clean
       if result.get("ok"):
           console.print("[green]OK[/green]")
       else:
           console.print(f"[red]FAIL[/red]: {result.get('error') or result.get('errors')}")
       for key, value in result.items():
           if key in ("ok", "error", "errors", "stage"):
               continue
           if isinstance(value, list) and value and isinstance(value[0], dict):
               _render_table(value, console, title=key)

Design notes:

* ``json_mode`` defaults to ``True`` to preserve the existing "stdout is JSON"
  contract.
* Human-readable output goes to **stderr** so stdout stays machine-clean even
  when human mode is on — satisfying "stdout machine-clean in JSON mode."
* Uses ``rich`` (already a dependency) for tables — no wheel reinvention.

Progress events (progress.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Defines the stage/progress event protocol to pave the way for #276. **This
issue only defines the protocol plus a JSONL sink** — TTY in-place refresh,
cancellation, and last-completed-stage reporting belong to #276.

.. code-block:: python

   from dataclasses import dataclass
   from typing import TextIO

   @dataclass
   class ProgressEvent:
       stage: str            # e.g. "load_dataset"
       fraction: float       # 0.0 ~ 1.0
       message: str = ""

   class ProgressSink:
       """Base class. #275 ships only the JSONL sink (deterministic, testable).
       TTY in-place refresh / non-TTY line output are added by #276."""
       def emit(self, event: ProgressEvent) -> None: ...
       def close(self) -> None: ...

   class JsonLinesSink(ProgressSink):
       def __init__(self, stream: TextIO): self.stream = stream
       def emit(self, event: ProgressEvent) -> None:
           import json
           self.stream.write(json.dumps({
               "type": "progress", "stage": event.stage,
               "fraction": event.fraction, "message": event.message,
           }) + "\n")
           self.stream.flush()
       def close(self) -> None: pass

Exception envelopes (errors.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unifies try/except into structured errors that identify the affected stage
without exposing raw training samples.

.. code-block:: python

   class SkillError(Exception):
       """Business error with stage info."""
       def __init__(self, message: str, *, stage: str = "execute"):
           super().__init__(message)
           self.stage = stage

   def envelope(exc: Exception, *, stage: str = "execute") -> dict:
       """Wrap any exception into {"ok": False, "error": ..., "stage": ...}."""
       return {
           "ok": False,
           "error": f"{type(exc).__name__}: {exc}",
           "stage": stage,
       }

``envelope()`` matches the existing ``{"ok": False, "error": f"{type(exc).__name__}: {exc}"}``
shape, only adding a ``stage`` field. Scripts raise ``SkillError`` themselves,
so they control the message and never leak raw data into errors.

Unified entry decorator (__init__.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single decorator takes over "exception envelope + JSON output + exit code";
the script's ``main`` only returns a ``Result``.

.. code-block:: python

   import functools
   from .args import build_parser, validate_positive
   from .result import Result, exit_code
   from .render import emit
   from .errors import SkillError, envelope
   from .progress import ProgressEvent, JsonLinesSink

   def skill_main(func):
       """Decorate a script main(). Takes over: exception envelope + JSON
       output + exit code. The wrapped function returns Result or dict."""
       @functools.wraps(func)
       def wrapper() -> int:
           try:
               result = func()
               if isinstance(result, Result):
                   result = result.to_dict()
               emit(result)
               return exit_code(result)
           except SkillError as exc:
               emit({"ok": False, "error": str(exc), "stage": exc.stage})
               return 1
           except Exception as exc:
               emit(envelope(exc))
               return 1
       return wrapper

   __all__ = [
       "skill_main", "build_parser", "validate_positive",
       "Result", "exit_code", "emit",
       "SkillError", "envelope",
       "ProgressEvent", "JsonLinesSink",
   ]

Migration plan
--------------

The issue asks to "migrate representative scripts," not all of them. Three
scripts are chosen to cover the three typical scenarios:

.. list-table::
   :header-rows: 1
   :widths: 12 30 20 38

   * - Batch
     - Script
     - Type
     - Validates
   * - P1
     - ``check_capacity.py`` (43 lines)
     - Pure compute
     - args + result + errors modules; simplest, runs first
   * - P1
     - ``inspect_algorithms.py`` (36 lines)
     - Compute with exceptions
     - ``@skill_main`` exception envelope
   * - P2
     - ``monitor_gpu.py`` (119 lines)
     - Streaming / long-running
     - progress protocol + JSON Lines output
   * - P3
     - ``compare_ckpt_diff.py`` (179 lines)
     - Human-readable (no JSON)
     - SDK compatibility with non-JSON output

The remaining 20 scripts are not migrated and stay as-is for backward
compatibility.

Migration example: check_capacity.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Before** (43 lines):

.. code-block:: python

   def main() -> int:
       parser = argparse.ArgumentParser()
       parser.add_argument("--batch-size", type=int, required=True)
       parser.add_argument("--n-samples", type=int, required=True)
       # ... 6 args ...
       args = parser.parse_args()
       values = vars(args)
       errors = [f"{key} must be positive" for key, value in values.items()
                 if key != "memory_fraction" and value <= 0]
       if not 0 < args.memory_fraction <= 0.9:
           errors.append("memory_fraction must be in (0, 0.9]")
       if args.world_size % args.tp_size:
           errors.append("world_size must be divisible by tp_size")
       demand = args.batch_size * args.n_samples
       waves = math.ceil(demand / args.max_running_prompts) if args.max_running_prompts > 0 else None
       result = {
           "ok": not errors,
           "errors": errors,
           "rollout_demand": demand,
           "minimum_admission_waves": waves,
           "data_parallel_size": args.world_size // args.tp_size if not args.world_size % args.tp_size else None,
           "settings": values,
       }
       print(json.dumps(result, indent=2, sort_keys=True))
       return 0 if result["ok"] else 1

   if __name__ == "__main__":
       raise SystemExit(main())

**After** (~30 lines):

.. code-block:: python

   import sys
   import pathlib
   import math
   sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
   from areno_skill_sdk import skill_main, build_parser, Result, SkillError, validate_positive

   @skill_main
   def main():
       parser = build_parser("Check AReno capacity relationships without allocating a model.")
       parser.add_argument("--batch-size", type=int, required=True)
       parser.add_argument("--n-samples", type=int, required=True)
       # ... 6 args ...
       args = parser.parse_args()

       validate_positive(args, exclude=("memory_fraction",))
       if not 0 < args.memory_fraction <= 0.9:
           raise SkillError("memory_fraction must be in (0, 0.9]", stage="validate")
       if args.world_size % args.tp_size:
           raise SkillError("world_size must be divisible by tp_size", stage="validate")

       demand = args.batch_size * args.n_samples
       waves = math.ceil(demand / args.max_running_prompts) if args.max_running_prompts > 0 else None
       return Result(
           ok=True,
           data={
               "rollout_demand": demand,
               "minimum_admission_waves": waves,
               "data_parallel_size": args.world_size // args.tp_size,
               "settings": vars(args),
           },
       )

   if __name__ == "__main__":
       raise SystemExit(main())

Behavior is unchanged — same stdout JSON, same exit codes:

.. list-table::
   :header-rows: 1
   :widths: 25 40 25 10

   * - Scenario
     - Before stdout
     - After stdout
     - Exit
   * - Success
     - ``{"ok": true, "rollout_demand": ..., ...}``
     - ``{"ok": true, "rollout_demand": ..., ...}``
     - 0→0
   * - Invalid arg
     - ``{"ok": false, "errors": ["..."]}``
     - ``{"ok": false, "errors": ["..."]}`` (option A, see below)
     - 1→1

.. warning::

   **Compatibility detail — ``error`` vs ``errors``:** before migration this
   script used ``errors`` (a list). A naive migration that raises ``SkillError``
   would produce ``error`` (a string), changing the field name. Two options:

   * **Option A (recommended):** scripts that aggregate multiple errors keep
     using ``errors`` — collect errors in ``main`` and return
     ``Result(ok=False, errors=[...])`` without raising ``SkillError``. Only
     "single-error abort" scripts raise ``SkillError`` (producing ``error``).
     This preserves both field names exactly — zero compat risk.
   * **Option B:** make ``SkillError`` carry multiple errors and have
     ``envelope`` always emit ``errors``.

   Option A keeps ``errors`` (multi) and ``error`` (single) semantics distinct,
   matching today's usage, so migrated scripts have unchanged field names.

Migration invariants
~~~~~~~~~~~~~~~~~~~~

Every migration must preserve:

1. ``--help`` behavior (argparse kept).
2. On success: stdout is JSON with ``ok: true``, exit code 0.
3. On failure: stdout is JSON with ``ok: false``, exit code 1.
4. JSON top-level field names unchanged (``error``/``errors``/business fields).
5. Existing test ``test_agent_skills_cpu.py`` still passes.

Test plan
---------

A new ``tests/test_skill_sdk_cpu.py`` follows the existing ``*_cpu.py`` naming
convention (CPU-runnable, no GPU/network).

SDK unit tests
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Test
     - Covers
   * - ``test_result_success_dict``
     - Success path: ``Result(ok=True, data={...}).to_dict()`` == ``{"ok":True, ...}``
   * - ``test_result_error_envelope``
     - Invalid input: ``SkillError`` wrapped by ``@skill_main`` into ``{"ok":False,"error":...,"stage":...}``, exit 1
   * - ``test_validate_positive_raises``
     - Boundary: raises ``SkillError(stage="validate")`` for 0/negative
   * - ``test_emit_json_stdout_clean``
     - stdout machine-clean: only JSON, no extra text
   * - ``test_emit_human_to_stderr``
     - Human mode does not pollute stdout; tables on stderr
   * - ``test_exit_code_semantics``
     - ``exit_code({"ok":True})`` == 0, ``exit_code({"ok":False})`` == 1
   * - ``test_progress_jsonl_deterministic``
     - ``JsonLinesSink`` emits one JSON per line, fixed fields

Migrated-script regression tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reuses the subprocess pattern from ``test_agent_skills_cpu.py``:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Test
     - Asserts
   * - ``test_check_capacity_migrated_success``
     - subprocess runs migrated script; ``ok: true`` + exit 0
   * - ``test_check_capacity_migrated_invalid``
     - invalid args (negative); ``ok: false`` + exit 1 + ``errors`` present
   * - ``test_check_capacity_backward_compat``
     - old flags still work, behavior unchanged

Integration test
~~~~~~~~~~~~~~~~

``test_sdk_end_to_end`` — a tiny local fixture: a minimal script exercises the
full SDK flow (args → validate → compute → emit), asserting stdout JSON + exit
code.

Mapping to the issue's testing requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Issue requirement
     - Test
   * - Core logic + success
     - ``test_result_success_dict``, migrated-script success tests
   * - Malformed input
     - ``test_result_error_envelope``, ``test_validate_positive_raises``, invalid tests
   * - Boundary values
     - ``test_validate_positive_raises`` (zero boundary)
   * - Disabled/default behavior
     - ``test_check_capacity_backward_compat``
   * - Deterministic output
     - ``test_progress_jsonl_deterministic``, ``test_emit_json_stdout_clean``
   * - Integration across modules
     - ``test_sdk_end_to_end``
   * - Assert fields/messages, not just exit status
     - all tests assert JSON field contents
   * - Existing behavior unchanged
     - existing ``test_agent_skills_cpu.py`` unmodified, still passes

Documentation
-------------

Per the issue: "Document the user-facing option or command, input contract,
defaults, output fields, limitations, and one copyable example."

Deliverables:

1. ``.agents/scripts/areno_skill_sdk/README.md`` — SDK usage: per-module API,
   input contract, defaults, output fields (``ok``/``error``/``errors``/``stage``
   /business), limitations (human mode on stderr), one copyable migration
   example.
2. Update ``.agents/skills/areno-tune-capacity/SKILL.md`` if
   ``check_capacity`` is migrated — note the script now uses the SDK, but the
   invocation is unchanged.
3. This page (``docs/sdk/skill-sdk.rst``).

Copyable example
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Invocation is unchanged after migration:
   python .agents/skills/areno-tune-capacity/scripts/check_capacity.py \
     --batch-size 4 --n-samples 16 --max-running-prompts 8 \
     --mini-bs 2 --world-size 4 --tp-size 2

Output (unchanged):

.. code-block:: json

   {
     "ok": true,
     "rollout_demand": 64,
     "minimum_admission_waves": 8,
     "data_parallel_size": 2,
     "settings": {"batch_size": 4, "n_samples": 16, "...": "..."}
   }

Effort and risks
----------------

.. list-table::
   :header-rows: 1
   :widths: 50 15 15 20

   * - Work item
     - Lines
     - Difficulty
     - Notes
   * - SDK six modules
     - ~300
     - Medium
     - Design-heavy
   * - Migrate 3-4 scripts
     - net -50
     - Low
     - Boilerplate removed
   * - CPU tests
     - ~200
     - Low
     -
   * - Docs (README + SKILL.md)
     - ~100
     - Low
     -
   * - **Total**
     - **~550 net**
     - **Medium-low**
     -

Risks and mitigations
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 35 10 10 45

   * - Risk
     - Prob
     - Impact
     - Mitigation
   * - SDK API design poor, migration costly
     - Med
     - Med
     - Migrate P1's two simplest scripts first to validate API
   * - Migration breaks test contract
     - Low
     - High
     - Invariants in §Migration invariants; run ``test_agent_skills_cpu.py`` each batch
   * - ``error`` vs ``errors`` field incompat
     - Med
     - Med
     - Option A: multi-error keeps ``errors``, single-error uses ``error``
   * - Progress module does #276's work
     - Low
     - Low
     - #275 ships only protocol + JSONL sink; TTY/cancellation deferred to #276
   * - ``sys.path`` insert inelegant
     - Low
     - Low
     - Zero-dep lightest option; revisitable if .agents becomes a package

Out of scope
~~~~~~~~~~~~

* No full migration of all 24 scripts (only 3-4 representative).
* No argparse→click switch (flag-compat risk).
* No TTY in-place refresh or cancellation (that is #276).
* No changes to AReno's public SDK (``areno/api/``) — this SDK serves only
  ``.agents/skills/`` scripts.
* No new dependencies (rich/click/tqdm already deps; stdlib suffices).

Acceptance criteria mapping
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Issue criterion
     - How this design meets it
   * - Human output clear, stdout machine-clean in JSON mode; compat tests,
     documented extension patterns, no new heavy runtime dep
     - ``emit(json_mode=True)`` pure JSON on stdout; human mode on stderr;
       reuses rich (existing dep); ``test_emit_json_stdout_clean``
   * - Uses existing AReno contracts; no external DB or mandatory sandbox
     - Reuses ``{"ok":...}`` contract; pure Python + existing deps; no DB/sandbox
   * - Default behavior backward compatible
     - ``json_mode`` defaults True; unmigrated scripts untouched; migrated
       scripts keep field names (Option A)
   * - Automated tests cover success, invalid input, boundary/failure
     - §Test plan covers all
   * - User docs include minimal runnable example + observable output
     - §Documentation + copyable example

Implementation order
--------------------

Each step is independently verifiable and can be a standalone commit:

1. **SDK skeleton** — create ``.agents/scripts/areno_skill_sdk/`` with
   args + result + errors + ``__init__`` (skip render/progress complexity).
   *verify:* ``python -c "from areno_skill_sdk import skill_main, Result"`` imports.
2. **P1 migration** — migrate ``check_capacity.py`` + ``inspect_algorithms.py``.
   *verify:* ``pytest tests/test_agent_skills_cpu.py`` passes; stdout JSON
   contract unchanged.
3. **render + progress modules** — add table rendering + progress protocol.
   *verify:* ``test_emit_json_stdout_clean``, ``test_progress_jsonl_deterministic``.
4. **P2 migration** — migrate ``monitor_gpu.py`` (validates progress).
   *verify:* streaming output + progress events work.
5. **P3 migration** — migrate ``compare_ckpt_diff.py`` (validates human-readable).
   *verify:* human-readable output preserved.
6. **Tests + docs** — complete ``test_skill_sdk_cpu.py`` + README.
   *verify:* ``pytest tests/ -k cpu`` all pass; docs contain copyable example.
