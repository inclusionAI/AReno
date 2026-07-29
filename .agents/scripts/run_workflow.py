#!/usr/bin/env python3
"""Composable local skill-workflow executor.

Executes a YAML-defined workflow of AReno skill scripts with typed inputs,
step dependencies, structured output passing, and restart-from-failed-step.

Uses only the Python standard library. The YAML subset supported covers
mappings, sequences, strings, integers, floats, booleans, and null -- enough
for workflow definitions without pulling in a YAML dependency.

Script contract: each step's script must be a Python script that accepts
argparse-style flags and prints a single JSON object to stdout. The executor
parses that JSON and makes its keys addressable as ``${steps.<id>.<key>}``
in subsequent steps' ``inputs``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Minimal YAML parser (standard library only)
# ---------------------------------------------------------------------------
#
# Supports the subset needed for workflow files:
#   - top-level mapping
#   - nested mappings and sequences
#   - inline lists ``[a, b]`` and inline scalars
#   - string / int / float / bool / null values
#   - ``#`` comments
#   - simple quoted strings (single or double)
#
# This is NOT a full YAML 1.2 parser; it deliberately rejects constructs it
# does not understand so that invalid workflows fail loudly.


class YamlParseError(ValueError):
    """Raised when the YAML subset parser encounters unsupported syntax."""


_BOOL_TRUE = {"true", "yes", "on"}
_BOOL_FALSE = {"false", "no", "off"}
_NULL = {"null", "~", ""}
_INTERP = re.compile(r"\$\{steps\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_.]+)\}")


def _strip_inline_comment(text: str) -> str:
    """Remove a trailing ``#`` comment that is not inside quotes."""
    in_single = in_double = False
    for i, ch in enumerate(text):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return text[:i]
    return text


def _parse_scalar(raw: str) -> Any:
    """Convert a raw YAML scalar string to its Python value."""
    val = raw.strip()
    if not val:
        return None
    # Quoted strings
    if (val[0] == val[-1]) and val[0] in ('"', "'"):
        return val[1:-1]
    # Inline list
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    # Inline mapping (not supported -- keep simple)
    if val.startswith("{"):
        raise YamlParseError(f"inline mappings are not supported: {raw!r}")
    # Interpolation placeholders are kept as strings (resolved later)
    if "${" in val:
        return val
    low = val.lower()
    if low in _BOOL_TRUE:
        return True
    if low in _BOOL_FALSE:
        return False
    if low in _NULL:
        return None
    # Int
    try:
        return int(val)
    except ValueError:
        pass
    # Float
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _indent_level(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_yaml(text: str) -> dict[str, Any]:
    """Parse a YAML subset into a Python dict.

    Raises :class:`YamlParseError` on unsupported syntax.
    """
    # Pre-process: strip comments, drop blank lines, normalise tabs to spaces.
    raw_lines: list[tuple[int, str]] = []
    for line in text.splitlines():
        stripped = _strip_inline_comment(line).rstrip()
        if not stripped.strip():
            continue
        if "\t" in stripped:
            raise YamlParseError("tabs are not allowed for indentation")
        raw_lines.append((_indent_level(stripped), stripped.lstrip(" ")))

    if not raw_lines:
        return {}

    result, consumed = _parse_block(raw_lines, 0, min_indent=0)
    # Any leftover lines mean inconsistent indentation / unsupported nesting.
    if consumed < len(raw_lines):
        raise YamlParseError(
            f"unexpected trailing content at line index {consumed}: {raw_lines[consumed]!r}"
        )
    if not isinstance(result, dict):
        raise YamlParseError("top-level YAML must be a mapping")
    return result


def _parse_block(
    lines: list[tuple[int, str]], start: int, min_indent: int
) -> tuple[Any, int]:
    """Parse a block (mapping or sequence) starting at *start*.

    Returns ``(value, next_index)`` where *next_index* is the index of the
    first line that does not belong to this block.
    """
    if start >= len(lines):
        return None, start

    indent, content = lines[start]
    if indent < min_indent:
        return None, start

    # Sequence?
    if content.startswith("- "):
        return _parse_sequence(lines, start, indent)
    if content == "-":
        # Empty list item -- not supported for our schema.
        raise YamlParseError("empty sequence items are not supported")

    # Otherwise treat as mapping.
    return _parse_mapping(lines, start, indent)


def _parse_mapping(
    lines: list[tuple[int, str]], start: int, indent: int
) -> tuple[dict[str, Any], int]:
    result: dict[str, Any] = {}
    i = start
    while i < len(lines):
        cur_indent, cur_content = lines[i]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise YamlParseError(
                f"unexpected over-indentation at line {i}: {cur_content!r}"
            )
        if cur_content.startswith("- "):
            raise YamlParseError("unexpected sequence item in mapping context")
        # Split "key: value"
        key, sep, value = cur_content.partition(":")
        if not sep:
            raise YamlParseError(f"missing ':' in mapping line: {cur_content!r}")
        key = key.strip()
        if not key:
            raise YamlParseError(f"empty key in mapping line: {cur_content!r}")
        value = value.strip()
        if value:
            result[key] = _parse_scalar(value)
            i += 1
        else:
            # Value is on subsequent indented lines.
            i += 1
            if i >= len(lines) or lines[i][0] <= indent:
                # Empty value -> null
                result[key] = None
                continue
            child_indent = lines[i][0]
            if child_indent <= indent:
                result[key] = None
                continue
            child_value, i = _parse_block(lines, i, child_indent)
            result[key] = child_value
    return result, i


def _parse_sequence(
    lines: list[tuple[int, str]], start: int, indent: int
) -> tuple[list[Any], int]:
    result: list[Any] = []
    i = start
    while i < len(lines):
        cur_indent, cur_content = lines[i]
        if cur_indent < indent:
            break
        if cur_indent > indent:
            raise YamlParseError(
                f"unexpected over-indentation at line {i}: {cur_content!r}"
            )
        if not cur_content.startswith("- "):
            break
        item_content = cur_content[2:].strip()
        if ":" in item_content and not item_content.startswith(("'", '"')):
            # This is a mapping item like ``- id: foo``.
            # Re-inject as a single-line mapping at indent+2 and parse.
            synthetic = (indent + 2, item_content)
            # Collect all continuation lines for this list item.
            j = i + 1
            item_lines = [synthetic]
            while j < len(lines) and lines[j][0] > indent:
                item_lines.append(lines[j])
                j += 1
            parsed, _ = _parse_mapping(item_lines, 0, indent + 2)
            result.append(parsed)
            i = j
        else:
            result.append(_parse_scalar(item_content))
            i += 1
    return result, i


# ---------------------------------------------------------------------------
# Workflow model
# ---------------------------------------------------------------------------


class WorkflowError(ValueError):
    """Raised for invalid workflow definitions."""


class StepResult:
    """Result of a single step execution."""

    # Status constants used for reliable classification instead of
    # string-matching on error messages.
    STATUS_SUCCESS = "success"
    STATUS_SKIPPED_RESTART = "skipped_restart"
    STATUS_SKIPPED_FAILURE = "skipped_failure"
    STATUS_FAILED = "failed"

    __slots__ = ("step_id", "ok", "output", "error", "returncode", "status")

    def __init__(
        self,
        step_id: str,
        ok: bool,
        output: dict[str, Any] | None,
        error: str,
        returncode: int,
        status: str = STATUS_SUCCESS,
    ):
        self.step_id = step_id
        self.ok = ok
        self.output = output
        self.error = error
        self.returncode = returncode
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "returncode": self.returncode,
            "status": self.status,
        }


def _validate_workflow(wf: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the workflow dict and return the normalised steps list."""
    if not isinstance(wf, dict):
        raise WorkflowError("workflow root must be a mapping")
    for required in ("name", "steps"):
        if required not in wf:
            raise WorkflowError(f"workflow is missing required key: {required!r}")
    steps = wf["steps"]
    if not isinstance(steps, list) or not steps:
        raise WorkflowError("'steps' must be a non-empty sequence")
    seen_ids: set[str] = set()
    normalised: list[dict[str, Any]] = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise WorkflowError(f"step at index {idx} is not a mapping")
        step_id = step.get("id")
        if not step_id or not isinstance(step_id, str):
            raise WorkflowError(f"step at index {idx} lacks a string 'id'")
        if step_id in seen_ids:
            raise WorkflowError(f"duplicate step id: {step_id!r}")
        seen_ids.add(step_id)
        script = step.get("script")
        if not script or not isinstance(script, str):
            raise WorkflowError(f"step {step_id!r} lacks a string 'script' path")
        depends_on = step.get("depends_on", [])
        if depends_on is None:
            depends_on = []
        if not isinstance(depends_on, list):
            raise WorkflowError(f"step {step_id!r} 'depends_on' must be a list")
        for dep in depends_on:
            if not isinstance(dep, str):
                raise WorkflowError(f"step {step_id!r} has a non-string dependency")
        inputs = step.get("inputs", {})
        if inputs is None:
            inputs = {}
        if not isinstance(inputs, dict):
            raise WorkflowError(f"step {step_id!r} 'inputs' must be a mapping")
        normalised.append(
            {
                "id": step_id,
                "script": script,
                "inputs": inputs,
                "depends_on": list(depends_on),
            }
        )
    # Validate dependency references exist.
    for step in normalised:
        for dep in step["depends_on"]:
            if dep not in seen_ids:
                raise WorkflowError(
                    f"step {step['id']!r} depends on unknown step {dep!r}"
                )
    return normalised


def _topo_sort(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Topologically sort steps; raise on cycles."""
    by_id = {s["id"]: s for s in steps}
    visited: dict[str, int] = {}  # 0 = visiting, 1 = done
    order: list[dict[str, Any]] = []

    def visit(sid: str, path: list[str]) -> None:
        state = visited.get(sid)
        if state == 1:
            return
        if state == 0:
            cycle = " -> ".join(path + [sid])
            raise WorkflowError(f"dependency cycle detected: {cycle}")
        visited[sid] = 0
        for dep in by_id[sid]["depends_on"]:
            visit(dep, path + [sid])
        visited[sid] = 1
        order.append(by_id[sid])

    for s in steps:
        visit(s["id"], [])
    return order


def _collect_deps(
    steps: list[dict[str, Any]], targets: set[str], collected: set[str]
) -> None:
    """Recursively collect all transitive dependencies of *targets* into *collected*."""
    by_id = {s["id"]: s for s in steps}
    queue = list(targets)
    while queue:
        sid = queue.pop()
        if sid in collected:
            continue
        collected.add(sid)
        for dep in by_id[sid]["depends_on"]:
            if dep not in collected:
                queue.append(dep)


def _resolve_value(
    value: Any, outputs: dict[str, dict[str, Any]], tolerant: bool = False
) -> Any:
    """Resolve ``${steps.<id>.<key>}`` placeholders in a value.

    If the entire string is a single placeholder, the referenced Python object
    is returned directly (preserving type). If the placeholder is embedded in a
    larger string, it is stringified and substituted.

    When *tolerant* is true (e.g. dry-run), unresolved placeholders are kept
    as-is instead of raising.
    """
    if not isinstance(value, str) or "${" not in value:
        return value

    def lookup(step_id: str, key_path: str) -> Any:
        if step_id not in outputs:
            if tolerant:
                return None
            raise WorkflowError(
                f"placeholder references unknown or unexecuted step: {step_id!r}"
            )
        obj: Any = outputs[step_id]
        for part in key_path.split("."):
            if isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                if tolerant:
                    return None
                raise WorkflowError(
                    f"placeholder ${{steps.{step_id}.{key_path}}} references "
                    f"missing key '{part}'"
                )
        return obj

    def replace(m: re.Match[str]) -> str:
        result = lookup(m.group(1), m.group(2))
        return value if result is None else str(result)

    # Full-string placeholder -> return typed object (or original if tolerant).
    m = _INTERP.fullmatch(value.strip())
    if m is not None:
        result = lookup(m.group(1), m.group(2))
        if result is None and tolerant:
            return value
        return result

    # Embedded placeholder -> stringify substitution.
    return _INTERP.sub(replace, value)


def _resolve_inputs(
    raw_inputs: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    tolerant: bool = False,
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for key, val in raw_inputs.items():
        resolved[key] = _resolve_value(val, outputs, tolerant=tolerant)
    return resolved


def _build_command(
    script_path: Path, inputs: dict[str, Any]
) -> list[str]:
    """Build the command list for a step's script invocation.

    Boolean ``True`` values emit only the flag (``--flag``) without a value,
    matching argparse ``store_true`` semantics. ``False`` and ``None`` omit
    the flag entirely.
    """
    cmd: list[str] = [sys.executable, str(script_path)]
    for key, val in inputs.items():
        if val is None or val is False:
            continue
        if val is True:
            cmd.append(key)
            continue
        cmd.append(key)
        cmd.append(str(val))
    return cmd


def _run_step(
    step: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
    cwd: Path,
    dry_run: bool = False,
) -> StepResult:
    """Execute a single workflow step."""
    sid = step["id"]
    try:
        resolved_inputs = _resolve_inputs(
            step["inputs"], outputs, tolerant=dry_run
        )
    except WorkflowError as exc:
        return StepResult(sid, False, None, str(exc), 1)

    script_path = cwd / step["script"]

    if dry_run:
        cmd = _build_command(script_path, resolved_inputs)
        exists = script_path.is_file()
        # Dry-run always succeeds; missing scripts are reported as a warning.
        return StepResult(
            sid,
            True,
            {"dry_run": True, "command": cmd, "script_exists": exists},
            "" if exists else f"warning: script not found: {step['script']}",
            0,
        )

    if not script_path.is_file():
        return StepResult(
            sid, False, None, f"script not found: {step['script']}", 1
        )

    cmd = _build_command(script_path, resolved_inputs)

    try:
        process = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return StepResult(sid, False, None, "step timed out after 300s", 1)
    except Exception as exc:
        return StepResult(sid, False, None, f"{type(exc).__name__}: {exc}", 1)

    if process.returncode != 0:
        return StepResult(
            sid,
            False,
            None,
            f"exit code {process.returncode}: {process.stderr.strip()[:500]}",
            process.returncode,
        )

    # Parse JSON output.
    stdout = process.stdout.strip()
    if not stdout:
        return StepResult(
            sid, False, None, "script produced no stdout output", process.returncode
        )
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return StepResult(
            sid,
            False,
            None,
            f"script stdout is not valid JSON: {exc}: {stdout[:200]}",
            process.returncode,
        )
    if not isinstance(parsed, dict):
        return StepResult(
            sid,
            False,
            None,
            f"script stdout JSON is not an object: {type(parsed).__name__}",
            process.returncode,
        )

    outputs[sid] = parsed
    return StepResult(sid, True, parsed, "", process.returncode)


def run_workflow(
    workflow_path: Path,
    start_from: str | None = None,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Execute a workflow file and return a structured summary.

    Parameters
    ----------
    workflow_path:
        Path to the YAML workflow file.
    start_from:
        If given, skip the named step itself but re-execute all steps that
        the named step depends on (transitively) so their outputs are
        available for the resumed step and its downstream siblings.  This
        ensures placeholders referencing prior steps resolve correctly.
    dry_run:
        If true, resolve and print commands without executing scripts.
    cwd:
        Working directory for resolving relative script paths. Defaults to
        the workflow file's parent directory.
    """
    base_dir = cwd or workflow_path.parent.resolve()
    text = workflow_path.read_text(encoding="utf-8")

    try:
        wf = parse_yaml(text)
    except YamlParseError as exc:
        return {
            "ok": False,
            "error": f"YAML parse error: {exc}",
            "workflow": str(workflow_path),
        }

    try:
        steps = _validate_workflow(wf)
        ordered = _topo_sort(steps)
    except WorkflowError as exc:
        return {
            "ok": False,
            "error": f"workflow validation error: {exc}",
            "workflow": str(workflow_path),
        }

    # Determine which steps to re-execute vs. skip for restart.
#
# When --start-from is given, the named step and all steps that come *after*
# it in topological order are executed normally. Steps that come *before* it
# are re-executed only if they are (transitively) depended upon by the
# start-from step or any step after it -- this ensures their outputs are
# available for placeholder resolution. Steps that no downstream step needs
# are marked skipped.
    executed_ids: set[str] = set()
    if start_from is not None:
        ids = [s["id"] for s in ordered]
        if start_from not in ids:
            return {
                "ok": False,
                "error": f"--start-from step not found: {start_from!r}",
                "workflow": str(workflow_path),
                "available_steps": ids,
            }
        start_idx = ids.index(start_from)
        # Collect all step ids from start_idx onward (the "resume" set).
        resume_ids = {s["id"] for s in ordered[start_idx:]}
        # Transitively collect dependencies of the resume set.
        needed: set[str] = set()
        _collect_deps(ordered, resume_ids, needed)
        executed_ids = resume_ids | needed
    else:
        start_idx = 0
        executed_ids = {s["id"] for s in ordered}

    outputs: dict[str, dict[str, Any]] = {}
    results: list[StepResult] = []

    failed = False
    for step in ordered:
        sid = step["id"]
        if sid not in executed_ids:
            results.append(
                StepResult(
                    sid, True, None, "skipped (not needed for restart)", 0,
                    status=StepResult.STATUS_SKIPPED_RESTART,
                )
            )
            continue
        if failed:
            results.append(
                StepResult(
                    sid, False, None, "skipped (prior failure)", 0,
                    status=StepResult.STATUS_SKIPPED_FAILURE,
                )
            )
            continue
        result = _run_step(step, outputs, base_dir, dry_run=dry_run)
        if not result.ok:
            result.status = StepResult.STATUS_FAILED
        results.append(result)
        if not result.ok:
            failed = True

    summary = {
        "ok": not failed,
        "workflow": str(workflow_path),
        "name": wf.get("name", ""),
        "description": wf.get("description", ""),
        "steps_total": len(ordered),
        "steps_executed": sum(
            1 for r in results
            if r.status in (StepResult.STATUS_SUCCESS, StepResult.STATUS_FAILED)
        ),
        "results": [r.to_dict() for r in results],
        "outputs": outputs,
    }
    if failed:
        summary["error"] = next(
            (r.error for r in results if r.status == StepResult.STATUS_FAILED and r.error),
            "one or more steps failed",
        )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Execute a composable local skill-workflow from a YAML file."
    )
    parser.add_argument("workflow", type=Path, help="Path to the workflow YAML file.")
    parser.add_argument(
        "--start-from",
        metavar="STEP_ID",
        help="Resume execution from the step after the given id (restart from failure).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print commands without executing scripts.",
    )
    args = parser.parse_args()

    if not args.workflow.is_file():
        print(
            json.dumps(
                {"ok": False, "error": f"workflow file not found: {args.workflow}"},
                indent=2,
            )
        )
        return 1

    summary = run_workflow(
        args.workflow,
        start_from=args.start_from,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())