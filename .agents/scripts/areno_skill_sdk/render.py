"""Table and JSON rendering for skill scripts.

Dual-mode output: JSON mode writes pure JSON to stdout (machine-clean); human
mode renders rich tables to stderr so stdout stays machine-clean even when
human mode is on.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO


def emit(
    result: dict[str, Any],
    *,
    json_mode: bool = True,
    indent: int = 2,
    sort_keys: bool = True,
    ensure_ascii: bool = True,
    stream: TextIO | None = None,
) -> None:
    """Emit a result dict.

    ``json_mode=True`` (default) writes only JSON to stdout, preserving the
    existing "stdout is JSON" contract of the unmigrated scripts.
    ``json_mode=False`` writes human-readable rich text to stderr so stdout
    stays machine-clean. ``stream`` overrides the JSON output destination (used
    by tests and JSON-Lines-style scripts).

    ``sort_keys`` and ``ensure_ascii`` default to the values used by the
    majority of legacy scripts (sorted, ASCII-escaped). Scripts that previously
    emitted unsorted JSON with non-ASCII content preserved (e.g.
    ``inspect_dataset``) pass ``sort_keys=False, ensure_ascii=False`` to keep
    their exact pre-migration byte output.
    """
    if json_mode:
        target = stream if stream is not None else sys.stdout
        target.write(
            json.dumps(result, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii) + "\n"
        )
        target.flush()
    else:
        _emit_human(result)


def _emit_human(result: dict[str, Any]) -> None:
    """Render a human-readable summary to stderr.

    stderr is used deliberately so that stdout remains machine-clean even in
    human mode, satisfying the issue's "stdout machine-clean in JSON mode"
    acceptance criterion. Uses ``rich`` when available and falls back to plain
    text so the SDK remains importable in minimal CPU environments without the
    full AReno runtime deps installed.
    """
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        _emit_human_plain(result)
        return

    console = Console(file=sys.stderr)
    if result.get("ok"):
        console.print("[green]OK[/green]")
    else:
        message = result.get("error") or result.get("errors") or "failed"
        console.print(f"[red]FAIL[/red]: {message}")
    for key, value in result.items():
        if key in ("ok", "error", "errors", "stage"):
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            _render_table_rich(value, console, Table, title=key)
        elif not isinstance(value, (list, dict)):
            console.print(f"  [bold]{key}[/bold]: {value}")


def _emit_human_plain(result: dict[str, Any]) -> None:
    """Plain-text fallback when ``rich`` is not installed."""
    stream = sys.stderr
    if result.get("ok"):
        stream.write("OK\n")
    else:
        message = result.get("error") or result.get("errors") or "failed"
        stream.write(f"FAIL: {message}\n")
    for key, value in result.items():
        if key in ("ok", "error", "errors", "stage"):
            continue
        if isinstance(value, list) and value and isinstance(value[0], dict):
            _render_table_plain(value, stream, title=key)
        elif not isinstance(value, (list, dict)):
            stream.write(f"  {key}: {value}\n")


def _render_table_rich(rows: list[dict[str, Any]], console: Any, Table: Any, *, title: str) -> None:
    """Render a list of homogeneous dicts as a rich table."""
    table = Table(title=title, show_lines=False)
    columns = list(rows[0].keys())
    for column in columns:
        table.add_column(column)
    for row in rows:
        table.add_row(*(_format_cell(row.get(column)) for column in columns))
    console.print(table)


def _render_table_plain(rows: list[dict[str, Any]], stream: TextIO, *, title: str) -> None:
    """Render a list of homogeneous dicts as a plain aligned table."""
    columns = list(rows[0].keys())
    widths = [max(len(column), *(len(_format_cell(row.get(column))) for row in rows)) for column in columns]
    stream.write(f"\n{title}\n")
    stream.write("  " + "  ".join(column.ljust(widths[i]) for i, column in enumerate(columns)) + "\n")
    for row in rows:
        stream.write("  " + "  ".join(_format_cell(row.get(column)).ljust(widths[i]) for i, column in enumerate(columns)) + "\n")


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return str(value)