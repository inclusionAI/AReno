"""Output formatting for ``areno logs``.

Two formats are supported:

* ``text`` — human-readable with optional ANSI colour when writing to a TTY.
* ``json`` — one JSON object per line (JSONL) for machine consumption.

Both formats preserve timestamp and source context as required by the issue.
"""

from __future__ import annotations

import json
import sys
from typing import Literal

from areno.cli.log_filter import LogEntry

OutputFormat = Literal["text", "json"]

# ANSI colour codes keyed by severity.
_SEVERITY_COLORS: dict[str, str] = {
    "error": "\033[31m",   # red
    "warn": "\033[33m",    # yellow
    "info": "\033[32m",    # green
    "debug": "\033[36m",   # cyan
}
_RESET = "\033[0m"


class LogFormatter:
    """Format :class:`LogEntry` objects for CLI output."""

    def __init__(self, output: OutputFormat = "text") -> None:
        self.output = output
        # Only use colour in text mode when stdout is a TTY.
        self.use_color = output == "text" and sys.stdout.isatty()

    def format(self, entry: LogEntry) -> str:
        """Return the formatted string for one entry."""

        if self.output == "json":
            return self._format_json(entry)
        return self._format_text(entry)

    def _format_text(self, entry: LogEntry) -> str:
        """Human-readable: ``[timestamp] [stage] [rank N] [SEVERITY] message``.

        Lines that failed to parse (no timestamp) are emitted as-is.
        """

        if not entry.timestamp:
            return entry.raw

        parts: list[str] = []

        # Timestamp.
        parts.append(f"[{entry.timestamp}]")

        # Stage (if known).
        if entry.stage:
            parts.append(f"[{entry.stage}]")

        # Rank (if known and valid).
        if entry.rank >= 0:
            parts.append(f"[rank {entry.rank}]")

        # Severity with optional colour.
        severity_upper = entry.severity.upper() if entry.severity else "UNKNOWN"
        if self.use_color and entry.severity in _SEVERITY_COLORS:
            color = _SEVERITY_COLORS[entry.severity]
            parts.append(f"{color}[{severity_upper}]{_RESET}")
        else:
            parts.append(f"[{severity_upper}]")

        # Source (logger name or file label).
        if entry.source:
            parts.append(f"({entry.source})")

        header = " ".join(parts)
        return f"{header} {entry.message}"

    @staticmethod
    def _format_json(entry: LogEntry) -> str:
        """One JSON object per line (JSONL)."""

        return json.dumps(
            {
                "timestamp": entry.timestamp,
                "severity": entry.severity,
                "source": entry.source,
                "message": entry.message,
                "rank": entry.rank if entry.rank >= 0 else None,
                "stage": entry.stage or None,
            },
            ensure_ascii=False,
        )


def format_error(
    *,
    stage: str,
    input_name: str,
    message: str,
    output: OutputFormat = "text",
) -> str:
    """Format a validation error in the requested output mode.

    The error always identifies the affected stage and input without
    exposing training samples or hiding the original error.
    """

    if output == "json":
        return json.dumps(
            {
                "error": {
                    "stage": stage,
                    "input": input_name,
                    "message": message,
                }
            },
            ensure_ascii=False,
        )
    return f"Error [stage={stage}] {message} (input: {input_name})"