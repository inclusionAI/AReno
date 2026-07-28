"""Streaming JSONL quality scanner.

Scans JSONL files (or stdin) line-by-line without loading the full dataset
into memory, and reports data quality issues: blank lines, JSON parse
errors, non-object records, and schema surprises.

The scanner keeps only a bounded, redacted preview of bad entries so that
sensitive training data is never exposed in logs or CLI output.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import IO, Any, Iterable


class ErrorType(str, Enum):
    """Categories of quality issues detected during scanning."""

    BLANK_LINE = "blank_line"
    JSON_PARSE = "json_parse"
    NON_OBJECT = "non_object"
    SCHEMA_MISSING_FIELD = "schema_missing_field"
    SCHEMA_EMPTY_FIELD = "schema_empty_field"


@dataclass
class ScanError:
    """A single quality issue found during scanning.

    ``line_number`` is 1-based.  ``detail`` is a short, redacted
    description — it never contains the full record text.
    """

    line_number: int
    error_type: ErrorType
    detail: str = ""


@dataclass
class ScanResult:
    """Aggregate quality report for a JSONL scan.

    ``errors`` is capped at ``max_errors`` entries; ``errors_truncated``
    records how many additional errors were observed but not stored.
    """

    total_lines: int = 0
    valid_records: int = 0
    blank_lines: int = 0
    json_errors: int = 0
    non_object_records: int = 0
    schema_issues: int = 0
    errors: list[ScanError] = field(default_factory=list)
    errors_truncated: int = 0

    @property
    def total_errors(self) -> int:
        """Total count of all error categories."""

        return (
            self.blank_lines
            + self.json_errors
            + self.non_object_records
            + self.schema_issues
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""

        return {
            "total_lines": self.total_lines,
            "valid_records": self.valid_records,
            "blank_lines": self.blank_lines,
            "json_errors": self.json_errors,
            "non_object_records": self.non_object_records,
            "schema_issues": self.schema_issues,
            "total_errors": self.total_errors,
            "errors": [
                {
                    "line": e.line_number,
                    "type": e.error_type.value,
                    "detail": e.detail,
                }
                for e in self.errors
            ],
            "errors_truncated": self.errors_truncated,
        }


# ---------------------------------------------------------------------------
# Core scanning logic
# ---------------------------------------------------------------------------

def _redact(text: str, max_len: int = 80) -> str:
    """Truncate and redact text for safe preview output.

    Ensures that error details in logs and CLI output never expose
    full training data records.  Only the first ``max_len`` characters
    are shown, followed by a ``[redacted]`` marker.
    """

    text = text.strip()
    if len(text) > max_len:
        return text[:max_len] + "... [redacted]"
    return text


def scan_jsonl(
    source: str | IO[str] | Iterable[str],
    *,
    required_fields: list[str] | None = None,
    max_errors: int = 100,
) -> ScanResult:
    """Stream-scan a JSONL source and return a quality report.

    Parameters
    ----------
    source:
        A file path (``str``), a file-like object, or any iterable of
        lines.  When a path is given the file is opened in read-only
        text mode and closed after scanning.
    required_fields:
        Field names that every valid object record must contain and
        whose values must be non-empty strings.  ``None`` disables
        schema checks.
    max_errors:
        Maximum number of :class:`ScanError` entries stored in the
        result.  Additional errors are counted but not retained.

    The scanner never raises on malformed input — every recoverable
    error is recorded and scanning continues.
    """

    result = ScanResult()

    lines = _iter_lines(source)

    for line_number, raw_line in enumerate(lines, start=1):
        result.total_lines += 1
        stripped = raw_line.strip()

        # --- blank line --------------------------------------------------
        if not stripped:
            result.blank_lines += 1
            _add_error(result, ScanError(line_number, ErrorType.BLANK_LINE), max_errors)
            continue

        # --- JSON parse --------------------------------------------------
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            result.json_errors += 1
            _add_error(
                result,
                ScanError(line_number, ErrorType.JSON_PARSE, _redact(str(exc))),
                max_errors,
            )
            continue

        # --- non-object --------------------------------------------------
        if not isinstance(record, dict):
            result.non_object_records += 1
            _add_error(
                result,
                ScanError(
                    line_number,
                    ErrorType.NON_OBJECT,
                    f"type is {type(record).__name__}",
                ),
                max_errors,
            )
            continue

        # --- schema checks ----------------------------------------------
        if required_fields:
            for field_name in required_fields:
                if field_name not in record:
                    result.schema_issues += 1
                    _add_error(
                        result,
                        ScanError(
                            line_number,
                            ErrorType.SCHEMA_MISSING_FIELD,
                            f"missing field '{field_name}'",
                        ),
                        max_errors,
                    )
                elif record[field_name] is None or (
                    isinstance(record[field_name], str) and not record[field_name].strip()
                ):
                    result.schema_issues += 1
                    _add_error(
                        result,
                        ScanError(
                            line_number,
                            ErrorType.SCHEMA_EMPTY_FIELD,
                            f"empty field '{field_name}'",
                        ),
                        max_errors,
                    )

        # If we got here without adding a schema error the record is valid.
        if not result.errors or result.errors[-1].line_number != line_number:
            result.valid_records += 1

    return result


def _iter_lines(source: str | IO[str] | Iterable[str]) -> Iterable[str]:
    """Yield lines from a path, file-like object, or iterable.

    When *source* is a string it is treated as a file path and opened
    in read-only text mode.  The file is automatically closed after
    iteration completes (or is interrupted).  When *source* is already
    an iterable (e.g. ``sys.stdin`` or a list) it is consumed directly.
    """

    if isinstance(source, str):
        with open(source, encoding="utf-8") as fh:
            yield from fh
    elif hasattr(source, "__iter__"):
        yield from source
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")


def _add_error(result: ScanResult, error: ScanError, max_errors: int) -> None:
    """Append an error to the result, respecting the bounded preview.

    Once ``len(result.errors)`` reaches ``max_errors``, subsequent
    errors are only counted in ``result.errors_truncated`` (not stored
    as :class:`ScanError` objects) to keep memory usage bounded.
    """

    if len(result.errors) < max_errors:
        result.errors.append(error)
    else:
        result.errors_truncated += 1


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_table(result: ScanResult) -> str:
    """Produce a human-readable table from scan results.

    The table includes:
      - Summary counts (total lines, valid records, error breakdown)
      - Bounded error preview with line numbers and redacted details
      - Truncation notice when errors exceed ``max_errors``

    This format is intended for interactive terminal use.
    """

    lines = [
        "JSONL Quality Scan Report",
        "=" * 40,
        "",
        f"  Total lines:        {result.total_lines:>10}",
        f"  Valid records:      {result.valid_records:>10}",
        f"  Blank lines:        {result.blank_lines:>10}",
        f"  JSON errors:        {result.json_errors:>10}",
        f"  Non-object records: {result.non_object_records:>10}",
        f"  Schema issues:      {result.schema_issues:>10}",
        f"  Total errors:       {result.total_errors:>10}",
    ]

    if result.errors:
        shown = len(result.errors)
        truncated = result.errors_truncated
        header = f"\nError preview (showing {shown}"
        if truncated:
            header += f" of {shown + truncated}"
        header += "):"
        lines.append(header)

        for err in result.errors:
            detail = f" - {err.detail}" if err.detail else ""
            lines.append(f"  line {err.line_number:>6}: {err.error_type.value}{detail}")

    if result.errors_truncated:
        lines.append(f"\n  ({result.errors_truncated} additional errors not shown)")

    return "\n".join(lines)


def render_json(result: ScanResult) -> str:
    """Produce machine-readable JSON from scan results.

    The JSON output is sorted and indented for readability while
    remaining parseable by CI/CD pipelines and dashboard consumers.
    All field names match those in :meth:`ScanResult.to_dict`.
    """

    return json.dumps(result.to_dict(), indent=2, sort_keys=True)
