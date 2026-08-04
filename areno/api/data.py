"""Lightweight dataclasses that flow through the rollout/training pipeline.

`PromptItem` is the unit produced by `Trainer.load_prompt_batches` after
tokenising a dataset row. `PromptBatch` groups a fixed-size set of items
together and carries diagnostic counters so the trainer can surface how many
records were skipped for exceeding the prompt-length budget.

The streaming JSONL quality scanner (`scan_jsonl_stream`) inspects a file or
stdin line by line without loading the full dataset into memory, reporting blank
lines, JSON failures, non-object records, and bounded redacted previews of bad
entries. `scan_loader_output` applies the same checks to records produced by a
``--dataset-loader-fn`` so users can validate custom normalisation code before
starting a training run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass(slots=True)
class PromptItem:
    """A dataset record after prompt tokenization and length filtering.

    `prompt` keeps the raw text used for downstream decoding/rewards,
    `input_tokens` holds the tokenized prefix that will be prepended to every
    rollout response, and `record` preserves the original row so reward
    functions can read task-specific fields (gold answers, test cases, ...).
    """

    prompt: str
    solutions: list[str] | None
    input_tokens: list[int]
    record: dict[str, Any]


@dataclass(slots=True)
class PromptBatch:
    """A batch of prompts plus counters for skipped over-length examples.

    `scanned` is how many raw dataset rows were inspected to build this batch
    (including skips), `skipped_long` is how many were dropped this round, and
    `total_skipped_long` accumulates the drop count across the epoch so the
    metric logger can report it as a cumulative counter.
    """

    items: list[PromptItem]
    scanned: int
    skipped_long: int
    total_skipped_long: int

    @property
    def prompts(self) -> list[str]:
        """Return raw prompt strings in batch order for rollout."""

        return [item.prompt for item in self.items]


# ---------------------------------------------------------------------------
# Streaming JSONL / loader-output quality scanner
# ---------------------------------------------------------------------------

# Categories reported by the scanner.  ``BLANK`` is a whitespace-only line,
# ``JSON_ERROR`` is a line that could not be parsed as JSON, ``NON_OBJECT`` is
# valid JSON that is not a JSON object (e.g. an array or scalar), and
# ``SCHEMA`` is an object missing keys declared as required.
SCAN_BLANK = "blank"
SCAN_JSON_ERROR = "json_error"
SCAN_NON_OBJECT = "non_object"
SCAN_SCHEMA = "schema"

# Hard cap on the raw-line preview stored in a ``ScanIssue``.  This keeps
# memory bounded even for pathological inputs.
_PREVIEW_MAX_CHARS = 200


@dataclass(slots=True)
class ScanIssue:
    """A single problem found while scanning one record/line.

    ``line_number`` is 1-based; ``raw_preview`` is a truncated, redacted copy
    of the offending line or record so logs stay safe to share.
    """

    category: str
    line_number: int
    detail: str
    raw_preview: str


@dataclass(slots=True)
class ScanReport:
    """Aggregated result of a streaming scan pass.

    ``total_lines`` counts every physical line inspected (including blanks).
    ``object_lines`` is how many parsed as JSON objects.  ``issues`` is a
    bounded list (see ``max_issues``) so memory stays predictable on large
    files; ``truncated_issues`` records how many additional issues were
    dropped after the cap.
    """

    total_lines: int = 0
    object_lines: int = 0
    blank_lines: int = 0
    json_error_lines: int = 0
    non_object_lines: int = 0
    schema_error_lines: int = 0
    issues: list[ScanIssue] = field(default_factory=list)
    truncated_issues: int = 0
    source: str = ""

    @property
    def ok(self) -> bool:
        """Whether the scan found no issues at all."""

        return not self.issues and self.truncated_issues == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON / structured output."""

        return {
            "source": self.source,
            "total_lines": self.total_lines,
            "object_lines": self.object_lines,
            "blank_lines": self.blank_lines,
            "json_error_lines": self.json_error_lines,
            "non_object_lines": self.non_object_lines,
            "schema_error_lines": self.schema_error_lines,
            "truncated_issues": self.truncated_issues,
            "issues": [
                {
                    "category": issue.category,
                    "line_number": issue.line_number,
                    "detail": issue.detail,
                    "raw_preview": issue.raw_preview,
                }
                for issue in self.issues
            ],
        }


def _truncate_preview(text: str) -> str:
    """Truncate a raw line/record to a bounded length for safe logging."""

    if len(text) <= _PREVIEW_MAX_CHARS:
        return text
    return text[:_PREVIEW_MAX_CHARS] + "..."


def _redact_preview(text: str) -> str:
    """Redact values that look like secrets from a preview string.

    Only applied to raw previews of bad lines so we never expose full training
    samples in logs or CLI output.  Key-based redaction is intentionally
    lightweight: it covers common secret key names without pulling in a regex
    dependency beyond the stdlib.
    """

    import re

    # Replace the value of keys that commonly carry secrets/tokens.
    pattern = re.compile(
        r'("(?:api[_-]?key|token|secret|password|access[_-]?key|credential)"\s*:\s*")([^"]*)(")',
        re.IGNORECASE,
    )
    return pattern.sub(r"\1<redacted>\3", text)


def _record_issue(
    report: ScanReport,
    *,
    category: str,
    line_number: int,
    detail: str,
    raw_preview: str,
    max_issues: int,
) -> None:
    """Append an issue to the report, respecting the ``max_issues`` cap."""

    if len(report.issues) < max_issues:
        report.issues.append(
            ScanIssue(
                category=category,
                line_number=line_number,
                detail=detail,
                raw_preview=_redact_preview(_truncate_preview(raw_preview)),
            )
        )
    else:
        report.truncated_issues += 1


def scan_jsonl_stream(
    source: TextIO | str | Path,
    *,
    required_keys: tuple[str, ...] = (),
    max_issues: int = 50,
    encoding: str = "utf-8",
) -> ScanReport:
    """Stream-scan a JSONL file or stdin, reporting quality issues.

    The file is read line by line so memory stays bounded regardless of file
    size.  Each non-blank line is parsed as JSON; lines that fail to parse, are
    not JSON objects, or miss any of *required_keys* are recorded as issues
    with their 1-based line number and a truncated, redacted preview.

    Parameters
    ----------
    source:
        A path/str to a JSONL file, or a text stream (e.g. ``sys.stdin``).
    required_keys:
        Object keys that every record must contain.  Empty by default, meaning
        only structural checks (blank / JSON / object) are performed.
    max_issues:
        Maximum number of :class:`ScanIssue` entries stored in the report.
        Additional issues are counted in ``truncated_issues``.
    encoding:
        File encoding when *source* is a path.

    Returns
    -------
    ScanReport
        Aggregated counts and a bounded list of issues.
    """

    if max_issues < 0:
        raise ValueError("max_issues must be non-negative")

    report = ScanReport()
    owns_handle = False
    stream: TextIO

    if isinstance(source, str | Path):
        path = Path(source)
        report.source = str(path)
        stream = path.open("r", encoding=encoding)
        owns_handle = True
    else:
        report.source = getattr(source, "name", "<stdin>")
        stream = source

    try:
        line_number = 0
        for raw_line in stream:
            line_number += 1
            report.total_lines += 1
            stripped = raw_line.strip()
            if not stripped:
                report.blank_lines += 1
                _record_issue(
                    report,
                    category=SCAN_BLANK,
                    line_number=line_number,
                    detail="blank line",
                    raw_preview="",
                    max_issues=max_issues,
                )
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report.json_error_lines += 1
                _record_issue(
                    report,
                    category=SCAN_JSON_ERROR,
                    line_number=line_number,
                    detail=f"invalid JSON: {exc.msg}",
                    raw_preview=stripped,
                    max_issues=max_issues,
                )
                continue
            if not isinstance(record, dict):
                report.non_object_lines += 1
                _record_issue(
                    report,
                    category=SCAN_NON_OBJECT,
                    line_number=line_number,
                    detail=f"expected JSON object, got {type(record).__name__}",
                    raw_preview=stripped,
                    max_issues=max_issues,
                )
                continue
            missing = [k for k in required_keys if k not in record]
            if missing:
                report.schema_error_lines += 1
                _record_issue(
                    report,
                    category=SCAN_SCHEMA,
                    line_number=line_number,
                    detail=f"missing required keys: {', '.join(missing)}",
                    raw_preview=stripped,
                    max_issues=max_issues,
                )
                continue
            report.object_lines += 1
    finally:
        if owns_handle:
            stream.close()

    return report


def scan_loader_output(
    records: Iterable[Any],
    *,
    required_keys: tuple[str, ...] = (),
    max_issues: int = 50,
    source: str = "<loader>",
) -> ScanReport:
    """Scan records returned by a ``--dataset-loader-fn`` for quality issues.

    This complements :func:`scan_jsonl_stream` for cases where the user has a
    custom loader that transforms raw data into training rows.  Each record is
    checked to be a ``dict`` and to contain all *required_keys*.  Non-dict
    records and schema violations are recorded with their 1-based index.

    Parameters
    ----------
    records:
        An iterable of records produced by a loader function.
    required_keys:
        Keys every record must contain.
    max_issues:
        Maximum number of issues stored.
    source:
        Label used in the report ``source`` field.
    """

    if max_issues < 0:
        raise ValueError("max_issues must be non-negative")

    report = ScanReport(source=source)
    index = 0
    for record in records:
        index += 1
        report.total_lines += 1
        if not isinstance(record, dict):
            report.non_object_lines += 1
            _record_issue(
                report,
                category=SCAN_NON_OBJECT,
                line_number=index,
                detail=f"expected dict, got {type(record).__name__}",
                raw_preview=repr(record),
                max_issues=max_issues,
            )
            continue
        missing = [k for k in required_keys if k not in record]
        if missing:
            report.schema_error_lines += 1
            _record_issue(
                report,
                category=SCAN_SCHEMA,
                line_number=index,
                detail=f"missing required keys: {', '.join(missing)}",
                raw_preview=json.dumps(record, ensure_ascii=False, default=str),
                max_issues=max_issues,
            )
            continue
        report.object_lines += 1

    return report


def format_scan_report(report: ScanReport, *, json_output: bool = False) -> str:
    """Render a :class:`ScanReport` as human-readable text or JSON.

    The human-readable form is a short summary followed by per-issue lines.
    The JSON form is the ``to_dict`` result pretty-printed.
    """

    if json_output:
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    lines: list[str] = []
    lines.append(f"source: {report.source or '<stdin>'}")
    lines.append(
        f"total_lines: {report.total_lines}  objects: {report.object_lines}  "
        f"blank: {report.blank_lines}  json_errors: {report.json_error_lines}  "
        f"non_object: {report.non_object_lines}  schema_errors: {report.schema_error_lines}"
    )
    if report.truncated_issues:
        lines.append(f"(truncated {report.truncated_issues} additional issues)")
    if report.ok:
        lines.append("status: OK")
    else:
        lines.append("status: ISSUES FOUND")
        for issue in report.issues:
            preview = f"  preview: {issue.raw_preview}" if issue.raw_preview else ""
            lines.append(f"  line {issue.line_number}: [{issue.category}] {issue.detail}{preview}")
    return "\n".join(lines)
