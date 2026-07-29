"""Pure-function log filtering for ``areno logs``.

This module contains no I/O so it is trivially unit-testable on CPU.
``LogEntry`` represents one parsed log line; ``FilterSpec`` collects the
user-supplied filter dimensions; ``matches`` applies an AND across all
specified dimensions.

The entry fields mirror AReno's two logging formats:

* CLI:      ``%(asctime)s %(levelname)s %(name)s: %(message)s``
* Engine:   ``%(asctime)s %(levelname)s %(name)s %(filename's:%(lineno)d - %(message)s``
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Valid values exposed to the CLI.
VALID_STAGES = frozenset({"train", "eval", "rollout", "serve"})
VALID_SEVERITIES = frozenset({"debug", "info", "warn", "error"})


@dataclass
class LogEntry:
    """One parsed log line.

    ``raw`` preserves the original un-parsed text for debugging and for
    lines that do not match the expected format (in which case only
    ``raw`` and ``source`` are populated).
    """

    timestamp: str
    severity: str
    source: str
    message: str
    rank: int
    stage: str
    raw: str


@dataclass
class FilterSpec:
    """User-supplied filter dimensions.

    A dimension set to ``None`` means "do not filter on this dimension".
    All specified dimensions are combined with AND logic.
    """

    rank: int | None = None
    stage: str | None = None
    severity: str | None = None
    text_pattern: re.Pattern | None = None


def matches(entry: LogEntry, spec: FilterSpec) -> bool:
    """Return ``True`` if *entry* satisfies every dimension in *spec*.

    Dimensions that are ``None`` in *spec* are skipped (i.e. match everything).
    """

    if spec.rank is not None and entry.rank != spec.rank:
        return False
    if spec.stage is not None and entry.stage != spec.stage:
        return False
    if spec.severity is not None and entry.severity != spec.severity:
        return False
    if spec.text_pattern is not None and not spec.text_pattern.search(entry.message):
        return False
    return True


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------

# AReno CLI format:  "2026-07-28 10:00:00,123 INFO areno.engine.training: step=0"
# AReno engine format: "2026-07-28 10:00:00 INFO areno.engine.training training.py:45 - step=0"
#
# Both start with <timestamp> <LEVEL> <logger-name>.  We capture the common
# prefix and treat the remainder as the message.
_LOG_RE = re.compile(
    r"^("
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{3,6})?"
    r")\s+"
    r"(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\s+"
    r"([^\s:]+)\s*[:\-]?\s*"   # logger name (no trailing colon) + optional : or -
    r"(.*)$"                   # rest = message
)

# Maps raw level strings to the canonical lowercase severity.
_SEVERITY_MAP = {
    "debug": "debug",
    "info": "info",
    "warn": "warn",
    "warning": "warn",
    "error": "error",
    "critical": "error",
}

# Maps logger-name fragments to stages.
_STAGE_KEYWORDS = (
    ("training", "train"),
    ("trainer", "train"),
    ("train", "train"),
    ("inference", "rollout"),
    ("rollout", "rollout"),
    ("eval", "eval"),
    ("evaluate", "eval"),
    ("serve", "serve"),
    ("server", "serve"),
)


def parse_line(raw: str, source: str = "") -> LogEntry:
    """Parse a single raw log line into a :class:`LogEntry`.

    Lines that do not match the expected format are returned with only
    ``raw`` and ``source`` populated; everything else is empty/zero.
    """

    stripped = raw.rstrip("\n\r")
    m = _LOG_RE.match(stripped)
    if m is None:
        return LogEntry(
            timestamp="",
            severity="",
            source=source,
            message="",
            rank=-1,
            stage="",
            raw=stripped,
        )

    timestamp, level, logger_name, rest = m.groups()
    severity = _SEVERITY_MAP.get(level.lower(), level.lower())

    # The "rest" may start with "filename:lineno - " (engine format) or be
    # the message directly (CLI format).  Strip the filename:lineno prefix
    # if present.
    message = _strip_file_lineno(rest)

    rank = _extract_rank(message, logger_name)
    stage = _infer_stage(logger_name)

    return LogEntry(
        timestamp=timestamp,
        severity=severity,
        source=logger_name,
        message=message,
        rank=rank,
        stage=stage,
        raw=stripped,
    )


_FILE_LINENO_RE = re.compile(r"^\S+\.py:\d+\s*-\s*")


def _strip_file_lineno(rest: str) -> str:
    """Remove a leading ``filename.py:lineno - `` prefix if present."""
    return _FILE_LINENO_RE.sub("", rest)


_RANK_RE = re.compile(r"\brank[=\s:](\d+)", re.IGNORECASE)


def _extract_rank(message: str, logger_name: str) -> int:
    """Try to find a rank number in the message or logger name."""
    m = _RANK_RE.search(message)
    if m:
        return int(m.group(1))
    m = _RANK_RE.search(logger_name)
    if m:
        return int(m.group(1))
    return -1


def _infer_stage(logger_name: str) -> str:
    """Infer the training stage from the logger name."""
    lower = logger_name.lower()
    for keyword, stage in _STAGE_KEYWORDS:
        if keyword in lower:
            return stage
    return ""


def compile_grep(pattern: str) -> re.Pattern[str]:
    """Compile a user-supplied grep pattern.

    Raises ``re.error`` on invalid patterns; the CLI layer converts that
    into a user-facing validation message.
    """
    return re.compile(pattern)