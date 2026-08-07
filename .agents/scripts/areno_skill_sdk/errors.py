"""Exception envelopes for skill scripts.

Unifies try/except into structured errors that identify the affected stage
without exposing raw training samples.
"""

from __future__ import annotations


class SkillError(Exception):
    """Business error carrying the stage that failed.

    Raised by skill scripts to abort with a structured envelope. The ``stage``
    attribute satisfies the issue's "failure identifies the affected stage"
    requirement.
    """

    def __init__(self, message: str, *, stage: str = "execute") -> None:
        super().__init__(message)
        self.stage = stage


def envelope(exc: Exception, *, stage: str = "execute") -> dict[str, object]:
    """Wrap any exception into a ``{"ok": False, "error": ..., "stage": ...}`` dict.

    The ``error`` field matches the existing
    ``f"{type(exc).__name__}: {exc}"`` convention used across the unmigrated
    scripts, only adding a ``stage`` field.
    """
    return {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "stage": stage,
    }
