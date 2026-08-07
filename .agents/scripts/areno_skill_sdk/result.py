"""Stable result objects and exit codes for skill scripts.

Unifies the ``{"ok": bool, ...}`` shape used across the 24 skill scripts.
``Result.to_dict()`` produces output structurally identical to the hand-built
dicts in the unmigrated scripts, so existing tests asserting stdout JSON keep
passing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Sentinel letting `Result` distinguish "caller did not pass errors" from
# "caller passed an empty list". Scripts like check_capacity always emit an
# `errors` field (even when empty) to match their pre-migration JSON shape, so
# they pass `errors=[]` explicitly; scripts that never use multi-error reporting
# leave it as the sentinel and the field is omitted.
_NO_ERRORS: list[str] = []


@dataclass
class Result:
    """Structured outcome of a skill script.

    ``ok`` is the success flag. ``data`` carries business fields. ``errors``
    holds multiple validation messages (kept as a list to preserve the existing
    ``errors`` field name). ``stage`` identifies the failed stage on error.
    """

    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default=None)  # type: ignore[assignment]
    stage: str | None = None

    def __post_init__(self) -> None:
        # Normalize the sentinel default: ``None`` means "no errors field".
        if self.errors is None:
            self.errors = _NO_ERRORS

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the ``{"ok": ..., ...}`` contract.

        Field ordering: ``ok`` first, then ``stage``/``errors`` if present,
        then business ``data`` fields. ``errors`` is emitted only when the
        caller passed it explicitly (including an empty list), so scripts that
        never report multi-errors do not gain a spurious field, while scripts
        like ``check_capacity`` keep their exact pre-migration shape.
        """
        out: dict[str, Any] = {"ok": self.ok}
        if self.stage:
            out["stage"] = self.stage
        if self.errors is not _NO_ERRORS:
            out["errors"] = self.errors
        out.update(self.data)
        return out


def exit_code(result: dict[str, Any]) -> int:
    """Map a result dict to a process exit code.

    ``ok=True`` -> 0, otherwise -> 1. Preserves the existing
    ``0 if result["ok"] else 1`` semantics used by every unmigrated script.
    """
    return 0 if result.get("ok") else 1
