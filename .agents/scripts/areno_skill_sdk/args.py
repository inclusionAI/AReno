"""Argument handling for skill scripts.

A thin wrapper over :mod:`argparse` providing a parser builder plus validation
helpers. argparse is kept (not switched to click) to preserve every existing
flag's behavior with zero migration risk.
"""

from __future__ import annotations

import argparse
from typing import Any

from .errors import SkillError


def build_parser(description: str = "", **kwargs: Any) -> argparse.ArgumentParser:
    """Build an :class:`argparse.ArgumentParser` with unified behavior.

    ``allow_abbrev=False`` disables argparse's default prefix-stripping (where
    ``--batch`` would match ``--batch-size``) so flag behavior is predictable
    during migration.
    """
    return argparse.ArgumentParser(
        description=description,
        allow_abbrev=False,
        **kwargs,
    )


def validate_positive(
    args: argparse.Namespace,
    *,
    exclude: tuple[str, ...] = (),
) -> None:
    """Validate that all numeric args are positive.

    Raises :class:`SkillError` with ``stage="validate"`` for any non-positive
    numeric value, satisfying the issue's "validate before expensive
    initialization + identify affected stage" requirement. Keys in ``exclude``
    (such as ``memory_fraction``, which has its own range check) are skipped.
    """
    for key, value in vars(args).items():
        if key in exclude:
            continue
        if isinstance(value, (int, float)) and value <= 0:
            raise SkillError(f"{key} must be positive", stage="validate")