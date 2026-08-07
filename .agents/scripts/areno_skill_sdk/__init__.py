"""Shared runtime SDK for repository skill scripts.

A lightweight scaffolding library that extracts the repeated boilerplate
(argument parsing, JSON output, error handling) duplicated across the 24
scripts under ``.agents/skills/``. It is not a new framework and does not
replace any existing component — it only centralizes the scaffolding so
scripts keep just their business logic.

Public surface
--------------

* :func:`skill_main` — decorator taking over exception envelope + JSON output
  + exit code.
* :func:`build_parser`, :func:`validate_positive` — argument handling.
* :class:`Result`, :func:`exit_code` — result objects and exit codes.
* :func:`emit` — table/JSON rendering.
* :class:`SkillError`, :func:`envelope` — exception envelopes.
* :class:`ProgressEvent`, :class:`JsonLinesSink` — progress protocol.

Typical usage
-------------

.. code-block:: python

   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / "scripts"))
   from areno_skill_sdk import skill_main, build_parser, Result, SkillError

   @skill_main
   def main():
       parser = build_parser("Do something useful.")
       parser.add_argument("--count", type=int, required=True)
       args = parser.parse_args()
       if args.count <= 0:
           raise SkillError("count must be positive", stage="validate")
       return Result(ok=True, data={"count": args.count})

   if __name__ == "__main__":
       raise SystemExit(main())
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

from .args import build_parser, validate_positive
from .errors import SkillError, envelope
from .progress import JsonLinesSink, ProgressEvent, ProgressSink
from .render import emit
from .result import Result, exit_code

__all__ = [
    "skill_main",
    "build_parser",
    "validate_positive",
    "Result",
    "exit_code",
    "emit",
    "SkillError",
    "envelope",
    "ProgressEvent",
    "ProgressSink",
    "JsonLinesSink",
]


def skill_main(func: Callable[..., Any]) -> Callable[..., int]:
    """Decorate a skill script's ``main()``.

    Takes over exception envelope + JSON output + exit code. The wrapped
    function returns a :class:`Result` or a ``dict``; the decorator emits it
    and returns the exit code.

    Behavior:

    * Normal return -> ``emit(result)`` then ``exit_code(result)``.
    * :class:`SkillError` -> ``emit({"ok": False, "error": ..., "stage": ...})``
      then ``1``.
    * Any other :class:`Exception` -> ``emit(envelope(exc))`` then ``1``.

    This unifies the three hand-written patterns currently duplicated across
    the unmigrated scripts.
    """

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
        except Exception as exc:  # noqa: BLE001 - intentional top-level envelope
            emit(envelope(exc))
            return 1

    return wrapper
