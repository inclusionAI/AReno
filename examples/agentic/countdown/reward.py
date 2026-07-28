"""Reward function for the Countdown tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the calculate tool call."""

    source = record.source_record
    numbers = game.normalize_numbers(source["numbers"])
    target = int(source["target"])
    a, b, op = _tool_call(record)
    return game.score_move(numbers, target, a, b, op)


def _tool_call(record: Any) -> tuple[int | None, int | None, str | None]:
    """Extract (a, b, op) from the model's calculate tool call."""

    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "calculate":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None, None, None
        if isinstance(arguments, dict):
            a = arguments.get("a")
            b = arguments.get("b")
            op = arguments.get("op")
            try:
                a = int(a) if a is not None else None
            except (TypeError, ValueError):
                a = None
            try:
                b = int(b) if b is not None else None
            except (TypeError, ValueError):
                b = None
            if op not in game.OPERATIONS:
                op = None
            return a, b, op
    return None, None, None