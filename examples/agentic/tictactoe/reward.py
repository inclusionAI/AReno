"""Reward function for the Tic-Tac-Toe tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the choose_square tool call."""

    source = record.source_record
    board = game.normalize_board(source["board"])
    return game.score_move(board, _tool_square(record))


def _tool_square(record: Any) -> int | None:
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "choose_square":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            square = arguments.get("square")
            try:
                return int(square)
            except (TypeError, ValueError):
                return None
    return None
