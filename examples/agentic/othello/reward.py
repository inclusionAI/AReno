"""Reward function for the 6x6 Othello tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the choose_move tool call."""

    source = record.source_record
    board = game.normalize_board(source["board"])
    player = source.get("player", "B")
    row, col = _tool_move(record)
    return game.score_move(board, row, col, player)


def _tool_move(record: Any) -> tuple[int | None, int | None]:
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "choose_move":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None, None
        if isinstance(arguments, dict):
            row = arguments.get("row")
            col = arguments.get("col")
            try:
                return int(row), int(col)
            except (TypeError, ValueError):
                return None, None
    return None, None
