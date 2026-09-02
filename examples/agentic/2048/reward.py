"""Reward function for the 2048 tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the choose_moves tool call."""

    source = record.source_record
    board = game.normalize_board(source["board"])
    return game.score_episode_moves(board, _tool_moves(record), seed=int(source["seed"]))


def _tool_moves(record: Any) -> list[str] | None:
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "choose_moves":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            return game.parse_moves(arguments)
    return None