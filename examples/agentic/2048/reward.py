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
    moves = game.parse_moves(_tool_moves(record))
    if not moves:
        moves = game.parse_moves(getattr(record, "completion", None))
    return game.score_moves(
        board,
        moves,
        seed=int(source["seed"]),
        baseline_score=float(source["random_baseline"]["score"]),
        record_id=source.get("id", "?"),
    )


def _tool_moves(record: Any) -> Any:
    """Return the ``moves`` payload from a ``choose_moves`` tool call, if any."""

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
        return arguments
    return None