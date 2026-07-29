"""Reward function for the 6x6 Othello tool-call example."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the choose_move tool call.

    Uses :func:`game.parse_tool_move_raw` (keeps out-of-range coordinates) so
    that :func:`game.score_move` can apply its tiered rewards: a tool call with
    a bad coordinate is a different (less negative) tier than no tool call at
    all, which prevents the policy from collapsing to "never call the tool".
    """

    source = record.source_record
    board = game.normalize_board(source["board"])
    player = source.get("player", "B")
    return game.score_move(board, game.parse_tool_move_raw(record.tool_calls), player)