"""Reward function for the Qwen3.5-VL tic-tac-toe image example."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the selected square."""

    source = record.source_record
    board = game.normalize_board(source["board"])
    return game.score_move(board, _text_square(record.completion))


def _text_square(text: str) -> int | None:
    lowered = str(text).lower()
    match = re.search(r"\bsquare\s*([1-9])\b", lowered)
    if match:
        return int(match.group(1))
    for square in range(1, 10):
        if game.square_name(square) in lowered:
            return square
    return None
