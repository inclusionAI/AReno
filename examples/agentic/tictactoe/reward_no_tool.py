"""Reward function for the Tic-Tac-Toe XML no-tool example."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the final XML move tag."""

    source = record.source_record
    board = game.normalize_board(source["board"])
    return game.score_move(board, game.parse_xml_move(record.completion))
