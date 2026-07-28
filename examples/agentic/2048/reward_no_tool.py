"""Reward function for the 2048 XML no-tool example."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the final XML moves tag."""

    source = record.source_record
    board = game.normalize_board(source["board"])
    moves = game.parse_xml_moves(record.completion)
    return game.score_moves(
        board,
        moves,
        seed=int(source["seed"]),
        baseline_score=float(source["random_baseline"]["score"]),
        record_id=source.get("id", "?"),
    )