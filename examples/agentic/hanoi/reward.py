"""Outcome and efficiency reward for Towers of Hanoi trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record) -> float:
    """Score one episode by replaying the emitted ``move`` tool calls.

    The reward is shared between completion and a small efficiency component
    relative to the known optimum. Any illegal move terminates the replay and
    scores 0.0 exactly.
    """

    source = dict(record.source_record)
    n = int(source["n"])
    moves = _extract_moves(record.tool_calls)
    return float(game.score_episode(n, moves)["reward"])


def _extract_moves(tool_calls) -> list[tuple[object, object]]:
    moves: list[tuple[object, object]] = []
    for call in tool_calls:
        if call.get("name") != "move":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                moves.append((None, None))
                continue
        if not isinstance(arguments, dict):
            moves.append((None, None))
            continue
        moves.append((arguments.get("source"), arguments.get("target")))
    return moves
