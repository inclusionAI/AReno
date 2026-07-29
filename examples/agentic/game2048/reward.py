"""Reward function for the 2048 agentic example.

Replays the agent's move sequence on the seeded board and returns
the final merge score as the episode reward.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record) -> float:
    """Replay the agent's moves and return the final merge score."""

    source = record.source_record
    board = game.normalize_board(source["board"])
    seed = int(source["seed"])
    max_moves = int(source.get("max_moves", game.DEFAULT_MAX_MOVES))

    rng = random.Random(seed)
    total_merge_score = 0
    move_count = 0

    for direction in _extract_directions(record):
        if move_count >= max_moves:
            break
        board, score, valid, terminal = game.move(board, direction, rng)
        if valid:
            total_merge_score += score
        move_count += 1
        if terminal:
            break

    return float(total_merge_score)


def _extract_directions(record) -> list[str]:
    """Pull move directions from the agent's tool calls, in order."""

    directions = []
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "move":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict):
            direction = arguments.get("direction")
            if isinstance(direction, str) and direction.upper() in game.DIRECTIONS:
                directions.append(direction.upper())
    return directions