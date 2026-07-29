"""Reward function for the maze agentic example.

Replays the agent's move sequence against the initial maze state and
scores the outcome.  Supports two shaping modes selectable via the
``reward_mode`` field in ``source_record``:

- ``"bfs"`` (default): BFS closest-approach distance shaping.
- ``"pbrs"``: Potential-Based Reward Shaping with gamma=0.95.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one episode by extracting and replaying move tool calls."""

    source = dict(record.source_record)
    directions = _extract_moves(record)
    results = _replay_episode(source, directions)
    shortest = source.get("shortest_path_len", 0)
    mode = source.get("reward_mode", "bfs")

    if mode == "pbrs":
        return game.score_episode_pbrs(results, shortest, source)
    return game.score_episode(results, shortest)


def _extract_moves(record: Any) -> list[str]:
    """Extract the sequence of directions from ``move`` tool calls."""

    directions: list[str] = []
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
            if isinstance(direction, str):
                directions.append(direction)
    return directions


def _replay_episode(source: dict, directions: list[str]) -> list[game.MoveResult]:
    """Replay *directions* against the initial maze state from *source*."""

    state = game.deserialize_maze(source)
    results: list[game.MoveResult] = []
    for direction in directions:
        if state.steps_taken >= state.max_steps:
            break
        result = game.apply_move(state, direction)
        results.append(result)
        if result.terminal:
            break
        state = result.state
    return results