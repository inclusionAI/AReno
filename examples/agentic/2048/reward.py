"""Reward function for the 2048 tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """Score one completion.

    A valid ``choose_moves`` tool call is REQUIRED. Its parsed moves drive the
    episode reward; if the turn produced no ``choose_moves`` call at all, return
    a penalty worse than any legitimate episode, so the policy cannot coast on
    plain-text directions parsed out of a prose response. The old text fallback
    let verbose non-tool responses out-score real tool calls, and the RL
    actively abandoned tool use (tool_calls collapsed 4/4 -> 0/4 over 11 steps).
    """

    source = record.source_record
    board = game.normalize_board(source["board"])
    baseline_score = float(source["random_baseline"]["score"])
    record_id = source.get("id", "?")

    arguments, called = _choose_moves_arguments(record)
    if not called:
        # Worse than any legitimate episode (score 0 with every move invalid),
        # so emitting a tool call always beats falling back to prose.
        penalty = -(baseline_score + game.INVALID_PENALTY * game.DEFAULT_EPISODE_CAP + 1.0)
        game.logger.info(
            "2048 no-tool-call penalty id=%s baseline=%.1f penalty=%.3f",
            record_id,
            baseline_score,
            penalty,
        )
        return penalty

    moves = game.parse_moves(arguments)
    return game.score_moves(
        board,
        moves,
        seed=int(source["seed"]),
        baseline_score=baseline_score,
        record_id=record_id,
    )


def _choose_moves_arguments(record: Any) -> tuple[Any, bool]:
    """Return ``(arguments, called)`` for the ``choose_moves`` tool call.

    ``called`` is False only when the turn made no ``choose_moves`` call at all
    (the case we penalize). A call with unparseable arguments still counts as
    ``called=True`` with ``arguments=None``; the episode then replays zero
    moves, scoring below a real plan but above the no-call penalty.
    """

    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "choose_moves":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        return arguments, True
    return None, False