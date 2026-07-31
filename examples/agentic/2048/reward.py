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
        # Mild flat penalty: just below the worst tool-call episode floor (~+4).
        # The old baseline-linked penalty (~-215 vs +5) dominated the group
        # advantage normalisation and caused GSPO death spiral.
        sample_index = int(record.metadata.get("sample_index", 0))
        offset = game._PER_SAMPLE_OFFSETS[sample_index % len(game._PER_SAMPLE_OFFSETS)]
        penalty = -5.0 + offset
        game.logger.info(
            "2048 no-tool-call penalty id=%s baseline=%.1f penalty=%.3f",
            record_id,
            baseline_score,
            penalty,
        )
        return penalty

    moves = game.parse_moves(arguments)
    reward = game.score_moves(
        board,
        moves,
        seed=int(source["seed"]),
        baseline_score=baseline_score,
        trials=int(source["random_baseline"].get("trials", 8)),
        record_id=record_id,
        sample_index=int(record.metadata.get("sample_index", 0)),
    )
    # Format bonus: reward the model for outputting well-formed tool-call JSON.
    # This is orthogonal to game outcome — the model learns format first, then
    # strategy.  Without this, a 0.8B model has no RL signal telling it that
    # <tool_call>{"name":"choose_moves",...}</tool_call> is the right shape;
    # it only sees game-score feedback, which is too sparse for format learning.
    if arguments is not None:
        reward += game.FORMAT_BONUS
    if not moves:
        game.logger.warning(
            "2048 empty-moves id=%s arguments=%s",
            record_id,
            str(arguments)[:120] if arguments else "None",
        )
    return reward


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