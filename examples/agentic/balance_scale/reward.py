"""Reward function for the balance-scale agentic example."""

from __future__ import annotations

import json
from typing import Any

FULL_ANSWER_REWARD = 1.0
IDENTITY_ONLY_REWARD = 0.5
WRONG_REWARD = 0.0


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the answer tool call.

    Returns:
        1.0 — correct ball identity and weight direction.
        0.5 — correct ball identity only.
        0.0 — wrong answer or no answer tool call found.
    """

    source = record.source_record
    correct_index = int(source["odd_ball_index"])
    correct_direction = source["odd_ball_direction"]
    answer_index, answer_direction = _extract_answer(record)
    if answer_index is None:
        return WRONG_REWARD
    if answer_index != correct_index:
        return WRONG_REWARD
    if answer_direction == correct_direction:
        return FULL_ANSWER_REWARD
    return IDENTITY_ONLY_REWARD


def _extract_answer(record: Any) -> tuple[int | None, str | None]:
    """Find the last answer tool call and return (ball_index, direction)."""

    for call in reversed(record.tool_calls):
        name = call.get("name") if isinstance(call, dict) else None
        if name != "answer":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return (None, None)
        if not isinstance(arguments, dict):
            return (None, None)
        ball_index = arguments.get("ball_index")
        direction = arguments.get("direction")
        try:
            ball_index = int(ball_index)
        except (TypeError, ValueError):
            ball_index = None
        return (ball_index, direction)
    return (None, None)
