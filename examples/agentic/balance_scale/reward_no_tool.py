"""Reward function for the balance-scale XML no-tool example."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402

FULL_ANSWER_REWARD = 1.0
IDENTITY_ONLY_REWARD = 0.5
WRONG_REWARD = 0.0


def reward_fn(record: Any) -> float:
    """Score one completion by extracting the final <answer> XML tag.

    Returns:
        1.0 — correct ball identity and weight direction.
        0.5 — correct ball identity only.
        0.0 — wrong answer or no answer tag found.
    """

    source = record.source_record
    correct_index = int(source["odd_ball_index"])
    correct_direction = source["odd_ball_direction"]
    parsed = game.parse_xml_answer(record.completion)
    if parsed is None:
        return WRONG_REWARD
    answer_index, answer_direction = parsed
    if answer_index != correct_index:
        return WRONG_REWARD
    if answer_direction == correct_direction:
        return FULL_ANSWER_REWARD
    return IDENTITY_ONLY_REWARD