"""Outcome and process reward for Wordle trajectories."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_episode  # noqa: E402


def reward_fn(record) -> float:
    """Reward valid, non-repeated deduction and efficient success."""

    source = dict(record.source_record)
    guesses = []
    for call in record.tool_calls:
        if call.get("name") != "guess_word":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return -1.0
        if not isinstance(arguments, dict) or "word" not in arguments:
            return -1.0
        guesses.append(arguments["word"])

    if not guesses:
        return -1.0

    has_duplicates = len(guesses) != len(set(map(str, guesses)))

    # Compute base reward from score_episode, then penalize duplicates.
    # This preserves reward diversity even when some guesses repeat.
    base = score_episode(
        source["secret"],
        guesses,
        max_guesses=int(source["max_guesses"]),
    )
    if has_duplicates:
        # Deduct for repetition but keep the score informative: multiply
        # the base by a penalty factor and subtract a small fixed cost.
        return -0.3 + 0.6 * base
    return base

