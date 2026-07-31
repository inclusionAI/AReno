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
        if not isinstance(arguments, dict):
            return -1.0
        word = arguments.get("word") or arguments.get("guess")
        if not word:
            return -1.0
        guesses.append(word)

    if len(guesses) != len(set(map(str, guesses))):
        return -0.5

    return score_episode(
        source["secret"],
        guesses,
        max_guesses=int(source["max_guesses"]),
    )

