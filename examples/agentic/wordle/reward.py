"""Reward function for the Wordle tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


def reward_fn(record: Any) -> float:
    """
    Score one completion by extracting the guess_word tool call.

    Reward scheme:
    - +1.0: Correctly guessed the word
    - +0.5: Made progress (at least one letter in correct position)
    - +0.2: Valid word but no progress
    - -1.0: Invalid word (not in word list)
    - 0.0: Game lost (exhausted all guesses)
    """
    source = record.source_record
    target = source.get("target", "")
    current_game = source.get("game", {})

    guess = _tool_guess(record)
    if guess is None:
        # No valid tool call
        return -1.0

    # Check if guess is valid
    if not game.is_valid_word(guess):
        return -1.0

    # Check if guess is correct
    if guess == target:
        # Bonus for winning
        num_guesses = len(current_game.get("guesses", [])) + 1
        efficiency_bonus = (game.MAX_GUESSES - num_guesses) / game.MAX_GUESSES * 0.5
        return 1.0 + efficiency_bonus

    # Check for partial progress (at least one letter in correct position)
    try:
        feedback = game.check_guess(guess, target)
        exact_count = sum(1 for f in feedback if f == game.LetterStatus.EXACT)
        present_count = sum(1 for f in feedback if f == game.LetterStatus.PRESENT)

        if exact_count > 0:
            # Made progress towards solution
            return 0.5 + (exact_count * 0.1) + (present_count * 0.05)
        elif present_count > 0:
            # Some letters are in the word but wrong position
            return 0.2 + (present_count * 0.05)
        else:
            # Valid word but no letters in common
            return 0.0
    except Exception:
        return -1.0


def _tool_guess(record: Any) -> str | None:
    """Extract the guessed word from tool call."""
    for call in record.tool_calls:
        name = call.get("name") if isinstance(call, dict) else None
        if name != "guess_word":
            continue
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if isinstance(arguments, dict):
            word = arguments.get("word")
            if isinstance(word, str) and len(word) == 5 and word.isalpha():
                return word.lower()
    return None