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

    Reward scheme (tool-based):
    - +1.0  -> +1.5: Correctly guessed the word (bonus for fewer guesses)
    - +0.3  -> +0.9: Partial progress (some letters in correct position)
    - +0.1  -> +0.2: Valid word, letters present but wrong position
    -  0.0:       Valid word but no letters match
    - -0.5:       No valid tool call or invalid word
    """
    source = record.source_record
    target = source.get("target", "")
    current_game = source.get("game", {})

    guess = _tool_guess(record)
    if guess is None:
        # Model didn't call guess_word tool
        return -0.5

    # Check if guess is valid
    if not game.is_valid_word(guess):
        return -0.5

    # Check if guess is correct
    if guess == target:
        num_guesses = len(current_game.get("guesses", [])) + 1
        efficiency_bonus = (game.MAX_GUESSES - num_guesses) / game.MAX_GUESSES * 0.5
        return 1.0 + efficiency_bonus

    # Check for partial progress
    try:
        feedback = game.check_guess(guess, target)
        exact_count = sum(1 for f in feedback if f == game.LetterStatus.EXACT)
        present_count = sum(1 for f in feedback if f == game.LetterStatus.PRESENT)

        if exact_count > 0:
            return 0.3 + (exact_count * 0.2) + (present_count * 0.05)
        elif present_count > 0:
            return 0.1 + (present_count * 0.05)
        else:
            return 0.0
    except Exception:
        return -0.5


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