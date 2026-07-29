"""Reward function for the Wordle tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


# Reward constants - designed to encourage tool calling
REWARD_TRY = 0.1           # Bonus for simply making a tool call (encourage trying)
REWARD_NO_TOOL = -1.0      # Heavy penalty for not calling tool (worst option)
REWARD_INVALID = -0.3      # Lighter penalty for invalid word (tried but failed)
REWARD_NO_MATCH = 0.0      # Zero reward for valid word but no matches
REWARD_PARTIAL = 0.3       # Base reward for partial progress
REWARD_CORRECT = 1.0       # Base reward for correct guess


def reward_fn(record: Any) -> float:
    """
    Score one completion by extracting the guess_word tool call.

    Reward scheme (designed to prevent "reward hacking" - model giving up on tool calls):
    - +1.0  -> +1.5: Correctly guessed the word (bonus for fewer guesses)
    - +0.3  -> +0.9: Partial progress (some letters in correct position)
    - +0.1  -> +0.2: Valid word, letters present but wrong position
    -  0.0:       Valid word but no letters match (neutral)
    - -0.3:       Invalid word but tried (tolerable)
    - +0.1:       Simply called the tool (encouragement)
    - -1.0:       No tool call (heaviest penalty - worst option)

    Key insight: Model should prefer calling tools (even if wrong) over not calling.
    """
    source = record.source_record
    target = source.get("target", "")
    current_game = source.get("game", {})

    guess = _tool_guess(record)
    if guess is None:
        # Model didn't call guess_word tool - HEAVIEST penalty
        # This must be the worst option to prevent "reward hacking"
        return REWARD_NO_TOOL

    # Model called the tool - give encouragement bonus
    base_reward = REWARD_TRY

    # Check if guess is valid
    if not game.is_valid_word(guess):
        # Tried but failed - lighter penalty than not trying at all
        return base_reward + REWARD_INVALID

    # Check if guess is correct
    if guess == target:
        num_guesses = len(current_game.get("guesses", [])) + 1
        efficiency_bonus = (game.MAX_GUESSES - num_guesses) / game.MAX_GUESSES * 0.5
        return REWARD_CORRECT + efficiency_bonus

    # Check for partial progress
    try:
        feedback = game.check_guess(guess, target)
        exact_count = sum(1 for f in feedback if f == game.LetterStatus.EXACT)
        present_count = sum(1 for f in feedback if f == game.LetterStatus.PRESENT)

        if exact_count > 0:
            return base_reward + REWARD_PARTIAL + (exact_count * 0.2) + (present_count * 0.05)
        elif present_count > 0:
            return base_reward + 0.1 + (present_count * 0.05)
        else:
            return base_reward + REWARD_NO_MATCH  # Valid word, no matches = neutral
    except Exception:
        return base_reward + REWARD_INVALID


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