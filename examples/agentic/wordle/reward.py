"""Reward function for the Wordle tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


# Reward constants - bounded and symmetric to prevent gradient explosion
# Key principles:
# 1. ALL rewards in [-1.0, +1.0] to keep advantages bounded
# 2. Tool-call bonus is additive but capped
# 3. Penalty for not calling tool is -1.0 (worst, but not extreme)
REWARD_NO_TOOL = -1.0      # Worst option: didn't call tool
REWARD_TRY = 0.1           # Bonus for calling tool (encouragement)
REWARD_INVALID = 0.0       # Invalid word: zero reward (tried, no penalty)
REWARD_NO_MATCH = 0.1      # Valid word, no matches: small positive
REWARD_PARTIAL = 0.3       # Base for partial progress (exact matches)
REWARD_CORRECT = 1.0       # Correct guess (capped at 1.0, no efficiency bonus)


def reward_fn(record: Any) -> float:
    """
    Score one completion by extracting the guess_word tool call.

    Reward scheme (bounded [-1.0, +1.0] to prevent gradient explosion):
    - +1.0:       Correctly guessed the word
    - +0.3~+0.7:  Partial progress (exact letter matches)
    - +0.2:       Valid word, some letters present
    - +0.1:       Valid word but no matches, or invalid word
    - -1.0:       No tool call (worst option)

    Key design: rewards are bounded to [-1, 1] so that advantages
    (reward - mean) stay small, preventing grad_norm spikes that
    destroy model policy after checkpoint saves.
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
        return REWARD_CORRECT  # Fixed 1.0, no efficiency bonus to keep bounded

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