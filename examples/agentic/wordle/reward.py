"""Reward function for the Wordle tool-call example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


# Reward constants - designed for stable GSPO training
# Key principles:
# 1. ALL rewards in [-1.0, +1.0] to prevent gradient explosion
# 2. Spread rewards so batch advantage is non-zero even when all samples fail
# 3. IMPORTANT: when all samples get same reward, advantage=0, loss=0, grad=0
#    So we add a small random jitter to break ties (using guess hash)
REWARD_NO_TOOL = -1.0      # Worst: didn't call tool
REWARD_INVALID = -0.5      # Called tool but invalid word
REWARD_NO_MATCH = 0.0      # Valid word, no letters match
REWARD_PRESENT = 0.2       # Some letters present (wrong position)
REWARD_EXACT = 0.5         # Some letters in exact position
REWARD_CORRECT = 1.0       # Correct guess


def reward_fn(record: Any) -> float:
    """
    Score one completion by extracting the guess_word tool call.

    Reward scheme (bounded [-1.0, +1.0], spread for non-zero advantage):
    - +1.0:       Correctly guessed the word
    - +0.5+:      Partial progress (exact letter matches, scaled by count)
    - +0.2+:      Letters present but wrong position
    -  0.0:       Valid word but no letters match
    - -0.5:       Invalid word (called tool but bad word)
    - -1.0:       No tool call (worst option)

    Key design: rewards spread across [-1, 1] so that within a batch,
    advantages (reward - mean) are non-zero, preventing GSPO loss
    from collapsing to 0 and gradients from vanishing.

    Fallback jitter: when model doesn't call tool, all samples get -1.0,
    advantage becomes 0, loss becomes 0, gradient vanishes. We add a
    tiny deterministic jitter based on record hash to break ties.
    """
    import hashlib

    source = record.source_record
    target = source.get("target", "")
    current_game = source.get("game", {})

    guess = _tool_guess(record)
    if guess is None:
        # Model didn't call guess_word tool
        # Add tiny jitter to prevent all-zero advantage when entire batch fails
        # Jitter is deterministic per record (based on record id) and small [-0.01, 0.01]
        record_id = source.get("id", str(id(record)))
        jitter_hash = int(hashlib.md5(str(record_id).encode()).hexdigest()[:8], 16)
        jitter = ((jitter_hash % 200) - 100) / 10000.0  # [-0.01, 0.01]
        return REWARD_NO_TOOL + jitter

    # Model called the tool - no base bonus (avoid uniform rewards)
    # Different outcomes get different rewards to maintain batch advantage variance

    # Check if guess is valid
    if not game.is_valid_word(guess):
        # Called tool but invalid word: negative reward
        return REWARD_INVALID

    # Check if guess is correct
    if guess == target:
        return REWARD_CORRECT  # Fixed 1.0, no efficiency bonus to keep bounded

    # Check for partial progress
    try:
        feedback = game.check_guess(guess, target)
        exact_count = sum(1 for f in feedback if f == game.LetterStatus.EXACT)
        present_count = sum(1 for f in feedback if f == game.LetterStatus.PRESENT)

        if exact_count > 0:
            # Exact matches: base 0.5 + 0.1 per exact + 0.02 per present
            return REWARD_EXACT + (exact_count * 0.1) + (present_count * 0.02)
        elif present_count > 0:
            # Present only: base 0.2 + 0.05 per present
            return REWARD_PRESENT + (present_count * 0.05)
        else:
            return REWARD_NO_MATCH  # Valid word, no matches = neutral 0.0
    except Exception:
        return REWARD_INVALID


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