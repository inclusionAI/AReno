"""Reward function for the Wordle tool-call example."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import game  # noqa: E402


# Reward constants - designed for stable GSPO training.
#
# Key principles:
# 1. ALL rewards in [-1.0, +1.0] to prevent gradient explosion.
# 2. Spread rewards so batch advantage is non-zero even when all samples fail.
# 3. Anti-collapse jitter: when all samples get the same base reward,
#    advantage = 0, loss = 0, gradient = 0, and the model can never recover.
#    We add a deterministic jitter based on unique sample identity to every
#    reward, ensuring group std is always non-zero.
REWARD_NO_TOOL = -1.0      # Worst: didn't call tool
REWARD_INVALID = -0.5      # Called tool but invalid word
REWARD_NO_MATCH = 0.0      # Valid word, no letters match
REWARD_PRESENT = 0.2       # Some letters present (wrong position)
REWARD_EXACT = 0.5         # Some letters in exact position
REWARD_CORRECT = 1.0       # Correct guess


def _per_sample_jitter(record: Any) -> float:
    """Deterministic jitter in [-0.1, +0.1] based on sample identity."""
    source = record.source_record
    meta = getattr(record, "metadata", {}) or {}
    p_idx = meta.get("prompt_index", 0)
    s_idx = meta.get("sample_index", 0)
    seed = f"{p_idx}:{s_idx}:{source.get('id', '')}"
    h = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    return ((h % 1000) - 500) / 5000.0


def reward_fn(record: Any) -> float:
    """
    Score one completion by extracting the guess_word tool call.

    Reward scheme (bounded [-1.0, +1.0], spread for non-zero advantage):

    ===================  ===============================================
    Outcome              Reward
    ===================  ===============================================
    Correct guess        +1.0
    Partial exact match  +0.5 + 0.1 per exact + 0.02 per present
    Letters present      +0.2 + 0.05 per present
    Valid, no match       0.0
    Invalid word         -0.5
    No tool call         -1.0
    ===================  ===============================================

    A small per-sample jitter ([-0.1, +0.1]) is added to every reward to
    prevent GSPO advantage collapsing to zero when all samples in a group
    produce the same outcome.
    """
    source = record.source_record
    target = source.get("target", "")
    jitter = _per_sample_jitter(record)

    guess = _tool_guess(record)
    if guess is None:
        return REWARD_NO_TOOL + jitter

    # Validate the guessed word
    if not game.is_valid_word(guess):
        return REWARD_INVALID + jitter

    # Correct guess
    if guess == target:
        return REWARD_CORRECT + jitter

    # Partial progress
    try:
        feedback = game.check_guess(guess, target)
        exact_count = sum(1 for f in feedback if f == game.LetterStatus.EXACT)
        present_count = sum(1 for f in feedback if f == game.LetterStatus.PRESENT)

        if exact_count > 0:
            return REWARD_EXACT + (exact_count * 0.1) + (present_count * 0.02) + jitter
        elif present_count > 0:
            return REWARD_PRESENT + (present_count * 0.05) + jitter
        else:
            return REWARD_NO_MATCH + jitter
    except Exception:
        return REWARD_INVALID + jitter


def _tool_guess(record: Any) -> str | None:
    """Extract the guessed word from the tool call.

    Returns the lowercase word, or ``None`` if no valid ``guess_word``
    tool call is found.
    """
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
            # Accept any alphabetic word; length validation happens via
            # game.is_valid_word or game.normalize_word downstream.
            if isinstance(word, str) and word.isalpha():
                return word.lower()
    return None