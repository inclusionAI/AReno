"""Outcome and process reward for Wordle trajectories."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import score_episode  # noqa: E402

# Penalty for not calling the tool at all — must be strictly worse than any
# outcome where the model at least attempted a guess_word call, so that
# "learning to call the tool" produces a positive advantage gradient.
NO_TOOL_CALL_PENALTY = -1.0


def _per_sample_jitter(record) -> float:
    """Deterministic noise in [-0.1, +0.1] keyed on sample identity.

    Prevents all samples in a batch from sharing the same reward value,
    which would zero out group-normalised advantages.  The amplitude
    (±0.1) is smaller than the minimum reward tier gap (0.2) so it
    never flips the relative ordering of qualitatively different behaviours.
    """

    meta = record.metadata or {}
    seed = f"{meta.get('prompt_index', 0)}:{meta.get('sample_index', 0)}:{record.source_record.get('id', '')}"
    h = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    return ((h % 1000) - 500) / 5000.0


def reward_fn(record) -> float:
    """Reward valid, non-repeated deduction and efficient success.

    Layered reward design:
      +1.0  (approx)  solved efficiently
      +0.1*info        partial letter matches (intermediate gradient signal)
      -0.3             called tool but no valid guesses
      -0.5             repeated guesses
      -1.0             invalid guess arguments
      -1.0             **never called guess_word** (worst — must learn tool use first)

    A small deterministic jitter is added on top so that same-tier samples
    in a batch still receive slightly different rewards, avoiding zero
    advantage when all samples behave identically (e.g. all skip the tool).
    """

    source = dict(record.source_record)

    # Phase 1: extract guess_word tool calls.
    guess_word_calls = [
        call for call in record.tool_calls if call.get("name") == "guess_word"
    ]

    # Phase 2: if the model never called guess_word, apply the heaviest
    # penalty.  This is the key difference from the old design: "did not
    # call the tool" is strictly worse than "called the tool but guessed
    # wrong", giving RL a gradient to learn tool use itself.
    if not guess_word_calls:
        return NO_TOOL_CALL_PENALTY + _per_sample_jitter(record)

    # Phase 3: parse arguments from each call.
    guesses = []
    for call in guess_word_calls:
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return -1.0 + _per_sample_jitter(record)
        if not isinstance(arguments, dict):
            return -1.0 + _per_sample_jitter(record)
        word = arguments.get("word") or arguments.get("guess")
        if not word:
            return -1.0 + _per_sample_jitter(record)
        guesses.append(word)

    # Phase 4: penalise repeated guesses.
    if len(guesses) != len(set(map(str, guesses))):
        return -0.5 + _per_sample_jitter(record)

    # Phase 5: score the episode with the existing layered function.
    base = score_episode(
        source["secret"],
        guesses,
        max_guesses=int(source["max_guesses"]),
    )
    return base + _per_sample_jitter(record)

