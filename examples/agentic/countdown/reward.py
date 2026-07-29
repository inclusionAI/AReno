"""Outcome reward for Countdown arithmetic trajectories.

The reward function is the single signal that GSPO uses to improve the policy.
For Countdown, a natural signal is "how close was the answer the model
submitted to the target number?" -- we reward exact matches most, then taper
off as the relative error grows.

The submitted answer is extracted from the ``finish`` tool call's ``answer``
argument; if the model never called ``finish`` (or called it with malformed
arguments), the reward is 0 or a small negative penalty.

Reward structure:
    1.0  -- exact match with target
    0.7  -- within 10% of target (relative error)
    0.3  -- within 30% of target
    0.0-0.3 -- linear decay for relative errors in [30%, 100%]
    0.0  -- no finish call made
    -1.0 -- finish call present but arguments could not be parsed
"""

from __future__ import annotations

import json


def reward_fn(record) -> float:
    """Score one Countdown trajectory by comparing the final answer to target.

    Args:
        record: AReno's trajectory record. We use two fields:
            - ``record.source_record``: the original dataset row (carries
              ``target``).
            - ``record.tool_calls``: list of tool calls the model made during
              the episode. We look for the last ``finish`` call and extract
              its ``answer`` argument.

    Returns:
        A float reward in [-1.0, 1.0]. See module docstring for the schedule.
    """
    source = dict(record.source_record)
    target = float(source.get("target", 0))

    # Walk the tool calls and find the finish call. AReno exposes tool_calls
    # as a list of dicts; the ``arguments`` field may be a JSON string (as
    # returned by the model) or already-parsed dict, so we handle both.
    final_answer = None
    for call in record.tool_calls:
        if call.get("name") == "finish":
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    # Malformed JSON in the finish call -- penalize so the
                    # model learns to emit valid tool-call arguments.
                    return -1.0
            if isinstance(arguments, dict) and "answer" in arguments:
                try:
                    final_answer = float(arguments["answer"])
                except (ValueError, TypeError):
                    return -1.0
            break

    if final_answer is None:
        # The model never called finish; no information about correctness.
        return 0.0

    # Edge case: target == 0 can't be scored by relative error, so fall back
    # to an exact-match check.
    if target == 0:
        return 1.0 if final_answer == 0 else 0.0

    diff = abs(final_answer - target)
    relative_diff = diff / target

    if diff == 0:
        return 1.0  # Exact match
    elif relative_diff <= 0.1:
        return 0.7  # Within 10%
    elif relative_diff <= 0.3:
        return 0.3  # Within 30%
    else:
        # Linear decay from 0.3 to 0.0 for relative errors in [30%, 100%].
        # Beyond 100% the clamp via max(0.0, ...) yields 0 reward.
        return max(0.0, 0.3 - (relative_diff - 0.3) * (0.3 / 0.7))