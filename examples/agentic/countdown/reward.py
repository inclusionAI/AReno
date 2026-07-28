"""Outcome reward for Countdown arithmetic trajectories."""

from __future__ import annotations

import json


def reward_fn(record) -> float:
    """Calculate reward based on how close the final answer is to the target.

    Reward structure:
    - 1.0: Exact match with target
    - 0.7: Within 10% of target
    - 0.3: Within 30% of target
    - 0.0-0.3: Linear decay for further distances
    """

    source = dict(record.source_record)
    target = float(source.get("target", 0))

    # Extract final answer from tool calls
    final_answer = None
    for call in record.tool_calls:
        if call.get("name") == "finish":
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return -1.0
            if isinstance(arguments, dict) and "answer" in arguments:
                try:
                    final_answer = float(arguments["answer"])
                except (ValueError, TypeError):
                    return -1.0
            break

    if final_answer is None:
        return 0.0  # No finish call made

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
        # Linear decay from 0.3 to 0.0 for distances 30% to 100%
        return max(0.0, 0.3 - (relative_diff - 0.3) * (0.3 / 0.7))
