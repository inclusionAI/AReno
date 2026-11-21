"""Simple reward function for the smoke tutorial.

This reward function demonstrates the minimal reward contract for AReno.
It assigns rewards based on response length and quality heuristics.
"""

from __future__ import annotations


def reward_fn(record) -> float:
    """Score a completion based on simple heuristics.

    Reward rules:
    - 0.0: Empty or too short response (< 10 chars)
    - 0.3: Very short but non-empty (10-50 chars)
    - 0.6: Medium length (50-200 chars)
    - 1.0: Good length (> 200 chars) and contains reasoning markers
    """

    completion = record.completion.strip()

    if not completion:
        return 0.0

    length = len(completion)

    # Too short
    if length < 10:
        return 0.0

    # Very short
    if length < 50:
        return 0.3

    # Medium length
    if length < 200:
        return 0.6

    # Good length - bonus for reasoning markers
    reasoning_markers = ["because", "therefore", "since", "so", "thus", "hence"]
    has_reasoning = any(marker in completion.lower() for marker in reasoning_markers)

    return 1.0 if has_reasoning else 0.8
