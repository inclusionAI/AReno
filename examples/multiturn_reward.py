"""Minimal reward function for multi-turn conversation GSPO testing.

Reads the last assistant message from the record's `messages` field and
returns a simple length-based reward. No external dependencies required.
"""

from __future__ import annotations


def reward_fn(record: dict) -> float:
    """Return a reward based on the last assistant response length."""

    messages = record.get("messages", [])
    last_assistant = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "") or ""
            break
    # Simple: reward longer responses (capped) to give GSPO some signal
    return min(len(last_assistant) / 100.0, 1.0)