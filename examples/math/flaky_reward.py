"""Flaky reward function for testing quarantine (#248).

Every 3rd sample raises ValueError to simulate a failing reward function.
Use with --quarantine-enabled to verify that:
  1. Training continues past the failing sample.
  2. quarantine.{pid}.jsonl is written with the failure record.
  3. Sensitive fields (prompt/completion) are redacted to hashes.
"""

from __future__ import annotations

_CALL_COUNT = 0


def reward_fn(record) -> float:
    global _CALL_COUNT
    _CALL_COUNT += 1
    if _CALL_COUNT % 3 == 0:
        raise ValueError(f"simulated failure on call #{_CALL_COUNT}")
    return 1.0
