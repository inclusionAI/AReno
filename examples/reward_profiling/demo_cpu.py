"""Minimal deterministic example for reward profiling (issue #242).

Run without GPU, network, or sandbox:

    python3 examples/reward_profiling/demo_cpu.py

Demonstrates the successful path (slow-sample detection) and one
boundary/failure path (timeout enforcement with error identifiers).
"""

from __future__ import annotations

import time
import unittest


def reward_fn(record):
    """Deterministic reward: 1.0 if 'answer' appears in the completion.

    The second sample (sample_index=1) sleeps 50 ms to trigger the
    slow-sample threshold.
    """

    if record.metadata.get("sample_index") == 1:
        time.sleep(0.05)
    return 1.0 if "answer" in record.completion else 0.0


def main():
    from areno.api.reward_profiler import RewardProfiler, RewardTimeoutError
    from areno.api.rewards import RewardRecord

    records = [
        RewardRecord(
            prompt="q",
            completion=f"answer{i}",
            metadata={"prompt_index": 0, "sample_index": i},
        )
        for i in range(4)
    ]

    # --- Successful path: slow-sample detection ---
    profiler = RewardProfiler(reward_fn, enabled=True, slow_threshold_s=0.03, batch_timeout_s=1.0)
    rewards, profile = profiler.score_batch(records)
    print(f"rewards: {rewards}")
    print(f"slow_samples: {[(s.sample_index, round(s.duration_s, 4)) for s in profile.slow_samples]}")
    assert any(s.sample_index == 1 for s in profile.slow_samples), "Expected sample_index=1 to be slow"

    # --- Boundary/failure path: timeout enforcement ---
    # Non-positive config must raise ValueError with a clear message.
    tc = unittest.TestCase()
    try:
        RewardProfiler(reward_fn, enabled=True, slow_threshold_s=-1)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        tc.assertRegex(str(e), r"reward_slow_threshold_s must be > 0")
        print(f"Config validation OK: {e}")

    # A tiny batch timeout forces RewardTimeoutError with hook + sample identifiers.
    slow_profiler = RewardProfiler(reward_fn, enabled=True, batch_timeout_s=0.0001)
    try:
        slow_profiler.score_batch(records)
        raise AssertionError("Should have raised RewardTimeoutError")
    except RewardTimeoutError as e:
        assert e.hook == "reward_fn"
        assert e.sample_index is not None
        print(f"Timeout enforcement OK: hook={e.hook} sample={e.sample_index} elapsed={e.elapsed:.4f}s")

    print("All paths demonstrated successfully.")


if __name__ == "__main__":
    main()