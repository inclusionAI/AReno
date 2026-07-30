"""Test reward function for verifying slow-reward-hook timing (issue #242).

This reward function deliberately introduces per-sample delays so that
the --reward-timing-enabled / --reward-slow-threshold-s / --reward-timeout-s
flags produce observable outliers and timeouts.

Usage:
    areno train --algo gspo \
      --reward-fn-path examples/math/timing_test_reward.py \
      --reward-timing-enabled \
      --reward-slow-threshold-s 0.3 \
      --reward-timeout-s 2.0 \
      ...
"""

from __future__ import annotations

import time


def reward_fn(record) -> float:
    """Score one completion with an artificial delay based on sample index.

    Samples with an even sample_index sleep 0.01s (fast).
    Samples with an odd sample_index sleep 0.5s (slow, will be flagged as outlier).
    Sample (prompt_index=2, sample_index=3) sleeps 3s (will trigger timeout if
    --reward-timeout-s < 3).
    """

    meta = record.metadata if hasattr(record, "metadata") else {}
    prompt_index = int(meta.get("prompt_index", -1))
    sample_index = int(meta.get("sample_index", -1))

    # Special case: this sample will timeout if timeout_s < 3
    if prompt_index == 2 and sample_index == 3:
        time.sleep(3.0)
        return 1.0

    # Odd samples are slow
    if sample_index % 2 == 1:
        time.sleep(0.5)
        return 0.5

    # Even samples are fast
    time.sleep(0.01)
    return 1.0 if "correct" in record.completion.lower() else 0.0
