"""Math verifier reward with NaN injection for testing non-finite detection.

Returns normal 0/1 rewards for the first few calls, injects NaN for a
configurable window, then recovers to normal scoring — so training can
demonstrate the full lifecycle: normal → detect → skip → recover.
"""

from __future__ import annotations

import os

from math_verify import parse, verify

# Global call counter — increments across all prompts in a single scoring pass.
_call_count = 0

# Inject NaN after this many calls (≈ step 2 with batch_size=4 × n_samples=4).
# Override via env var ARENO_NAN_INJECT_START.
_NAN_START = int(os.getenv("ARENO_NAN_INJECT_START", "33"))

# Stop injecting NaN after this many calls so training can recover.
# Override via env var ARENO_NAN_INJECT_END.
# With 16 calls/step and start=33: end=64 means NaN on steps 2-3, recovery at step 4.
_NAN_END = int(os.getenv("ARENO_NAN_INJECT_END", "64"))


def reward_fn(record) -> float:
    """Score one completion; inject NaN for a window, then recover."""

    global _call_count
    _call_count += 1

    if _NAN_START <= _call_count <= _NAN_END:
        return float("nan")

    solutions = record.answer
    if solutions is None:
        raise KeyError("math reward expects `record.answer`; use the math dataset loader to normalize raw rows")
    ground_truth = solutions[0] if isinstance(solutions, list) else solutions
    gt_parsed = parse(ground_truth)
    pred_parsed = parse(record.completion)
    try:
        return 1.0 if verify(gt_parsed, pred_parsed) else 0.0
    except Exception:
        return 0.0

