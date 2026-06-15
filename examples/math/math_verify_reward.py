"""Math verifier reward: 1.0 when `math_verify` accepts one completion."""

from __future__ import annotations

from math_verify import parse, verify


def reward_fn(record) -> float:
    """Score one completion by verifying it against the first math solution."""

    solutions = record.answer
    if solutions is None:
        raise KeyError("math reward expects `record.answer`; use the math dataset loader to normalize raw rows")
    ground_truth = solutions[0] if isinstance(solutions, list) else solutions
    gt_parsed = parse(ground_truth)
    pred_parsed = parse(record.completion)
    try:
        # Binary reward: 1.0 if the symbolic comparison succeeds, else 0.
        return 1.0 if verify(gt_parsed, pred_parsed) else 0.0
    except Exception:
        # `verify` is intentionally tolerant; any unexpected failure is
        # treated as an incorrect answer rather than crashing the trainer.
        return 0.0
