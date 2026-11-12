"""Math verifier reward: 1.0 when `math_verify` accepts one completion."""

from __future__ import annotations

import threading

from math_verify import parse, verify


def _parse(expr: str):
    """Parse with math_verify's timeout only where signal.alarm works.

    ``parse`` and ``verify`` enforce their timeouts via ``signal.alarm``,
    which Python only allows in the main thread; threaded scorers (async
    rollout producers, ``parallel_workers`` pools) must disable them.
    """

    if threading.current_thread() is threading.main_thread():
        return parse(expr)
    return parse(expr, parsing_timeout=None)


def _verify(gt_parsed, pred_parsed) -> bool:
    """Compare parsed results with math_verify's timeout only on the main thread."""

    if threading.current_thread() is threading.main_thread():
        return bool(verify(gt_parsed, pred_parsed))
    return bool(verify(gt_parsed, pred_parsed, timeout_seconds=None))


def reward_fn(record) -> float:
    """Score one completion by verifying it against the first math solution."""

    solutions = record.answer
    if solutions is None:
        raise KeyError("math reward expects `record.answer`; use the math dataset loader to normalize raw rows")
    ground_truth = solutions[0] if isinstance(solutions, list) else solutions
    gt_parsed = _parse(ground_truth)
    pred_parsed = _parse(record.completion)
    try:
        # Binary reward: 1.0 if the symbolic comparison succeeds, else 0.
        return 1.0 if _verify(gt_parsed, pred_parsed) else 0.0
    except Exception:
        # `verify` is intentionally tolerant; any unexpected failure is
        # treated as an incorrect answer rather than crashing the trainer.
        return 0.0
