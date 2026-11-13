"""Held-out evaluation helpers shared by trainers.

Scheduling and metric aggregation live here as backend-free pure functions: no
torch, no rollout engine, no model weights. Any trainer that reports held-out
metrics can reuse them, and all of them are unit-testable on CPU.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def benchmark_due(
    step: int,
    *,
    eval_interval: int,
    max_steps: int | None,
    eval_before_step: bool,
    phase: str,
) -> bool:
    """Mirror the existing SFT held-out evaluation schedule for both phases."""

    if eval_interval < 1:
        raise ValueError("eval_interval must be positive")
    final_scheduled = max_steps is not None
    if phase == "before":
        return bool(
            eval_before_step
            and (
                step == 0
                or (step + 1) % eval_interval == 0
                or (final_scheduled and step + 1 == max_steps)
            )
        )
    if phase == "after":
        return bool(step % eval_interval == 0 or (final_scheduled and step == max_steps))
    raise ValueError(f"unknown benchmark phase {phase!r}; expected 'before' or 'after'")


def summarize_benchmark(
    teacher_rows: Sequence[Sequence[float]],
    student_rows: Sequence[Sequence[float]],
    prompt_lengths: Sequence[int],
    *,
    rewards: Sequence[float] | None = None,
) -> dict[str, float]:
    """Aggregate held-out teacher/student logprobs into comparable metrics.

    Each row covers ``prompt + response`` tokens; entry 0 is the framework's
    initial zero logprob. Only response positions ``[prompt_len:]`` contribute,
    matching ``packed_response_mask`` in the CUDA training pack.
    """

    if len(teacher_rows) != len(student_rows) or len(teacher_rows) != len(prompt_lengths):
        raise ValueError("benchmark rows and prompt lengths must have equal length")
    gaps: list[float] = []
    teacher_values: list[float] = []
    student_values: list[float] = []
    for teacher_row, student_row, prompt_len in zip(teacher_rows, student_rows, prompt_lengths, strict=True):
        teacher_row, student_row = list(teacher_row), list(student_row)
        if len(teacher_row) != len(student_row):
            raise ValueError("teacher and student rows must have equal length")
        start = max(1, int(prompt_len))
        for teacher_value, student_value in zip(teacher_row[start:], student_row[start:], strict=True):
            if not (math.isfinite(float(teacher_value)) and math.isfinite(float(student_value))):
                raise ValueError("benchmark logprobs contain non-finite values")
            teacher_values.append(float(teacher_value))
            student_values.append(float(student_value))
            gaps.append(float(teacher_value) - float(student_value))
    if not gaps:
        raise ValueError("benchmark produced no response tokens")
    stats = {
        "loss": -sum(student_values) / len(student_values),
        "teacher_logprob_mean": sum(teacher_values) / len(teacher_values),
        "student_logprob_mean": sum(student_values) / len(student_values),
        "teacher_student_gap": sum(gaps) / len(gaps),
        "teacher_student_gap_abs": sum(abs(gap) for gap in gaps) / len(gaps),
        "target_tokens": float(len(gaps)),
        "batches": float(len(teacher_rows)),
    }
    if rewards:
        stats["reward_mean"] = sum(float(reward) for reward in rewards) / len(rewards)
    return stats


def summarize_generation_rewards(
    rewards: Sequence[float], *, pass_threshold: float = 0.5
) -> dict[str, float]:
    """Aggregate per-sample task rewards from a held-out generation pass.

    ``accuracy`` counts rewards at or above ``pass_threshold`` as correct,
    which matches binary verifiers such as
    ``examples/math/math_verify_reward.py``. Continuous rewards still get a
    meaningful ``reward_mean``.
    """

    values = [float(reward) for reward in rewards]
    if not values:
        raise ValueError("generation benchmark produced no scored samples")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("generation rewards contain non-finite values")
    return {
        "reward_mean": sum(values) / len(values),
        "accuracy": sum(1 for value in values if value >= pass_threshold) / len(values),
        "samples": float(len(values)),
    }


__all__ = [
    "benchmark_due",
    "summarize_benchmark",
    "summarize_generation_rewards",
]
