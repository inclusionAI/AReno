"""Minimal example demonstrating reward hook validation (issue #222).

This file can be run standalone without GPU or external databases:

    python examples/reward_validation_demo.py

It demonstrates the successful path (valid reward function) and one
invalid input (reward function returning NaN).

Enable validation with::

    ARENO_REWARD_VALIDATION=1 python examples/reward_validation_demo.py
"""

from __future__ import annotations

from areno.api.reward_validation import (
    RewardValidationError,
    validate_and_wrap_reward_fn,
)
from areno.api.rewards import RewardRecord, make_reward_record


def _test_valid_reward():
    """A correct reward function should pass validation and return float."""

    def good_reward(record: RewardRecord) -> float:
        return 1.0 if "4" in record.completion else 0.0

    record = make_reward_record(
        prompt="What is 2+2?",
        completion="The answer is 4",
        source_record={},
        answer=["4"],
    )
    wrapped = validate_and_wrap_reward_fn(good_reward, __import__("pathlib").Path("good_reward"))
    result = wrapped(record)
    assert isinstance(result, float), f"expected float, got {type(result)}"
    assert result == 1.0, f"expected 1.0, got {result}"
    print("[PASS] valid reward function returned", result)


def _test_invalid_reward_nan():
    """A reward function returning NaN should raise RewardValidationError."""

    def nan_reward(record: RewardRecord) -> float:
        return float("nan")

    try:
        validate_and_wrap_reward_fn(nan_reward, __import__("pathlib").Path("nan_reward"))
        raise AssertionError("expected RewardValidationError for NaN")
    except RewardValidationError as exc:
        assert "non-finite" in str(exc), f"unexpected message: {exc}"
        assert "sample_index: dry-run" in str(exc), f"missing sample_index in: {exc}"
        print("[PASS] NaN reward rejected:", exc)


def _test_invalid_reward_string():
    """A reward function returning a string should raise RewardValidationError."""

    def str_reward(record: RewardRecord) -> float:
        return "good"

    try:
        validate_and_wrap_reward_fn(str_reward, __import__("pathlib").Path("str_reward"))
        raise AssertionError("expected RewardValidationError for string")
    except RewardValidationError as exc:
        assert "non-numeric" in str(exc), f"unexpected message: {exc}"
        assert "str" in str(exc), f"missing type name in: {exc}"
        print("[PASS] string reward rejected:", exc)


if __name__ == "__main__":
    import os

    os.environ["ARENO_REWARD_VALIDATION"] = "1"

    _test_valid_reward()
    _test_invalid_reward_nan()
    _test_invalid_reward_string()

    print("\nAll demo cases passed.")
