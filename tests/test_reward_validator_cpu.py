"""CPU tests for reward-hook runtime validation."""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

# Import directly to avoid heavy engine deps.
_api_dir = Path(__file__).resolve().parent.parent / "areno" / "api"
sys.path.insert(0, str(_api_dir))

from reward_validator import (  # noqa: E402
    RewardValidationError,
    validate_reward_batch,
    validate_reward_fn,
)


class _Record:
    def __init__(self, prompt="test", completion="answer"):
        self.prompt = prompt
        self.completion = completion


class TestSignatureValidation:
    def test_valid_single_arg(self):
        def good_fn(record):
            return 1.0
        wrapped = validate_reward_fn(good_fn)
        assert wrapped(_Record()) == 1.0

    def test_zero_arg_raises(self):
        def no_arg():
            return 1.0
        with pytest.raises(RewardValidationError, match="positional argument"):
            validate_reward_fn(no_arg)


class TestScalarRewards:
    def test_int_return(self):
        assert validate_reward_fn(lambda r: 1)(_Record()) == 1.0

    def test_float_return(self):
        assert validate_reward_fn(lambda r: 0.5)(_Record()) == 0.5

    def test_bool_return(self):
        assert validate_reward_fn(lambda r: True)(_Record()) == 1.0

    def test_negative_reward(self):
        assert validate_reward_fn(lambda r: -1.0)(_Record()) == -1.0


class TestNumpyRewards:
    def test_numpy_scalar(self):
        assert validate_reward_fn(lambda r: np.float32(0.75))(_Record()) == 0.75

    def test_zero_dim_array(self):
        assert validate_reward_fn(lambda r: np.array(0.5))(_Record()) == 0.5

    def test_single_element_array(self):
        assert validate_reward_fn(lambda r: np.array([0.3]))(_Record()) == 0.3

    def test_multi_element_array_raises(self):
        with pytest.raises(RewardValidationError, match="size 2"):
            validate_reward_fn(lambda r: np.array([0.1, 0.2]), strict=True)(_Record())


class TestSequenceRewards:
    def test_single_element_list(self):
        assert validate_reward_fn(lambda r: [0.8])(_Record()) == 0.8

    def test_multi_element_list_raises(self):
        with pytest.raises(RewardValidationError, match="length 2"):
            validate_reward_fn(lambda r: [0.1, 0.2], strict=True)(_Record())


class TestInvalidTypes:
    def test_none_raises(self):
        with pytest.raises(RewardValidationError, match="None"):
            validate_reward_fn(lambda r: None, strict=True)(_Record())

    def test_string_raises(self):
        with pytest.raises(RewardValidationError, match="str"):
            validate_reward_fn(lambda r: "good", strict=True)(_Record())

    def test_dict_raises(self):
        with pytest.raises(RewardValidationError, match="dict"):
            validate_reward_fn(lambda r: {"score": 1.0}, strict=True)(_Record())


class TestNonFiniteValues:
    def test_nan_raises(self):
        with pytest.raises(RewardValidationError, match="NaN"):
            validate_reward_fn(lambda r: float("nan"), strict=True)(_Record())

    def test_inf_raises(self):
        with pytest.raises(RewardValidationError, match="Inf"):
            validate_reward_fn(lambda r: float("inf"), strict=True)(_Record())

    def test_neg_inf_raises(self):
        with pytest.raises(RewardValidationError, match="Inf"):
            validate_reward_fn(lambda r: float("-inf"), strict=True)(_Record())


class TestExceptionHandling:
    def test_exception_strict_raises(self):
        def fn(r):
            raise ValueError("bad record")
        with pytest.raises(RewardValidationError, match="ValueError: bad record"):
            validate_reward_fn(fn, strict=True)(_Record())

    def test_exception_non_strict_returns_zero(self):
        def fn(r):
            raise ValueError("bad record")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert validate_reward_fn(fn, strict=False)(_Record()) == 0.0


class TestNonStrictMode:
    def test_nan_non_strict_warns_and_returns_zero(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_reward_fn(lambda r: float("nan"), strict=False)(_Record())
        assert result == 0.0
        assert len(w) == 1
        assert "NaN" in str(w[0].message)

    def test_none_non_strict_warns_and_returns_zero(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_reward_fn(lambda r: None, strict=False)(_Record())
        assert result == 0.0
        assert len(w) == 1


class TestBatchValidation:
    def test_valid_batch(self):
        def fn(r):
            return 1.0 if "correct" in r.completion else 0.0
        records = [_Record(completion="correct"), _Record(completion="wrong")]
        assert validate_reward_batch(fn, records) == [1.0, 0.0]

    def test_batch_length_matches(self):
        records = [_Record() for _ in range(5)]
        assert len(validate_reward_batch(lambda r: 0.5, records)) == 5

    def test_batch_exception_strict(self):
        def fn(r):
            if "bad" in r.completion:
                raise RuntimeError("explode")
            return 1.0
        with pytest.raises(RewardValidationError, match="sample=1"):
            validate_reward_batch(fn, [_Record(completion="good"), _Record(completion="bad")], strict=True)

    def test_batch_exception_non_strict(self):
        def fn(r):
            if "bad" in r.completion:
                raise RuntimeError("explode")
            return 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert validate_reward_batch(fn, [_Record(completion="good"), _Record(completion="bad")]) == [1.0, 0.0]

    def test_batch_nan_strict(self):
        def fn(r):
            return float("nan") if r.completion == "nan" else 1.0
        with pytest.raises(RewardValidationError, match="sample=1.*NaN"):
            validate_reward_batch(fn, [_Record(completion="ok"), _Record(completion="nan")], strict=True)


class TestBackwardCompatibility:
    def test_math_reward(self):
        def reward_fn(record):
            return 1.0 if record.completion == record.prompt else 0.0
        wrapped = validate_reward_fn(reward_fn)
        assert wrapped(_Record(prompt="3", completion="3")) == 1.0
        assert wrapped(_Record(prompt="3", completion="5")) == 0.0

    def test_game_reward(self):
        def reward_fn(record):
            if "win" in record.completion:
                return 1.0
            elif "illegal" in record.completion:
                return -1.0
            return 0.0
        wrapped = validate_reward_fn(reward_fn)
        assert wrapped(_Record(completion="win")) == 1.0
        assert wrapped(_Record(completion="illegal")) == -1.0


class TestBoundaryCases:
    def test_very_large_reward(self):
        assert validate_reward_fn(lambda r: 1e20)(_Record()) == 1e20

    def test_very_small_reward(self):
        assert validate_reward_fn(lambda r: 1e-20)(_Record()) == 1e-20

    def test_zero_reward(self):
        assert validate_reward_fn(lambda r: 0.0)(_Record()) == 0.0

    def test_empty_batch(self):
        assert validate_reward_batch(lambda r: 1.0, []) == []

    def test_single_record_batch(self):
        assert validate_reward_batch(lambda r: 0.5, [_Record()]) == [0.5]


class TestErrorMessages:
    def test_hook_name_in_error(self):
        with pytest.raises(RewardValidationError, match="hook=my_reward"):
            validate_reward_fn(lambda r: None, name="my_reward", strict=True)(_Record())

    def test_sample_index_in_batch_error(self):
        def fn(r):
            return float("nan")
        with pytest.raises(RewardValidationError, match="sample=0.*NaN"):
            validate_reward_batch(fn, [_Record(), _Record(), _Record()], strict=True)
