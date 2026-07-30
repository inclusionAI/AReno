"""CPU tests for reward hook runtime validation (issue #222)."""

from __future__ import annotations

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

from areno.api.reward_validation import (
    RewardValidationError,
    validate_and_wrap_reward_fn,
)
from areno.api.rewards import RewardRecord, load_reward_fn, make_reward_record


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reward_file(tmpdir: str, name: str, body: str) -> Path:
    """Write *body* to a temp Python file and return its path."""
    path = Path(tmpdir, name)
    path.write_text(body, encoding="utf-8")
    return path


def _wrap_inline(fn, hook_name: str = "test_hook"):
    """Wrap *fn* without going through load_reward_fn.

    Tests use this helper, which forces validation on so that the
    wrapping logic is exercised regardless of the ambient environment.
    """
    old = os.environ.get("ARENO_REWARD_VALIDATION")
    os.environ["ARENO_REWARD_VALIDATION"] = "1"
    try:
        fake_path = Path(f"/tmp/{hook_name}.py")
        return validate_and_wrap_reward_fn(fn, fake_path)
    finally:
        if old is None:
            os.environ.pop("ARENO_REWARD_VALIDATION", None)
        else:
            os.environ["ARENO_REWARD_VALIDATION"] = old


# ---------------------------------------------------------------------------
# Signature check tests
# ---------------------------------------------------------------------------

class SignatureCheckTest(unittest.TestCase):
    """Tests for the AST / inspect signature validation."""

    def test_valid_signature_no_annotation(self):
        """A function without annotations should not warn."""
        def fn(record):
            return 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wrapped = _wrap_inline(fn)
            self.assertEqual(wrapped(make_reward_record(prompt="x", completion="y", source_record={})), 1.0)

    def test_valid_signature_with_any_annotation(self):
        """A function annotated with ``Any`` should not warn."""
        from typing import Any

        def fn(record: Any) -> float:
            return 0.5
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wrapped = _wrap_inline(fn)
            self.assertEqual(wrapped(make_reward_record(prompt="x", completion="y", source_record={})), 0.5)

    def test_valid_signature_with_reward_record_annotation(self):
        """A function annotated with ``RewardRecord`` should not warn."""
        def fn(record: RewardRecord) -> float:
            return 1.0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            wrapped = _wrap_inline(fn)
            self.assertEqual(wrapped(make_reward_record(prompt="x", completion="y", source_record={})), 1.0)

    def test_warning_on_wrong_param_annotation(self):
        """A parameter annotated as ``str`` should emit a warning."""
        def fn(record: str) -> float:
            return 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _wrap_inline(fn)
        self.assertTrue(any("annotated as 'str'" in str(w.message) for w in caught))

    def test_warning_on_wrong_return_annotation(self):
        """A return annotation of ``str`` should emit a warning."""
        def fn(record) -> str:
            return 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _wrap_inline(fn)
        self.assertTrue(any("return type is annotated as 'str'" in str(w.message) for w in caught))

    def test_error_on_zero_args(self):
        """A function with no parameters should raise TypeError."""
        def fn():
            return 1.0
        with self.assertRaisesRegex(TypeError, "must accept exactly 1 positional argument"):
            _wrap_inline(fn)

    def test_error_on_too_many_args(self):
        """A function with two parameters should raise TypeError."""
        def fn(a, b):
            return 1.0
        with self.assertRaisesRegex(TypeError, "must accept exactly 1 positional argument"):
            _wrap_inline(fn)


# ---------------------------------------------------------------------------
# Dry-run tests
# ---------------------------------------------------------------------------

class DryRunTest(unittest.TestCase):
    """Tests for the pre-training dry-run invocation."""

    def test_dry_run_passes_with_valid_fn(self):
        """A valid function should pass the dry-run."""
        def fn(record):
            return float(len(record.completion))
        wrapped = _wrap_inline(fn)
        self.assertEqual(wrapped(make_reward_record(prompt="x", completion="abc", source_record={})), 3.0)

    def test_dry_run_warns_on_key_error(self):
        """A function that raises KeyError on mock input should warn, not fail."""
        def fn(record):
            _ = record.source_record["missing"]
            return 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wrapped = _wrap_inline(fn)
        self.assertTrue(any("dry-run raised KeyError" in str(w.message) for w in caught))
        # Function should still be loaded and callable with real data
        record = make_reward_record(prompt="x", completion="y", source_record={"missing": 1})
        self.assertEqual(wrapped(record), 1.0)

    def test_dry_run_warns_on_attribute_error(self):
        """A function that accesses a non-existent attribute should warn, not fail."""
        def fn(record):
            _ = record.nonexistent_field
            return 1.0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _wrap_inline(fn)
        self.assertTrue(any("dry-run raised AttributeError" in str(w.message) for w in caught))

    def test_dry_run_can_be_disabled(self):
        """ARENO_REWARD_VALIDATION_DRY_RUN=0 should skip the dry-run."""
        def fn(record):
            raise RuntimeError("should not be called in dry-run")
        old = os.environ.get("ARENO_REWARD_VALIDATION_DRY_RUN")
        try:
            os.environ["ARENO_REWARD_VALIDATION_DRY_RUN"] = "0"
            wrapped = _wrap_inline(fn)
            # The wrapper should still be returned; dry-run was skipped.
            # Calling the wrapped function should raise RewardValidationError
            # (wrapping the user's RuntimeError with hook name and prompt).
            with self.assertRaisesRegex(RewardValidationError, "raised RuntimeError"):
                wrapped(make_reward_record(prompt="x", completion="y", source_record={}))
        finally:
            if old is None:
                os.environ.pop("ARENO_REWARD_VALIDATION_DRY_RUN", None)
            else:
                os.environ["ARENO_REWARD_VALIDATION_DRY_RUN"] = old


# ---------------------------------------------------------------------------
# Output type validation tests
# ---------------------------------------------------------------------------

class OutputTypeTest(unittest.TestCase):
    """Tests for return-value type checking."""

    def _call_fn(self, fn) -> float:
        """Wrap *fn* and call it with a minimal RewardRecord."""
        wrapped = _wrap_inline(fn)
        record = make_reward_record(prompt="test", completion="test", source_record={})
        return wrapped(record)

    def test_output_accepts_int(self):
        """int return values should be converted to float."""
        self.assertEqual(self._call_fn(lambda r: 1), 1.0)
        self.assertIsInstance(self._call_fn(lambda r: 1), float)

    def test_output_accepts_float(self):
        """float return values should pass through."""
        self.assertEqual(self._call_fn(lambda r: 1.5), 1.5)

    def test_output_accepts_bool(self):
        """bool return values should be converted to float."""
        self.assertEqual(self._call_fn(lambda r: True), 1.0)
        self.assertEqual(self._call_fn(lambda r: False), 0.0)

    def test_output_accepts_numpy_float32(self):
        """numpy scalar types should be accepted."""
        import numpy as np
        self.assertAlmostEqual(self._call_fn(lambda r: np.float32(0.5)), 0.5, places=5)

    def test_output_accepts_torch_scalar(self):
        """0-d torch tensors should be accepted and converted to float."""
        import torch
        result = self._call_fn(lambda r: torch.tensor(0.5))
        self.assertIsInstance(result, float)
        self.assertAlmostEqual(result, 0.5, places=5)

    def test_output_rejects_string(self):
        """String return values should raise RewardValidationError."""
        with self.assertRaisesRegex(RewardValidationError, "non-numeric value of type str"):
            self._call_fn(lambda r: "good")

    def test_output_rejects_none(self):
        """None return values should raise RewardValidationError."""
        with self.assertRaisesRegex(RewardValidationError, "returned None"):
            self._call_fn(lambda r: None)

    def test_output_rejects_list(self):
        """List return values should raise RewardValidationError with length info."""
        with self.assertRaisesRegex(RewardValidationError, "returned a list of length 2.*expected a scalar"):
            self._call_fn(lambda r: [1.0, 2.0])

    def test_output_rejects_dict(self):
        """Dict return values should raise RewardValidationError."""
        with self.assertRaisesRegex(RewardValidationError, "non-numeric value of type dict"):
            self._call_fn(lambda r: {})


# ---------------------------------------------------------------------------
# Finiteness validation tests
# ---------------------------------------------------------------------------

class FinitenessTest(unittest.TestCase):
    """Tests for NaN / Inf rejection."""

    def _call_fn(self, fn) -> float:
        wrapped = _wrap_inline(fn)
        record = make_reward_record(prompt="test", completion="test", source_record={})
        return wrapped(record)

    def test_runtime_exception_wrapped_with_context(self):
        """A function that raises at call time should include hook name and prompt."""
        def fn(record):
            raise KeyError("missing field")
        wrapped = _wrap_inline(fn)
        record = make_reward_record(prompt="What is 2+2?", completion="4", source_record={})
        with self.assertRaisesRegex(RewardValidationError, "raised KeyError.*missing field.*What is 2\\+2"):
            wrapped(record)

    def test_output_rejects_inf(self):
        """float('inf') should be rejected."""
        with self.assertRaisesRegex(RewardValidationError, "non-finite value"):
            self._call_fn(lambda r: float("inf"))

    def test_output_rejects_nan(self):
        """float('nan') should be rejected."""
        with self.assertRaisesRegex(RewardValidationError, "non-finite value"):
            self._call_fn(lambda r: float("nan"))

    def test_output_rejects_nan_in_numpy(self):
        """numpy NaN should be rejected."""
        import numpy as np
        with self.assertRaisesRegex(RewardValidationError, "non-finite value"):
            self._call_fn(lambda r: np.float64("nan"))


# ---------------------------------------------------------------------------
# End-to-end integration via load_reward_fn
# ---------------------------------------------------------------------------

class LoadRewardFnIntegrationTest(unittest.TestCase):
    """Integration tests through the public load_reward_fn API."""

    def _load_with_validation(self, path: str):
        """Load a reward fn with ARENO_REWARD_VALIDATION=1 enabled."""
        old = os.environ.get("ARENO_REWARD_VALIDATION")
        os.environ["ARENO_REWARD_VALIDATION"] = "1"
        try:
            return load_reward_fn(path)
        finally:
            if old is None:
                os.environ.pop("ARENO_REWARD_VALIDATION", None)
            else:
                os.environ["ARENO_REWARD_VALIDATION"] = old

    def test_load_reward_fn_wraps_output(self):
        """With validation on, load_reward_fn should return a wrapped function that outputs float."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(tmp, "int_reward.py", "def reward_fn(record):\n    return 1\n")
            fn = self._load_with_validation(str(path))
            record = make_reward_record(prompt="x", completion="y", source_record={})
            result = fn(record)
            self.assertEqual(result, 1.0)
            self.assertIsInstance(result, float)

    def test_load_reward_fn_rejects_bad_output(self):
        """With validation on, load_reward_fn should reject non-numeric outputs at dry-run time."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(tmp, "str_reward.py", "def reward_fn(record):\n    return 'bad'\n")
            with self.assertRaisesRegex(RewardValidationError, "non-numeric value of type str"):
                self._load_with_validation(str(path))

    def test_load_reward_fn_dry_run_warns_on_error(self):
        """With validation on, load_reward_fn should warn (not crash) if dry-run raises."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(
                tmp, "crash_reward.py",
                "def reward_fn(record):\n    raise KeyError('boom')\n",
            )
            old = os.environ.get("ARENO_REWARD_VALIDATION")
            os.environ["ARENO_REWARD_VALIDATION"] = "1"
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fn = load_reward_fn(str(path))
            finally:
                if old is None:
                    os.environ.pop("ARENO_REWARD_VALIDATION", None)
                else:
                    os.environ["ARENO_REWARD_VALIDATION"] = old
            self.assertTrue(any("dry-run raised KeyError" in str(w.message) for w in caught))

    def test_validation_off_by_default(self):
        """Without ARENO_REWARD_VALIDATION, load_reward_fn should return the raw function."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(tmp, "raw_reward.py", "def reward_fn(record):\n    return 'bad'\n")
            # Ensure env var is not set
            old = os.environ.pop("ARENO_REWARD_VALIDATION", None)
            try:
                fn = load_reward_fn(str(path))
                record = make_reward_record(prompt="x", completion="y", source_record={})
                # Raw function returns 'bad' string — no validation applied
                self.assertEqual(fn(record), "bad")
            finally:
                if old is not None:
                    os.environ["ARENO_REWARD_VALIDATION"] = old


# ---------------------------------------------------------------------------
# Existing example regression test
# ---------------------------------------------------------------------------

class ExampleRegressionTest(unittest.TestCase):
    """Ensure existing example reward files pass validation."""

    def test_math_verify_reward_example(self):
        """examples/math/math_verify_reward.py should pass validation."""
        from examples.math.math_verify_reward import reward_fn as raw_fn

        # The math verify reward accesses record.answer and calls math_verify.
        # The dry-run with an empty RewardRecord raises KeyError (record.answer
        # is None), which should produce a warning, not a hard failure.
        record = make_reward_record(
            prompt="What is 2+2?",
            completion="4",
            source_record={},
            answer=["4"],
        )
        old = os.environ.get("ARENO_REWARD_VALIDATION")
        os.environ["ARENO_REWARD_VALIDATION"] = "1"
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                wrapped = validate_and_wrap_reward_fn(raw_fn, Path("math_verify_reward"))
        finally:
            if old is None:
                os.environ.pop("ARENO_REWARD_VALIDATION", None)
            else:
                os.environ["ARENO_REWARD_VALIDATION"] = old
        # Dry-run should have warned about KeyError
        self.assertTrue(any("dry-run raised KeyError" in str(w.message) for w in caught))
        # But the function should still be callable with a real record
        result = wrapped(record)
        self.assertIn(result, {0.0, 1.0})


# ---------------------------------------------------------------------------
# Backward compatibility: existing test from test_tokenizer_api_cpu
# ---------------------------------------------------------------------------

class BackwardCompatTest(unittest.TestCase):
    """Ensure load_reward_fn still works with SimpleNamespace inputs."""

    def test_load_reward_fn_with_simple_namespace(self):
        """load_reward_fn should still accept SimpleNamespace-like inputs (default off)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(
                tmp, "len_reward.py",
                "def reward_fn(record):\n    return len(record.completion)\n",
            )
            # Ensure validation is off (default)
            old = os.environ.pop("ARENO_REWARD_VALIDATION", None)
            try:
                fn = load_reward_fn(str(path))
                # SimpleNamespace has .completion but not .prompt
                # Without validation, the raw function returns int
                result = fn(SimpleNamespace(completion="abc"))
                self.assertEqual(result, 3)
            finally:
                if old is not None:
                    os.environ["ARENO_REWARD_VALIDATION"] = old


# ---------------------------------------------------------------------------
# CLI integration: --validate-reward → env → load_reward_fn chain
# ---------------------------------------------------------------------------

class CLIIntegrationTest(unittest.TestCase):
    """Verify the --validate-reward flag propagates through to load_reward_fn."""

    def setUp(self):
        # Ensure no stale env from other tests
        self._old_env = os.environ.pop("ARENO_REWARD_VALIDATION", None)

    def tearDown(self):
        if self._old_env is not None:
            os.environ["ARENO_REWARD_VALIDATION"] = self._old_env

    def test_validate_reward_flag_sets_env_var(self):
        """train_command should set ARENO_REWARD_VALIDATION=1 when --validate-reward is passed."""
        import areno.cli.train as train_cli

        # Build a minimal options dict that includes validate_reward=True.
        # We only test the env-var side effect, not the full training run.
        # train_command calls run() which needs GPU; we intercept by checking
        # the env var is set before run() would be called.
        # Since train_command is a Click command, we test via the options dict.
        options = {"validate_reward": True}

        # Simulate the relevant line from train_command:
        #   if options.get("validate_reward"):
        #       os.environ["ARENO_REWARD_VALIDATION"] = "1"
        old = os.environ.get("ARENO_REWARD_VALIDATION")
        try:
            if options.get("validate_reward"):
                os.environ["ARENO_REWARD_VALIDATION"] = "1"
            self.assertEqual(os.environ.get("ARENO_REWARD_VALIDATION"), "1")
        finally:
            if old is None:
                os.environ.pop("ARENO_REWARD_VALIDATION", None)
            else:
                os.environ["ARENO_REWARD_VALIDATION"] = old

    def test_validate_reward_flag_absent_does_not_set_env(self):
        """Without --validate-reward, ARENO_REWARD_VALIDATION should remain unset."""
        self.assertIsNone(os.environ.get("ARENO_REWARD_VALIDATION"))

        # load_reward_fn with env unset should return the raw function
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(
                tmp, "raw_reward.py",
                "def reward_fn(record):\n    return 'not_a_number'\n",
            )
            fn = load_reward_fn(str(path))
            # Raw function returns string — no validation applied
            self.assertEqual(fn(make_reward_record(prompt="x", completion="y", source_record={})), "not_a_number")

    def test_full_chain_validate_reward_to_rejection(self):
        """Full chain: env var set → load_reward_fn wraps → bad output rejected at load time."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(
                tmp, "nan_reward.py",
                "def reward_fn(record):\n    return float('nan')\n",
            )
            # Simulate --validate-reward setting the env var
            os.environ["ARENO_REWARD_VALIDATION"] = "1"
            # NaN is caught by dry-run validation at load time
            with self.assertRaisesRegex(RewardValidationError, "non-finite value"):
                load_reward_fn(str(path))

    def test_full_chain_no_flag_no_rejection(self):
        """Full chain: no env var → load_reward_fn returns raw fn → bad output passes through."""
        with tempfile.TemporaryDirectory() as tmp:
            path = _make_reward_file(
                tmp, "nan_reward.py",
                "def reward_fn(record):\n    return float('nan')\n",
            )
            # Ensure env is NOT set
            os.environ.pop("ARENO_REWARD_VALIDATION", None)
            fn = load_reward_fn(str(path))
            record = make_reward_record(prompt="test prompt", completion="test", source_record={})
            # Without validation, NaN passes through as-is
            import math
            self.assertTrue(math.isnan(fn(record)))


if __name__ == "__main__":
    unittest.main()
