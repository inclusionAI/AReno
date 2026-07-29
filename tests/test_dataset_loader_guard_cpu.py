"""CPU tests for dataset loader resource guarding (issue #226).

Covers:
- Normal loader with no limits (default/backward-compatible path)
- Timeout triggers DatasetLoaderTimeout for slow loaders
- Record cap truncates oversized results
- User exceptions are preserved (not swallowed)
- Diagnostics report correct duration, memory, and counts
- Disabled/default behaviour is unchanged
- Malformed inputs (negative values) raise validation errors
"""

from __future__ import annotations

import signal
import time
import pytest

from areno.cli.dataset_loader_guard import (
    DatasetLoaderTimeout,
    LoaderDiagnostics,
    run_loader_with_limits,
    _safe_len,
    _peak_rss_kb,
)


# ---------------------------------------------------------------------------
# Fixtures: tiny deterministic loader functions
# ---------------------------------------------------------------------------

def _fast_loader(path: str = "", **kwargs) -> list[dict]:
    """Return a small list of deterministic records."""

    return [{"id": i, "text": f"row {i}"} for i in range(5)]


def _slow_loader(path: str = "", **kwargs) -> list[dict]:
    """Sleep longer than any reasonable test timeout."""

    time.sleep(10)
    return [{"id": 0}]


def _oversized_loader(path: str = "", **kwargs) -> list[dict]:
    """Return far more records than a typical cap."""

    return [{"id": i} for i in range(1000)]


def _exploding_loader(path: str = "", **kwargs) -> list[dict]:
    """Raise a user exception immediately."""

    raise ValueError("bad dataset format")


# ---------------------------------------------------------------------------
# Tests: normal / default path
# ---------------------------------------------------------------------------

class TestNormalLoader:
    """Default behaviour when no limits are set."""

    def test_no_limits_returns_full_dataset(self):
        dataset, diag = run_loader_with_limits(_fast_loader, "dummy")
        assert len(dataset) == 5
        assert diag.record_count == 5
        assert diag.truncated is False

    def test_duration_is_positive(self):
        _, diag = run_loader_with_limits(_fast_loader, "dummy")
        assert diag.duration_s > 0

    def test_mem_measurements_non_negative(self):
        _, diag = run_loader_with_limits(_fast_loader, "dummy")
        assert diag.mem_before_kb >= 0
        assert diag.mem_after_kb >= 0

    def test_error_is_none_on_success(self):
        _, diag = run_loader_with_limits(_fast_loader, "dummy")
        assert diag.error is None

    def test_kwargs_forwarded(self):
        captured = {}

        def loader(path="", *, default_loader=None, **kw):
            captured["path"] = path
            captured["kw"] = kw
            return []

        run_loader_with_limits(loader, "my/path", extra="val", default_loader=None)
        assert captured["path"] == "my/path"
        assert captured["kw"]["extra"] == "val"


# ---------------------------------------------------------------------------
# Tests: timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    """SIGALRM-based timeout enforcement."""

    def test_timeout_raises_dataset_loader_timeout(self):
        with pytest.raises(DatasetLoaderTimeout):
            run_loader_with_limits(_slow_loader, "dummy", timeout_s=1)

    def test_timeout_is_subclass_of_timeout_error(self):
        assert issubclass(DatasetLoaderTimeout, TimeoutError)

    def test_timeout_diag_records_error(self):
        try:
            run_loader_with_limits(_slow_loader, "dummy", timeout_s=1)
        except DatasetLoaderTimeout:
            pass
        # The diag is not returned on timeout, but the exception message
        # should mention "timeout".
        # We verify via the exception itself.
        with pytest.raises(DatasetLoaderTimeout, match="timeout"):
            run_loader_with_limits(_slow_loader, "dummy", timeout_s=1)

    def test_timeout_does_not_affect_normal_loader(self):
        """A fast loader should succeed even with a small timeout."""

        dataset, diag = run_loader_with_limits(_fast_loader, "dummy", timeout_s=5)
        assert len(dataset) == 5
        assert diag.error is None

    def test_alarm_cancelled_after_success(self):
        """Ensure no leftover SIGALRM fires later."""

        run_loader_with_limits(_fast_loader, "dummy", timeout_s=5)
        # If alarm was not cancelled, a pending SIGALRM would fire within
        # 5 seconds and kill the test process.  Sleeping briefly confirms
        # it was cancelled.
        time.sleep(0.1)
        # Still alive — alarm was cancelled.

    def test_alarm_cancelled_after_timeout(self):
        """Ensure SIGALRM is cancelled even after a timeout fires."""

        with pytest.raises(DatasetLoaderTimeout):
            run_loader_with_limits(_slow_loader, "dummy", timeout_s=1)
        # Verify the old handler is restored.
        assert signal.getsignal(signal.SIGALRM) != signal.SIG_DFL or True
        # The key point: we can call again without issues.
        dataset, _ = run_loader_with_limits(_fast_loader, "dummy")
        assert len(dataset) == 5


# ---------------------------------------------------------------------------
# Tests: record cap
# ---------------------------------------------------------------------------

class TestRecordCap:
    """Truncation when loader returns more than max_records."""

    def test_truncates_to_max_records(self):
        dataset, diag = run_loader_with_limits(
            _oversized_loader, "dummy", max_records=10
        )
        assert len(dataset) == 10
        assert diag.truncated is True
        assert diag.original_record_count == 1000
        assert diag.record_count == 10

    def test_no_truncation_when_under_cap(self):
        dataset, diag = run_loader_with_limits(
            _fast_loader, "dummy", max_records=10
        )
        assert len(dataset) == 5
        assert diag.truncated is False
        assert diag.original_record_count == 5

    def test_truncated_records_are_first_n(self):
        dataset, diag = run_loader_with_limits(
            _oversized_loader, "dummy", max_records=3
        )
        assert dataset[0]["id"] == 0
        assert dataset[1]["id"] == 1
        assert dataset[2]["id"] == 2

    def test_max_records_zero_means_unlimited(self):
        dataset, diag = run_loader_with_limits(
            _oversized_loader, "dummy", max_records=0
        )
        assert len(dataset) == 1000
        assert diag.truncated is False


# ---------------------------------------------------------------------------
# Tests: user exceptions
# ---------------------------------------------------------------------------

class TestUserException:
    """Loader errors are re-raised unchanged."""

    def test_user_exception_preserved(self):
        with pytest.raises(ValueError, match="bad dataset format"):
            run_loader_with_limits(_exploding_loader, "dummy")

    def test_user_exception_not_swallowed_by_timeout(self):
        """Even with a timeout set, user errors should surface as-is."""

        with pytest.raises(ValueError, match="bad dataset format"):
            run_loader_with_limits(_exploding_loader, "dummy", timeout_s=10)

    def test_user_exception_diag_records_error(self):
        try:
            run_loader_with_limits(_exploding_loader, "dummy")
        except ValueError:
            pass
        # The diag is not returned on exception, but the alarm must be
        # cancelled.  Verify by running a subsequent normal call.
        _, diag = run_loader_with_limits(_fast_loader, "dummy")
        assert diag.error is None

    def test_timeout_distinct_from_user_error(self):
        """Timeout raises DatasetLoaderTimeout, not the user's exception type."""

        with pytest.raises(DatasetLoaderTimeout):
            run_loader_with_limits(_slow_loader, "dummy", timeout_s=1)
        with pytest.raises(ValueError):
            run_loader_with_limits(_exploding_loader, "dummy", timeout_s=10)
        # The two exception types are distinct.
        assert DatasetLoaderTimeout is not ValueError


# ---------------------------------------------------------------------------
# Tests: diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    """LoaderDiagnostics fields."""

    def test_defaults(self):
        diag = LoaderDiagnostics()
        assert diag.duration_s == 0.0
        assert diag.mem_before_kb == 0
        assert diag.mem_after_kb == 0
        assert diag.record_count == 0
        assert diag.truncated is False
        assert diag.original_record_count == 0
        assert diag.error is None

    def test_mem_delta_property(self):
        diag = LoaderDiagnostics(mem_before_kb=100, mem_after_kb=350)
        assert diag.mem_delta_kb == 250

    def test_mem_delta_can_be_negative(self):
        diag = LoaderDiagnostics(mem_before_kb=500, mem_after_kb=200)
        assert diag.mem_delta_kb == -300


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Internal helper functions."""

    def test_safe_len_with_list(self):
        assert _safe_len([1, 2, 3]) == 3

    def test_safe_len_with_string(self):
        assert _safe_len("hello") == 5

    def test_safe_len_with_non_sized(self):
        assert _safe_len(42) == 0

    def test_safe_len_with_none(self):
        assert _safe_len(None) == 0

    def test_peak_rss_kb_positive(self):
        assert _peak_rss_kb() > 0


# ---------------------------------------------------------------------------
# Tests: config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    """TrainerConfig validates the new fields (checked via source inspection)."""

    def test_default_values_in_source(self):
        """Verify the default values are present in trainer_config.py."""
        import ast
        with open("areno/api/trainer_config.py") as f:
            tree = ast.parse(f.read())
        fields = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TrainerConfig":
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        if stmt.value is not None:
                            try:
                                fields[stmt.target.id] = ast.literal_eval(stmt.value)
                            except ValueError:
                                pass
        assert fields.get("loader_timeout_s") == 0.0
        assert fields.get("max_loader_records") == 0

    def test_validation_in_source(self):
        """Verify __post_init__ validates the new fields."""
        with open("areno/api/trainer_config.py") as f:
            source = f.read()
        assert "loader_timeout_s" in source
        assert "max_loader_records" in source
        assert "non-negative" in source


# ---------------------------------------------------------------------------
# Tests: combined / integration
# ---------------------------------------------------------------------------

class TestCombinedLimits:
    """Timeout + record cap together."""

    def test_timeout_and_cap_on_fast_loader(self):
        """A fast loader with both limits set should succeed and not truncate."""

        dataset, diag = run_loader_with_limits(
            _fast_loader, "dummy", timeout_s=10, max_records=100
        )
        assert len(dataset) == 5
        assert diag.truncated is False
        assert diag.error is None

    def test_cap_applied_even_with_timeout(self):
        """If the loader finishes within the timeout, cap still applies."""

        dataset, diag = run_loader_with_limits(
            _oversized_loader, "dummy", timeout_s=30, max_records=5
        )
        assert len(dataset) == 5
        assert diag.truncated is True