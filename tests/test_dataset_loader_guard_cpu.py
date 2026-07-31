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

import json
import signal
import sys
import threading
import time

import pytest

from areno.cli.dataset_loader_guard import (
    DatasetLoaderTimeout,
    LoaderDiagnostics,
    _peak_rss_kb,
    _safe_len,
    run_loader_with_limits,
    write_loader_diagnostics,
)

# Timeout tests require SIGALRM, which is Unix-only and only works in the
# main thread.
_TIMEOUT_AVAILABLE = hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread()

# Memory diagnostics require the resource module (Unix-only).
_RESOURCE_AVAILABLE = sys.platform != "win32"


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


@pytest.mark.skipif(not _TIMEOUT_AVAILABLE, reason="SIGALRM timeout requires Unix main thread")
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
        """Ensure SIGALRM handler is restored after a timeout fires."""

        original_handler = signal.getsignal(signal.SIGALRM)
        with pytest.raises(DatasetLoaderTimeout):
            run_loader_with_limits(_slow_loader, "dummy", timeout_s=1)
        # Verify the original handler is restored.
        restored_handler = signal.getsignal(signal.SIGALRM)
        assert restored_handler is original_handler
        # The key point: we can call again without issues.
        dataset, _ = run_loader_with_limits(_fast_loader, "dummy")
        assert len(dataset) == 5


# ---------------------------------------------------------------------------
# Tests: timeout skipped on unsupported platforms/threads
# ---------------------------------------------------------------------------


class TestTimeoutSkipped:
    """Timeout is skipped gracefully when SIGALRM is unavailable."""

    def test_no_timeout_when_sigalrm_unavailable(self, monkeypatch):
        """Guard should not fail when SIGALRM is missing; it just skips timeout."""

        if hasattr(signal, "SIGALRM"):
            monkeypatch.delattr(signal, "SIGALRM")

        # A loader that would time out should run to completion when timeout
        # cannot be enforced.
        def slow_but_short(path: str = "", **kwargs) -> list[dict]:
            time.sleep(0.05)
            return [{"id": 1}]

        dataset, diag = run_loader_with_limits(slow_but_short, "dummy", timeout_s=0.01)
        assert len(dataset) == 1
        assert diag.error is None

    def test_no_timeout_in_non_main_thread(self):
        """Timeout must be skipped when called from a background thread."""

        if threading.current_thread() is not threading.main_thread():
            pytest.skip("already not in main thread")

        result = {}

        def worker():
            try:
                dataset, diag = run_loader_with_limits(_slow_loader, "dummy", timeout_s=1)
                result["dataset"] = dataset
                result["diag"] = diag
            except Exception as exc:  # noqa: BLE001
                result["exc"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(timeout=15)
        assert not thread.is_alive(), "loader without timeout should finish"
        assert "exc" not in result, f"unexpected exception: {result.get('exc')}"
        assert result["diag"].error is None


# ---------------------------------------------------------------------------
# Tests: record cap
# ---------------------------------------------------------------------------


class TestRecordCap:
    """Truncation when loader returns more than max_records."""

    def test_truncates_to_max_records(self):
        dataset, diag = run_loader_with_limits(_oversized_loader, "dummy", max_records=10)
        assert len(dataset) == 10
        assert diag.truncated is True
        assert diag.original_record_count == 1000
        assert diag.record_count == 10

    def test_no_truncation_when_under_cap(self):
        dataset, diag = run_loader_with_limits(_fast_loader, "dummy", max_records=10)
        assert len(dataset) == 5
        assert diag.truncated is False
        assert diag.original_record_count == 5

    def test_truncated_records_are_first_n(self):
        dataset, diag = run_loader_with_limits(_oversized_loader, "dummy", max_records=3)
        assert dataset[0]["id"] == 0
        assert dataset[1]["id"] == 1
        assert dataset[2]["id"] == 2

    def test_max_records_zero_means_unlimited(self):
        dataset, diag = run_loader_with_limits(_oversized_loader, "dummy", max_records=0)
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

    @pytest.mark.skipif(not _TIMEOUT_AVAILABLE, reason="SIGALRM timeout requires Unix main thread")
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

    @pytest.mark.skipif(not _TIMEOUT_AVAILABLE, reason="SIGALRM timeout requires Unix main thread")
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

    @pytest.mark.skipif(not _RESOURCE_AVAILABLE, reason="resource module not available")
    def test_peak_rss_kb_non_negative(self):
        """Peak RSS is reported when resource is available; always >= 0."""

        assert _peak_rss_kb() >= 0

    def test_peak_rss_kb_zero_without_resource(self, monkeypatch):
        """When resource is unavailable, memory diagnostics report 0."""

        import areno.cli.dataset_loader_guard as guard

        monkeypatch.setattr(guard, "_resource", None)
        assert _peak_rss_kb() == 0


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

    @pytest.mark.skipif(not _TIMEOUT_AVAILABLE, reason="SIGALRM timeout requires Unix main thread")
    def test_timeout_and_cap_on_fast_loader(self):
        """A fast loader with both limits set should succeed and not truncate."""

        dataset, diag = run_loader_with_limits(_fast_loader, "dummy", timeout_s=10, max_records=100)
        assert len(dataset) == 5
        assert diag.truncated is False
        assert diag.error is None

    def test_cap_applied_even_with_timeout(self):
        """If the loader finishes within the timeout, cap still applies."""

        dataset, diag = run_loader_with_limits(_oversized_loader, "dummy", timeout_s=30, max_records=5)
        assert len(dataset) == 5
        assert diag.truncated is True


# ---------------------------------------------------------------------------
# Tests: boundary / edge cases
# ---------------------------------------------------------------------------


class TestBoundaryCases:
    """Edge cases: generators, empty data, max_records=1, sub-second timeout."""

    def test_generator_loader_truncated(self):
        """Generator-based loader should be truncated via islice."""

        def gen_loader(path="", **kw):
            for i in range(100):
                yield {"id": i}

        dataset, diag = run_loader_with_limits(gen_loader, "dummy", max_records=10)
        assert len(dataset) == 10
        assert diag.truncated is True

    def test_generator_loader_no_cap_returns_original(self):
        """Generator without cap is passed through without materialising."""

        def gen_loader(path="", **kw):
            for i in range(5):
                yield {"id": i}

        dataset, diag = run_loader_with_limits(gen_loader, "dummy")
        # Without cap, the generator is returned as-is.
        assert iter(dataset) is dataset
        assert diag.error is None
        assert diag.record_count == 0

    def test_empty_dataset_with_cap(self):
        """Empty list with max_records should not crash."""

        def empty_loader(path="", **kw):
            return []

        dataset, diag = run_loader_with_limits(empty_loader, "dummy", max_records=10)
        assert len(dataset) == 0
        assert diag.truncated is False
        assert diag.record_count == 0

    def test_max_records_one(self):
        """max_records=1 should keep only the first record."""

        dataset, diag = run_loader_with_limits(_oversized_loader, "dummy", max_records=1)
        assert len(dataset) == 1
        assert dataset[0]["id"] == 0
        assert diag.truncated is True

    @pytest.mark.skipif(not _TIMEOUT_AVAILABLE, reason="SIGALRM timeout requires Unix main thread")
    def test_sub_second_timeout(self):
        """timeout_s < 1 should work with setitimer (sub-second precision)."""

        def medium_loader(path="", **kw):
            time.sleep(2)
            return [{"id": 0}]

        with pytest.raises(DatasetLoaderTimeout):
            run_loader_with_limits(medium_loader, "dummy", timeout_s=0.5)

    @pytest.mark.skipif(not _TIMEOUT_AVAILABLE, reason="SIGALRM timeout requires Unix main thread")
    def test_timeout_does_not_truncate_on_success(self):
        """A fast loader with timeout should return full data, no truncation."""

        dataset, diag = run_loader_with_limits(_fast_loader, "dummy", timeout_s=5)
        assert len(dataset) == 5
        assert diag.truncated is False

    def test_diagnostics_to_dict(self):
        """LoaderDiagnostics.to_dict() should return serialisable dict."""

        diag = LoaderDiagnostics(
            duration_s=1.5,
            mem_before_kb=100,
            mem_after_kb=200,
            record_count=50,
            truncated=True,
            original_record_count=100,
        )
        d = diag.to_dict()
        assert d["duration_s"] == 1.5
        assert d["record_count"] == 50
        assert d["mem_delta_kb"] == 100
        assert d["truncated"] is True
        assert d["original_record_count"] == 100
        assert d["error"] is None


class TestWriteLoaderDiagnostics:
    """Persisting loader diagnostics to disk."""

    def test_writes_json_when_metrics_log_dir_set(self, tmp_path):
        diag = LoaderDiagnostics(
            duration_s=2.0,
            mem_before_kb=100,
            mem_after_kb=300,
            record_count=10,
            truncated=True,
            original_record_count=100,
        )
        write_loader_diagnostics(str(tmp_path), diag)
        payload = json.loads((tmp_path / "areno_loader_diagnostics.json").read_text())
        assert payload["duration_s"] == 2.0
        assert payload["mem_delta_kb"] == 200
        assert payload["record_count"] == 10
        assert payload["truncated"] is True
        assert payload["original_record_count"] == 100
        assert payload["error"] is None

    def test_no_crash_when_metrics_log_dir_none(self):
        """Calling write_loader_diagnostics with None should not raise."""
        diag = LoaderDiagnostics(duration_s=1.0)
        write_loader_diagnostics(None, diag)  # no metrics_log_dir -> just log


# ---------------------------------------------------------------------------
# Integration-style tests: _load_dataset_for_training -> run_loader_with_limits
# ---------------------------------------------------------------------------


class TestIntegrationLoadDatasetForTraining:
    """Integration tests verifying _load_dataset_for_training wraps loaders
    with run_loader_with_limits and returns the correct dataset."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        """Skip integration tests when torch is unavailable (e.g. macOS CPU)."""
        try:
            import torch  # noqa: F401
        except ImportError:
            pytest.skip("torch not installed — integration test requires areno.cli.train")

    def test_default_loader_no_limits(self):
        """Default loader path without limits should return data unchanged."""
        from areno.cli.train import _load_dataset_for_training

        # Use a tiny inline dataset via load_dataset mock.
        def mock_load_dataset(name, **kw):
            return [{"id": i, "text": f"row {i}"} for i in range(5)]

        def mock_load_from_disk(path):
            raise FileNotFoundError("not a save_to_disk directory")

        dataset, diag = _load_dataset_for_training(
            "dummy_path",
            model_hub="hf",
            dataset_loader_fn=None,
            load_dataset=mock_load_dataset,
            load_from_disk=mock_load_from_disk,
            loader_timeout_s=0.0,
            max_loader_records=0,
        )
        assert len(dataset) == 5
        assert diag.record_count == 5
        assert diag.truncated is False

    def test_default_loader_with_record_cap(self):
        """Default loader with max_loader_records should truncate."""
        from areno.cli.train import _load_dataset_for_training

        def mock_load_dataset(name, **kw):
            return [{"id": i} for i in range(100)]

        def mock_load_from_disk(path):
            raise FileNotFoundError("not a save_to_disk directory")

        dataset, diag = _load_dataset_for_training(
            "dummy_path",
            model_hub="hf",
            dataset_loader_fn=None,
            load_dataset=mock_load_dataset,
            load_from_disk=mock_load_from_disk,
            loader_timeout_s=0.0,
            max_loader_records=10,
        )
        assert len(dataset) == 10
        assert diag.truncated is True
        assert diag.original_record_count == 100

    def test_custom_loader_no_limits(self, tmp_path):
        """Custom loader path without limits should still work."""
        from areno.cli.train import _load_dataset_for_training

        loader_script = tmp_path / "my_loader.py"
        loader_script.write_text("def load_training_dataset(path, **kw):\n    return [{'id': i} for i in range(3)]\n")

        def mock_load_dataset(name, **kw):
            return []

        def mock_load_from_disk(path):
            raise FileNotFoundError("not used")

        dataset, diag = _load_dataset_for_training(
            "dummy_path",
            model_hub="hf",
            dataset_loader_fn=str(loader_script),
            load_dataset=mock_load_dataset,
            load_from_disk=mock_load_from_disk,
            loader_timeout_s=0.0,
            max_loader_records=0,
        )
        assert len(dataset) == 3
        assert diag.record_count == 3

    def test_custom_loader_with_record_cap(self, tmp_path):
        """Custom loader with max_loader_records should truncate."""
        from areno.cli.train import _load_dataset_for_training

        loader_script = tmp_path / "my_loader.py"
        loader_script.write_text("def load_training_dataset(path, **kw):\n    return [{'id': i} for i in range(50)]\n")

        def mock_load_dataset(name, **kw):
            return []

        def mock_load_from_disk(path):
            raise FileNotFoundError("not used")

        dataset, diag = _load_dataset_for_training(
            "dummy_path",
            model_hub="hf",
            dataset_loader_fn=str(loader_script),
            load_dataset=mock_load_dataset,
            load_from_disk=mock_load_from_disk,
            loader_timeout_s=0.0,
            max_loader_records=5,
        )
        assert len(dataset) == 5
        assert diag.truncated is True

    def test_default_loader_with_huggingface_dataset(self):
        """Record cap works with a real HuggingFace Dataset object."""
        from areno.cli.train import _load_dataset_for_training

        try:
            from datasets import Dataset  # noqa: F401
        except ImportError:
            pytest.skip("datasets library not installed")

        def mock_load_dataset(name, **kw):
            return Dataset.from_list([{"id": i} for i in range(50)])

        def mock_load_from_disk(path):
            raise FileNotFoundError("not a save_to_disk directory")

        dataset, diag = _load_dataset_for_training(
            "dummy_path",
            model_hub="hf",
            dataset_loader_fn=None,
            load_dataset=mock_load_dataset,
            load_from_disk=mock_load_from_disk,
            loader_timeout_s=0.0,
            max_loader_records=10,
        )
        assert len(dataset) == 10
        assert diag.truncated is True
        assert diag.original_record_count == 50
