"""CPU-only tests for :mod:`areno.cli.run_summary`.

These tests verify the formatting logic and the ``print_run_summary`` helper
without requiring any GPU, model loading, or backend initialization.
"""

from __future__ import annotations

import io
import json

import pytest

from areno.cli.run_summary import (
    RunSummaryData,
    _format_duration,
    format_run_summary,
    print_run_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_summary(
    *,
    outcome: str = "success",
    duration_s: float = 754.0,
    algo: str = "gspo",
    model: str = "Qwen/Qwen3-0.6B",
    final_step: int = 200,
    final_epoch: int = 2,
    metrics: dict | None = None,
    samples_processed: int = 1000,
    samples_trained: int = 950,
    samples_skipped: int = 50,
    errors: list[str] | None = None,
) -> RunSummaryData:
    """Build a :class:`RunSummaryData` with sensible defaults."""
    data = RunSummaryData(algo=algo, model=model)
    data.outcome = outcome
    data.duration_s = duration_s
    data.final_step = final_step
    data.final_epoch = final_epoch
    data.metrics = metrics if metrics is not None else {"loss": 0.0023, "reward": 0.87}
    data.samples_processed = samples_processed
    data.samples_trained = samples_trained
    data.samples_skipped = samples_skipped
    data.errors = errors or []
    return data


# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_zero(self):
        assert _format_duration(0) == "0s"

    def test_negative_clamped_to_zero(self):
        assert _format_duration(-5) == "0s"

    def test_seconds(self):
        assert _format_duration(45) == "45s"

    def test_minutes(self):
        assert _format_duration(125) == "2m 05s"

    def test_hours(self):
        assert _format_duration(3725) == "1h 02m 05s"


# ---------------------------------------------------------------------------
# format_run_summary — human-readable
# ---------------------------------------------------------------------------

class TestFormatRunSummaryText:
    def test_success_contains_key_fields(self):
        text = format_run_summary(_make_summary())
        assert "success" in text
        assert "12m 34s" in text
        assert "gspo" in text
        assert "Qwen/Qwen3-0.6B" in text
        assert "200" in text
        assert "950 trained" in text
        assert "50 skipped" in text
        assert "loss" in text
        assert "reward" in text

    def test_interrupted_outcome(self):
        text = format_run_summary(_make_summary(outcome="interrupted"))
        assert "interrupted" in text

    def test_error_outcome_shows_errors(self):
        text = format_run_summary(
            _make_summary(outcome="error", errors=["ValueError: bad dataset row"])
        )
        assert "error" in text
        assert "ValueError: bad dataset row" in text

    def test_empty_metrics_shows_placeholder(self):
        text = format_run_summary(_make_summary(metrics={}))
        assert "(no metrics recorded)" in text

    def test_long_model_path_truncated(self):
        long_path = "/very/long/path/to/some/model/checkpoint/directory/" * 3
        text = format_run_summary(_make_summary(model=long_path))
        assert "..." in text

    def test_no_errors_section_when_empty(self):
        text = format_run_summary(_make_summary(errors=[]))
        assert "Errors" not in text

    def test_error_truncation(self):
        long_error = "X" * 100
        text = format_run_summary(_make_summary(errors=[long_error]))
        assert "..." in text

    def test_max_five_errors_shown(self):
        text = format_run_summary(
            _make_summary(errors=[f"error {i}" for i in range(10)])
        )
        # Should show at most 5 error lines.
        assert text.count("- error") <= 5


# ---------------------------------------------------------------------------
# format_run_summary — JSON
# ---------------------------------------------------------------------------

class TestFormatRunSummaryJson:
    def test_valid_json(self):
        text = format_run_summary(_make_summary(), json_output=True)
        parsed = json.loads(text)
        assert parsed["outcome"] == "success"
        assert parsed["algo"] == "gspo"
        assert parsed["final_step"] == 200
        assert parsed["samples"]["trained"] == 950
        assert "metrics" in parsed

    def test_json_contains_all_fields(self):
        parsed = json.loads(
            format_run_summary(
                _make_summary(
                    outcome="error",
                    errors=["OOM"],
                    metrics={"loss": 1.0},
                ),
                json_output=True,
            )
        )
        expected_keys = {
            "outcome", "duration_s", "algo", "model",
            "final_step", "final_epoch", "metrics", "samples", "errors",
        }
        assert set(parsed.keys()) == expected_keys
        assert parsed["errors"] == ["OOM"]

    def test_json_empty_metrics(self):
        parsed = json.loads(
            format_run_summary(_make_summary(metrics={}), json_output=True)
        )
        assert parsed["metrics"] == {}


# ---------------------------------------------------------------------------
# print_run_summary
# ---------------------------------------------------------------------------

class TestPrintRunSummary:
    def test_prints_to_stream(self):
        buf = io.StringIO()
        print_run_summary(_make_summary(), stream=buf)
        output = buf.getvalue()
        assert "AReno Training Summary" in output
        assert "success" in output

    def test_disabled_does_nothing(self):
        buf = io.StringIO()
        print_run_summary(_make_summary(), enabled=False, stream=buf)
        assert buf.getvalue() == ""

    def test_json_mode(self):
        buf = io.StringIO()
        print_run_summary(_make_summary(), json_output=True, stream=buf)
        parsed = json.loads(buf.getvalue())
        assert parsed["outcome"] == "success"

    def test_defaults_to_stderr(self, capsys):
        print_run_summary(_make_summary())
        captured = capsys.readouterr()
        assert "AReno Training Summary" in captured.err
        assert captured.out == ""


# ---------------------------------------------------------------------------
# RunSummaryData defaults
# ---------------------------------------------------------------------------

class TestRunSummaryData:
    def test_defaults(self):
        data = RunSummaryData()
        assert data.outcome == "success"
        assert data.duration_s == 0.0
        assert data.final_step == 0
        assert data.metrics == {}
        assert data.errors == []
        assert data.samples_processed == 0
        assert data.samples_trained == 0
        assert data.samples_skipped == 0


# ---------------------------------------------------------------------------
# _format_float direct tests
# ---------------------------------------------------------------------------

class TestFormatFloat:
    def test_zero(self):
        from areno.cli.run_summary import _format_float
        assert _format_float(0) == "0"

    def test_integer(self):
        from areno.cli.run_summary import _format_float
        assert _format_float(42) == "42"

    def test_small_decimal(self):
        from areno.cli.run_summary import _format_float
        assert _format_float(0.0312) == "0.0312"

    def test_scientific(self):
        from areno.cli.run_summary import _format_float
        assert "e" in _format_float(1e-07).lower()

    def test_large_number(self):
        from areno.cli.run_summary import _format_float
        assert _format_float(751632384.0) == "7.51632e+08"


# ---------------------------------------------------------------------------
# Integration-style: end-to-end summary flow
# ---------------------------------------------------------------------------

class TestSummaryIntegration:
    """Verify the full flow: populate RunSummaryData → format → assert fields."""

    def test_success_flow(self):
        """Simulate a successful training run and verify summary output."""
        data = RunSummaryData(algo="sft", model="/path/to/model")
        data.outcome = "success"
        data.duration_s = 120.5
        data.final_step = 10
        data.final_epoch = 1
        data.samples_processed = 1000
        data.samples_trained = 950
        data.samples_skipped = 50
        data.metrics = {"loss": 0.123, "lr": 1e-5}
        data.errors = []

        text = format_run_summary(data)
        assert "success" in text
        assert "2m 00s" in text
        assert "sft" in text
        assert "10" in text
        assert "950 trained" in text
        assert "50 skipped" in text
        assert "1000 processed" in text
        assert "0.123" in text
        assert "Errors" not in text

    def test_error_flow(self):
        """Simulate a failed training run and verify summary output."""
        data = RunSummaryData(algo="ppo", model="/path/to/model")
        data.outcome = "error"
        data.duration_s = 5.2
        data.final_step = 1
        data.final_epoch = 0
        data.samples_processed = 8
        data.samples_trained = 8
        data.metrics = {"loss": 2.5}
        data.errors = ["RuntimeError: CUDA out of memory"]

        text = format_run_summary(data)
        assert "error" in text
        assert "5s" in text
        assert "RuntimeError" in text
        assert "CUDA" in text

    def test_json_flow(self):
        """Verify JSON output contains all expected fields."""
        import json as json_mod
        data = RunSummaryData(algo="dpo", model="/path/to/model")
        data.outcome = "success"
        data.duration_s = 60.0
        data.final_step = 5
        data.final_epoch = 0
        data.samples_processed = 100
        data.samples_trained = 100
        data.metrics = {"loss": 0.5}
        data.errors = []

        out = format_run_summary(data, json_output=True)
        parsed = json_mod.loads(out)
        assert parsed["outcome"] == "success"
        assert parsed["algo"] == "dpo"
        assert parsed["final_step"] == 5
        assert parsed["samples"]["trained"] == 100
        assert parsed["metrics"]["loss"] == 0.5
        assert parsed["errors"] == []

    def test_disabled_summary(self):
        """When enabled=False, nothing should be printed."""
        import io
        data = RunSummaryData(algo="sft", model="test")
        stream = io.StringIO()
        print_run_summary(data, enabled=False, stream=stream)
        assert stream.getvalue() == ""

    def test_nan_metric_filtered_in_json(self):
        """NaN metrics should be filtered out in JSON mode."""
        import json as json_mod
        data = RunSummaryData(algo="sft", model="test")
        data.metrics = {"loss": float("nan"), "lr": 1e-5}
        out = format_run_summary(data, json_output=True)
        parsed = json_mod.loads(out)
        assert "loss" not in parsed["metrics"]
        assert "lr" in parsed["metrics"]