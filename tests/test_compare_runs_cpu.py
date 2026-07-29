"""CPU tests for the compare_runs.py two-run comparison script."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Module loading helper -- load the script as an importable module
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".agents/skills/areno-profile-performance/scripts/compare_runs.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("compare_runs", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Inline fixtures
# ---------------------------------------------------------------------------


def _metric_stats(mean, median=None, min_val=None, max_val=None, last=None, count=10):
    """Build a metric stats dict similar to summarize_time_metrics output."""
    return {
        "count": count,
        "mean": mean,
        "median": median if median is not None else mean,
        "min": min_val if min_val is not None else mean,
        "max": max_val if max_val is not None else mean,
        "last": last if last is not None else mean,
    }


def _baseline_data():
    return {
        "log_dir": "/fake/baseline",
        "status": {"pid": 12345, "stage": "train", "status": "completed", "step": 100},
        "config": {
            "kind": "train",
            "pid": 12345,
            "settings": {
                "Basic": {
                    "title": "Basic",
                    "items": [
                        {"key": "algo", "value": "grpo"},
                        {"key": "batch_size", "value": 32},
                        {"key": "max_new_tokens", "value": 1024},
                    ],
                }
            },
        },
        "time_metrics": {
            "throughput": _metric_stats(1000.0, count=99),
            "time/rollout": _metric_stats(5.0),
            "time/train": _metric_stats(10.0),
        },
        "monitor_summary": {
            "gpus": {"0": {"memory_used_mib": {"max": 40000.0}}},
            "target_process_memory_mib": {"12345": {"max": 38000.0}},
        },
    }


def _candidate_data():
    return {
        "log_dir": "/fake/candidate",
        "status": {"pid": 67890, "stage": "train", "status": "running", "step": 50},
        "config": {
            "kind": "train",
            "pid": 67890,
            "settings": {
                "Basic": {
                    "title": "Basic",
                    "items": [
                        {"key": "algo", "value": "grpo"},
                        {"key": "batch_size", "value": 32},
                        {"key": "max_new_tokens", "value": 2048},
                        {"key": "custom_key", "value": "new_val"},
                    ],
                }
            },
        },
        "time_metrics": {
            "throughput": _metric_stats(1200.0, count=49),
            "time/rollout": _metric_stats(4.0),
            "time/train": _metric_stats(11.55),
        },
        "monitor_summary": {
            "gpus": {"0": {"memory_used_mib": {"max": 38000.0}}},
            "target_process_memory_mib": {"67890": {"max": 36000.0}},
        },
    }


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


def test_percent_change_basic():
    mod = _load_module()
    assert mod.percent_change(100.0, 120.0) == 20.0
    assert mod.percent_change(100.0, 80.0) == -20.0


def test_percent_change_with_none():
    mod = _load_module()
    assert mod.percent_change(None, 100.0) is None
    assert mod.percent_change(100.0, None) is None
    assert mod.percent_change(None, None) is None


def test_percent_change_zero_baseline():
    mod = _load_module()
    assert mod.percent_change(0.0, 100.0) is None
    assert mod.percent_change(0, 0) is None


def test_compare_metric_pair_complete():
    mod = _load_module()
    base = _metric_stats(100.0)
    cand = _metric_stats(120.0)
    result = mod.compare_metric_pair(base, cand)
    assert result["pct_change"]["mean"] == 20.0
    assert result["pct_change"]["median"] == 20.0
    assert result["baseline"] == base
    assert result["candidate"] == cand


def test_compare_metric_pair_one_missing():
    mod = _load_module()
    base = _metric_stats(100.0)
    result = mod.compare_metric_pair(base, None)
    assert result["candidate"] is None
    for stat in ("mean", "median", "min", "max", "last"):
        assert result["pct_change"][stat] is None

    result2 = mod.compare_metric_pair(None, base)
    assert result2["baseline"] is None
    assert result2["pct_change"]["mean"] is None


def test_compare_settings_finds_mismatch():
    mod = _load_module()
    base_settings = {"S": {"items": [{"key": "algo", "value": "grpo"}, {"key": "lr", "value": 1e-4}]}}
    cand_settings = {"S": {"items": [{"key": "algo", "value": "grpo"}, {"key": "lr", "value": 5e-5}]}}
    result = mod.compare_settings(base_settings, cand_settings)
    mismatched_keys = [m["key"] for m in result["mismatched"]]
    assert "lr" in mismatched_keys
    assert "algo" in result["matched"]


def test_compare_settings_only_in_one_run():
    mod = _load_module()
    base_settings = {"S": {"items": [{"key": "a", "value": 1}, {"key": "b", "value": 2}]}}
    cand_settings = {"S": {"items": [{"key": "a", "value": 1}, {"key": "c", "value": 3}]}}
    result = mod.compare_settings(base_settings, cand_settings)
    assert result["only_baseline"] == ["b"]
    assert result["only_candidate"] == ["c"]


def test_identify_extremes_both_directions():
    mod = _load_module()
    comparisons = {
        "throughput": {"pct_change": {"mean": 20.0}},
        "time/train": {"pct_change": {"mean": -15.0}},
        "time/rollout": {"pct_change": {"mean": 5.0}},
    }
    extremes = mod.identify_extremes(comparisons)
    # throughput +20% = improvement (throughput metric, positive = good)
    assert extremes["largest_improvement"]["metric"] == "throughput"
    assert extremes["largest_improvement"]["pct_change"] == 20.0
    # time/rollout +5% = regression (timing metric, positive = bad)
    # time/train -15% = improvement (timing metric, negative = good, magnitude 15 < 20)
    assert extremes["largest_regression"]["metric"] == "time/rollout"
    assert extremes["largest_regression"]["pct_change"] == 5.0


def test_identify_extremes_no_valid_entries():
    mod = _load_module()
    comparisons = {"metric": {"pct_change": {"mean": None}}}
    extremes = mod.identify_extremes(comparisons)
    assert extremes["largest_improvement"] is None
    assert extremes["largest_regression"] is None


def test_identify_extremes_only_positive():
    mod = _load_module()
    comparisons = {
        "throughput": {"pct_change": {"mean": 20.0}},
        "throughput2": {"pct_change": {"mean": 10.0}},
    }
    extremes = mod.identify_extremes(comparisons)
    assert extremes["largest_improvement"]["pct_change"] == 20.0
    assert extremes["largest_regression"] is None


def test_compare_peak_memory():
    mod = _load_module()
    base_monitor = {"gpus": {"0": {"memory_used_mib": {"max": 40000.0}}}}
    cand_monitor = {"gpus": {"0": {"memory_used_mib": {"max": 38000.0}}}}
    result = mod.compare_peak_memory(base_monitor, cand_monitor)
    assert result["baseline_peak_mib"] == 40000.0
    assert result["candidate_peak_mib"] == 38000.0
    assert result["pct_change"] == -5.0


def test_compare_peak_memory_none():
    mod = _load_module()
    result = mod.compare_peak_memory(None, None)
    assert result["baseline_peak_mib"] is None
    assert result["candidate_peak_mib"] is None
    assert result["pct_change"] is None


def test_compare_peak_memory_from_process_memory():
    mod = _load_module()
    monitor = {"gpus": {}, "target_process_memory_mib": {"123": {"max": 35000.0}}}
    result = mod.compare_peak_memory(monitor, None)
    assert result["baseline_peak_mib"] == 35000.0
    assert result["candidate_peak_mib"] is None
    assert result["pct_change"] is None


def test_build_result_full_fixture():
    mod = _load_module()
    result = mod.build_result(_baseline_data(), _candidate_data())
    assert result["ok"] is True
    assert result["baseline"]["status"] == "completed"
    assert result["candidate"]["status"] == "running"
    assert result["baseline"]["step"] == 100
    assert result["candidate"]["step"] == 50
    # Throughput improved 20%
    assert "throughput" in result["metrics"]
    assert result["metrics"]["throughput"]["pct_change"]["mean"] == 20.0
    # Time/train increased (+15.5%) = regression for timing metric
    # time/rollout decreased (-20%) = improvement for timing metric
    # throughput increased (+20%) = improvement
    assert result["extremes"]["largest_regression"]["metric"] == "time/train"
    # Settings: max_new_tokens mismatch
    mismatched_keys = [m["key"] for m in result["settings_comparison"]["mismatched"]]
    assert "max_new_tokens" in mismatched_keys
    # only_candidate has custom_key
    assert "custom_key" in result["settings_comparison"]["only_candidate"]
    # Peak memory
    assert result["peak_memory"]["baseline_peak_mib"] is not None
    assert result["peak_memory"]["pct_change"] is not None
    # Warnings: candidate running + config mismatch
    warnings_text = " ".join(result["warnings"])
    assert "candidate status is 'running'" in warnings_text
    assert "configuration mismatch" in warnings_text


def test_format_terminal_report_contains_sections():
    mod = _load_module()
    result = mod.build_result(_baseline_data(), _candidate_data())
    report = mod.format_terminal_report(result)
    assert "=== AReno Run Comparison ===" in report
    assert "Throughput & Timing Metrics" in report
    assert "Peak Memory" in report
    assert "Configuration" in report
    assert "Summary" in report
    assert "Warnings" in report


def test_format_terminal_report_highlights_mismatch():
    mod = _load_module()
    result = mod.build_result(_baseline_data(), _candidate_data())
    report = mod.format_terminal_report(result)
    assert "MISMATCH" in report
    assert "max_new_tokens" in report


def test_format_terminal_report_shows_regression():
    mod = _load_module()
    result = mod.build_result(_baseline_data(), _candidate_data())
    report = mod.format_terminal_report(result)
    assert "REGRESSION" in report
    assert "IMPROVEMENT" in report


def test_format_terminal_report_shows_na_for_missing():
    mod = _load_module()
    result = mod.build_result(_baseline_data(), _candidate_data())
    # Force a None metric
    result["metrics"]["missing_metric"] = {
        "baseline": None,
        "candidate": _metric_stats(50.0),
        "pct_change": {"mean": None, "median": None, "min": None, "max": None, "last": None},
    }
    report = mod.format_terminal_report(result)
    assert "N/A" in report


def test_none_values_never_converted_to_zero():
    """Ensure that None stays None throughout the pipeline."""
    mod = _load_module()
    base = _baseline_data()
    base["time_metrics"]["throughput"] = None  # simulate missing
    base["monitor_summary"] = None
    result = mod.build_result(base, _candidate_data())
    # throughput should not appear in metrics (baseline has None)
    assert "throughput" not in result["metrics"]
    # peak_memory should have None for baseline
    assert result["peak_memory"]["baseline_peak_mib"] is None
    assert result["peak_memory"]["pct_change"] is None
    # Warnings should mention missing peak
    warnings_text = " ".join(result["warnings"])
    assert "baseline peak memory unavailable" in warnings_text


def test_missing_metric_in_one_run_goes_to_warnings():
    mod = _load_module()
    base = _baseline_data()
    base["time_metrics"]["time/reward"] = _metric_stats(2.0)
    result = mod.build_result(base, _candidate_data())
    warnings_text = " ".join(result["warnings"])
    assert "time/reward" in warnings_text
    assert "only in baseline" in warnings_text


def test_active_run_status_in_warnings():
    mod = _load_module()
    base = _baseline_data()
    base["status"]["status"] = "running"
    result = mod.build_result(base, _candidate_data())
    warnings_text = " ".join(result["warnings"])
    assert "baseline status is 'running'" in warnings_text
    assert "candidate status is 'running'" in warnings_text


# ---------------------------------------------------------------------------
# End-to-end tests (subprocess + tmp_path)
# ---------------------------------------------------------------------------


def _create_run_dir(tmp_path, name, status="completed", step=100, settings_extra=None, pid=12345):
    """Create a minimal run directory with config and state files."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    settings_items = [
        {"key": "algo", "value": "grpo"},
        {"key": "batch_size", "value": 32},
    ]
    if settings_extra:
        settings_items.extend(settings_extra)
    config = {
        "kind": "train",
        "pid": pid,
        "summary_text": "test run",
        "settings": {"Basic": {"title": "Basic", "items": settings_items}},
    }
    (run_dir / f"areno_run_config.{pid}.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    state = {
        "pid": pid,
        "stage": "train",
        "status": status,
        "step": step,
        "updated_at": 1700000000,
    }
    (run_dir / f"dashboard_state.{pid}.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )
    return run_dir


def test_subprocess_end_to_end_success(tmp_path):
    base_dir = _create_run_dir(tmp_path, "baseline", status="completed", step=100, pid=12345)
    cand_dir = _create_run_dir(
        tmp_path, "candidate", status="completed", step=80, pid=67890,
        settings_extra=[{"key": "max_new_tokens", "value": 2048}],
    )
    # Add a matching setting to baseline so it's not only_candidate
    base_config = json.loads((base_dir / "areno_run_config.12345.json").read_text())
    base_config["settings"]["Basic"]["items"].append({"key": "max_new_tokens", "value": 1024})
    (base_dir / "areno_run_config.12345.json").write_text(json.dumps(base_config), encoding="utf-8")

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(cand_dir), "--json-only"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout)
    assert result["ok"] is True
    assert result["baseline"]["status"] == "completed"
    assert result["candidate"]["status"] == "completed"
    # Config mismatch detected
    mismatched_keys = [m["key"] for m in result["settings_comparison"]["mismatched"]]
    assert "max_new_tokens" in mismatched_keys
    # No time metrics (no TensorBoard files), but no crash
    assert result["metrics"] == {}


def test_subprocess_nonexistent_dir(tmp_path):
    base_dir = _create_run_dir(tmp_path, "baseline")
    fake_dir = tmp_path / "does_not_exist"

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(fake_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 1
    result = json.loads(process.stdout)
    assert result["ok"] is False
    assert "stage" in result
    assert result["stage"] == "input_validation"
    assert "does not exist" in result["error"]
    assert result["inputs"]["candidate"] == str(fake_dir)


def test_subprocess_not_a_directory(tmp_path):
    base_dir = _create_run_dir(tmp_path, "baseline")
    not_a_dir = tmp_path / "a_file.txt"
    not_a_dir.write_text("hello", encoding="utf-8")

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(not_a_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 1
    result = json.loads(process.stdout)
    assert result["ok"] is False
    assert result["stage"] == "input_validation"
    assert "not a directory" in result["error"]


def test_subprocess_help_works():
    process = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert process.returncode == 0
    assert "baseline" in process.stdout
    assert "candidate" in process.stdout


def test_subprocess_default_outputs_json_and_terminal(tmp_path):
    """Default mode should output JSON to stdout and terminal to stderr."""
    base_dir = _create_run_dir(tmp_path, "baseline", pid=11111)
    cand_dir = _create_run_dir(tmp_path, "candidate", pid=22222)

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(cand_dir)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0
    # stdout should be valid JSON
    result = json.loads(process.stdout)
    assert result["ok"] is True
    # stderr should contain the terminal report
    assert "=== AReno Run Comparison ===" in process.stderr


def test_subprocess_terminal_only(tmp_path):
    base_dir = _create_run_dir(tmp_path, "baseline", pid=11111)
    cand_dir = _create_run_dir(tmp_path, "candidate", pid=22222)

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(cand_dir), "--terminal-only"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0
    assert "=== AReno Run Comparison ===" in process.stdout
    # stdout should NOT be JSON
    assert not process.stdout.strip().startswith("{")


def test_subprocess_running_status_warning(tmp_path):
    base_dir = _create_run_dir(tmp_path, "baseline", status="completed", pid=11111)
    cand_dir = _create_run_dir(tmp_path, "candidate", status="running", pid=22222)

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(cand_dir), "--json-only"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0
    result = json.loads(process.stdout)
    warnings_text = " ".join(result["warnings"])
    assert "candidate status is 'running'" in warnings_text


def test_subprocess_monitor_jsonl(tmp_path):
    """Test that monitor JSONL files are loaded and peak memory extracted."""
    base_dir = _create_run_dir(tmp_path, "baseline", pid=11111)
    cand_dir = _create_run_dir(tmp_path, "candidate", pid=22222)

    # Write a GPU monitor JSONL
    monitor_lines = [
        json.dumps({"timestamp": 1, "gpus": [{"index": 0, "memory_used_mib": 30000.0}], "target_processes": [{"pid": 11111, "memory_used_mib": 28000.0}]}),
        json.dumps({"timestamp": 2, "gpus": [{"index": 0, "memory_used_mib": 40000.0}], "target_processes": [{"pid": 11111, "memory_used_mib": 35000.0}]}),
    ]
    (base_dir / "gpu_monitor.11111.jsonl").write_text("\n".join(monitor_lines), encoding="utf-8")

    monitor_lines_cand = [
        json.dumps({"timestamp": 1, "gpus": [{"index": 0, "memory_used_mib": 32000.0}], "target_processes": [{"pid": 22222, "memory_used_mib": 30000.0}]}),
        json.dumps({"timestamp": 2, "gpus": [{"index": 0, "memory_used_mib": 38000.0}], "target_processes": [{"pid": 22222, "memory_used_mib": 36000.0}]}),
    ]
    (cand_dir / "gpu_monitor.22222.jsonl").write_text("\n".join(monitor_lines_cand), encoding="utf-8")

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(cand_dir), "--json-only"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0
    result = json.loads(process.stdout)
    assert result["peak_memory"]["baseline_peak_mib"] is not None
    assert result["peak_memory"]["candidate_peak_mib"] is not None
    # Baseline peak should be 40000 (max GPU memory)
    assert result["peak_memory"]["baseline_peak_mib"] == 40000.0
    assert result["peak_memory"]["candidate_peak_mib"] == 38000.0


def test_subprocess_no_monitor_flag(tmp_path):
    """--no-monitor should skip monitor loading and produce None peak memory."""
    base_dir = _create_run_dir(tmp_path, "baseline", pid=11111)
    cand_dir = _create_run_dir(tmp_path, "candidate", pid=22222)
    (base_dir / "gpu_monitor.11111.jsonl").write_text(
        json.dumps({"gpus": [{"index": 0, "memory_used_mib": 40000.0}]}), encoding="utf-8"
    )

    process = subprocess.run(
        [sys.executable, str(_SCRIPT), str(base_dir), str(cand_dir), "--json-only", "--no-monitor"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == 0
    result = json.loads(process.stdout)
    assert result["peak_memory"]["baseline_peak_mib"] is None
    assert result["peak_memory"]["candidate_peak_mib"] is None