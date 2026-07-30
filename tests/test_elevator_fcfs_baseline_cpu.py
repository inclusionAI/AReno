"""CPU tests for the FCFS baseline runner (examples/agentic/elevator/fcfs_baseline.py).

Asserts emitted metric fields and error messages rather than exit status only:
delivery rate bounds, overload refusal on overload scenarios, no invalid actions
on empty-door scenarios, low delivery on terminate scenarios, aggregation by
scenario, and JSON/human-readable output modes.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "elevator"


def _load_module(name: str):
    """Load an elevator example module without importing the areno/torch stack."""

    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    modname = f"agentic_elevator_{name}_for_tests"
    try:
        spec = importlib.util.spec_from_file_location(modname, EXAMPLE_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[modname] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        sys.modules.pop(modname, None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


# [单测用例]测试场景：聚合指标字段齐全且 delivery_rate 在 [0,1]
def test_aggregate_metrics_contain_all_required_fields():
    fcfs = _load_module("fcfs_baseline")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=20, seed=2026, scenario="mixed")
    metrics = fcfs.run_dataset(records)
    for field in ("episodes", "total_delivered", "total_passengers", "delivery_rate",
                  "mean_wait_per_passenger", "total_invalid_actions", "total_overload_refused", "by_scenario"):
        assert field in metrics, f"missing field {field}"
    assert metrics["episodes"] == 20
    assert 0.0 <= metrics["delivery_rate"] <= 1.0
    assert metrics["total_passengers"] >= metrics["total_delivered"]


# [单测用例]测试场景：overload 场景产生 overload_refused > 0
def test_overload_scenario_produces_refused_boardings():
    fcfs = _load_module("fcfs_baseline")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=8, seed=2026, scenario="overload")
    metrics = fcfs.run_dataset(records)
    assert metrics["total_overload_refused"] > 0, "overload scenario should refuse boardings"


# [单测用例]测试场景：empty_door 场景 FCFS 不产生非法动作
def test_empty_door_scenario_produces_no_invalid_actions():
    fcfs = _load_module("fcfs_baseline")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=8, seed=2026, scenario="empty_door")
    metrics = fcfs.run_dataset(records)
    assert metrics["total_invalid_actions"] == 0, f"FCFS knows door state, got {metrics['total_invalid_actions']}"


# [单测用例]测试场景：terminate 场景 delivery_rate < 1
def test_terminate_scenario_does_not_deliver_all():
    fcfs = _load_module("fcfs_baseline")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=8, seed=2026, scenario="terminate")
    metrics = fcfs.run_dataset(records)
    assert metrics["delivery_rate"] < 1.0, "terminate horizon too short to deliver everyone"


# [单测用例]测试场景：by_scenario 分组覆盖所有场景名
def test_by_scenario_groups_per_scenario_metrics():
    fcfs = _load_module("fcfs_baseline")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=30, seed=2026, scenario="mixed")
    metrics = fcfs.run_dataset(records)
    by = metrics["by_scenario"]
    assert isinstance(by, dict)
    for name, m in by.items():
        assert "episodes" in m and "delivery_rate" in m and "mean_wait_per_passenger" in m
        assert "invalid_actions" in m and "overload_refused" in m


# [单测用例]测试场景：空记录返回 episodes=0
def test_empty_records_return_zero_episodes():
    fcfs = _load_module("fcfs_baseline")
    metrics = fcfs.run_dataset([])
    assert metrics == {"episodes": 0}


# [单测用例]测试场景：_format_report 输出含关键字段
def test_format_report_contains_key_fields():
    fcfs = _load_module("fcfs_baseline")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=10, seed=2026, scenario="mixed")
    metrics = fcfs.run_dataset(records)
    report = fcfs._format_report(metrics)
    assert "Elevator FCFS baseline" in report
    assert "delivery_rate" in report
    assert "by_scenario" in report


# [单测用例]测试场景：_load_records 默认 fallback 到 generator
def test_load_records_defaults_to_generator_when_no_path():
    fcfs = _load_module("fcfs_baseline")
    records = fcfs._load_records(None)
    assert len(records) > 0
    assert all("passengers" in r for r in records)


# [单测用例]测试场景：_load_records 对不存在文件抛 FileNotFoundError
def test_load_records_raises_on_missing_file(tmp_path):
    fcfs = _load_module("fcfs_baseline")
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        fcfs._load_records(str(missing))
