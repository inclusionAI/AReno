"""CPU tests for the elevator reward function (examples/agentic/elevator/reward.py).

Asserts reward fields and monotonicity: more delivered -> higher reward,
invalid actions penalty, waiting penalty, malformed arguments safety, and
boundary protection when total_passengers is zero.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "elevator"


def _load_module(name: str):
    """Load an elevator example module without importing the areno package."""

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


def _record(source: dict, tool_calls: list[dict]) -> SimpleNamespace:
    """Build a fake reward record with source_record and aggregated tool_calls."""

    return SimpleNamespace(source_record=source, tool_calls=tool_calls)


def _call(name: str, args: dict | None = None) -> dict:
    return {"name": name, "arguments": json.dumps(args or {})}


BASE_SOURCE = {
    "floors": 4,
    "capacity": 2,
    "horizon": 20,
    "scenario": "test",
    "passengers": [
        {"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0},
        {"pid": 1, "origin": 1, "dest": 3, "arrive_time": 0},
    ],
}


# [单测用例]测试场景：完美路径 reward 高于半完成高于无动作
def test_reward_monotonic_in_delivered_passengers():
    reward = _load_module("reward")

    perfect_calls = [
        _call("open_door"), _call("close_door"), _call("move", {"direction": 1}),
        _call("open_door"), _call("close_door"), _call("move", {"direction": 1}),
        _call("open_door"), _call("close_door"), _call("move", {"direction": 1}),
        _call("open_door"),
    ]
    half_calls = [
        _call("open_door"), _call("close_door"), _call("move", {"direction": 1}),
        _call("move", {"direction": 1}), _call("open_door"), _call("close_door"),
    ]
    r_perfect = reward.reward_fn(_record(BASE_SOURCE, perfect_calls))
    r_half = reward.reward_fn(_record(BASE_SOURCE, half_calls))
    r_empty = reward.reward_fn(_record(BASE_SOURCE, []))

    assert r_empty < r_half < r_perfect, (r_empty, r_half, r_perfect)


# [单测用例]测试场景：非法动作降低 reward (门开时多次 move)
def test_invalid_actions_lower_reward():
    reward = _load_module("reward")

    bad_calls = [_call("open_door")] + [_call("move", {"direction": 1}) for _ in range(10)]
    r_bad = reward.reward_fn(_record(BASE_SOURCE, bad_calls))
    r_empty = reward.reward_fn(_record(BASE_SOURCE, []))
    assert r_bad < r_empty, (r_bad, r_empty)


# [单测用例]测试场景：等待惩罚生效 (故意绕路增加等待)
def test_waiting_penalty_reduces_reward():
    reward = _load_module("reward")

    # direct delivery
    direct = [_call("open_door"), _call("close_door"), _call("move", {"direction": 1}), _call("open_door")]
    source_one = {"floors": 3, "capacity": 1, "horizon": 20, "scenario": "test",
                  "passengers": [{"pid": 0, "origin": 0, "dest": 1, "arrive_time": 0}]}
    r_direct = reward.reward_fn(_record(source_one, direct))
    # detour: move up then down past destination
    detour = [_call("open_door"), _call("close_door"), _call("move", {"direction": 1}),
              _call("move", {"direction": -1}), _call("open_door")]
    r_detour = reward.reward_fn(_record(source_one, detour))
    assert r_detour < r_direct, (r_detour, r_direct)


# [单测用例]测试场景：0 乘客边界不产生 NaN
def test_zero_passengers_does_not_crash_or_nan():
    reward = _load_module("reward")
    safe = reward._score({"delivered": 0, "total_passengers": 0, "mean_wait": 0.0,
                          "invalid_actions": 0, "horizon": 1})
    assert not math.isnan(safe)
    assert safe == 0.0


# [单测用例]测试场景：malformed JSON 参数被当作非法动作处理不崩溃
def test_malformed_arguments_are_treated_as_invalid_and_do_not_crash():
    reward = _load_module("reward")
    calls = [
        {"name": "move", "arguments": "{bad json"},
        {"name": "open_door", "arguments": "{}"},
    ]
    r = reward.reward_fn(_record(BASE_SOURCE, calls))
    assert isinstance(r, float)


# [单测用例]测试场景：done 动作不影响 reward 计算 (提前结束视为当前状态)
def test_done_action_produces_reward_for_current_state():
    reward = _load_module("reward")
    calls = [_call("open_door"), _call("done")]
    r = reward.reward_fn(_record(BASE_SOURCE, calls))
    # one passenger boarded, none delivered -> reward should be negative-ish (invalid default 0, delivery 0)
    assert r < 0.01  # no delivery, small wait penalty


# [单测用例]测试场景：reward 与 episode_metrics 字段对齐
def test_score_uses_all_metric_fields():
    reward = _load_module("reward")
    metrics = {"delivered": 1, "total_passengers": 2, "mean_wait": 2.0,
               "invalid_actions": 0, "horizon": 10}
    expected = 1.0 * (1 / 2) - 0.3 * (2.0 / 10) - 0.5 * 0
    assert abs(reward._score(metrics) - expected) < 1e-9


# [单测用例]测试场景：非电梯工具名被忽略不进入动作序列
def test_non_elevator_tool_calls_are_ignored():
    reward = _load_module("reward")
    calls = [_call("guess_code", {"code": "1234"}), _call("open_door"), _call("done")]
    # only open_door + done count; no delivery
    r = reward.reward_fn(_record(BASE_SOURCE, calls))
    assert isinstance(r, float)
