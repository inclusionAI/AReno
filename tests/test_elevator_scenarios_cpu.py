"""CPU tests for elevator scenario fixtures crossing game/loader/reward/baseline.

Integration-style tests that run each of the five acceptance scenarios
(overload, empty_door, concurrent, peak, terminate) through the full pipeline:
generator -> loader -> game environment -> reward, asserting scenario-specific
observable fields rather than exit status. Also verifies default behavior is
unchanged when the feature is not exercised (plain records still load).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _default_loader(rows):
    def _load(_path):
        for row in rows:
            yield row
    return _load


# [单测用例]测试场景：overload fixture 产生 overload_refused > 0
def test_overload_fixture_refuses_boarding_beyond_capacity():
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=6, seed=2026, scenario="overload")
    assert all(r["scenario"] == "overload" for r in records)
    # capacity is 1 but each record has >=4 passengers; opening at the busiest
    # floor must refuse at least one boarding.
    refused_seen = False
    for record in records:
        state = game.build_state(record)
        # move to the first floor that has waiting passengers, then open
        target = next(f for f in range(state.floors) if any(p.arrive_time <= state.time for p in state.waiting.get(f, [])))
        while state.floor != target:
            if state.door_open:
                game.step(state, {"name": "close_door"})
            direction = 1 if target > state.floor else -1
            game.step(state, {"name": "move", "direction": direction})
        result = game.step(state, {"name": "open_door"})
        if result["refused"] > 0:
            refused_seen = True
        assert state.overload_refused >= 0
    assert refused_seen, "overload fixtures must produce at least one refused boarding"


# [单测用例]测试场景：empty_door fixture 起始门开，open 立即非法
def test_empty_door_fixture_starts_with_open_door():
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=4, seed=2026, scenario="empty_door")
    for record in records:
        assert record["door_open"] is True
        state = game.build_state(record)
        assert state.door_open is True
        result = game.step(state, {"name": "open_door"})
        assert result["invalid"] is True


# [单测用例]测试场景：concurrent fixture 多层同时刻到达
def test_concurrent_fixture_has_multi_floor_simultaneous_arrivals():
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=4, seed=2026, scenario="concurrent")
    for record in records:
        t0_origins = {p["origin"] for p in record["passengers"] if p["arrive_time"] == 0}
        assert len(t0_origins) >= 2, "concurrent fixture should have >=2 floors with t=0 arrivals"


# [单测用例]测试场景：peak fixture 长 horizon 高密度
def test_peak_fixture_has_long_horizon_and_high_density():
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=4, seed=2026, scenario="peak")
    for record in records:
        assert record["horizon"] >= 64, "peak horizon should be long"
        assert len(record["passengers"]) >= 8, "peak should have high passenger density"


# [单测用例]测试场景：terminate fixture horizon 极短无法全送达
def test_terminate_fixture_cannot_deliver_all_within_horizon():
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(count=4, seed=2026, scenario="terminate")
    for record in records:
        assert record["horizon"] <= 4
        metrics = game.run_fcfs_episode(record)
        assert metrics["delivered"] < metrics["total_passengers"], "terminate should not deliver all"


# [单测用例]测试场景：loader 接受 generator 全场景并构建 prompt
def test_loader_accepts_all_scenarios_and_builds_prompts():
    loader = _load_module("dataset_loader")
    generator = _load_module("dataset_generator")
    for scenario in generator.SCENARIOS:
        rows = generator.generate_records(count=6, seed=2026, scenario=scenario)
        records = loader.load_training_dataset("unused", default_loader=_default_loader(rows))
        assert len(records) == 6
        assert all("prompt" in r and "passengers" in r for r in records)


# [单测用例]测试场景：reward 对全场景产出有限浮点
def test_reward_produces_finite_float_across_scenarios():
    import math
    reward = _load_module("reward")
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    for scenario in generator.SCENARIOS:
        rows = generator.generate_records(count=4, seed=2026, scenario=scenario)
        for source in rows:
            # run a short FCFS episode to get plausible tool calls, then score
            state = game.build_state(source)
            actions = []
            for _ in range(min(source["horizon"], 8)):
                if game.is_terminal(state):
                    break
                action = game.fcfs_policy(state)
                if action["name"] == "done":
                    break
                actions.append(action)
                game.step(state, action)
            calls = [{"name": a["name"], "arguments": json.dumps({k: v for k, v in a.items() if k != "name"})}
                     for a in actions]
            r = reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=calls))
            assert isinstance(r, float)
            assert not math.isnan(r) and not math.isinf(r)


# [单测用例]测试场景：默认行为不变——无 scenario 字段的记录仍可加载
def test_default_behavior_unchanged_without_scenario_field():
    loader = _load_module("dataset_loader")
    rows = [{"floors": 3, "capacity": 1, "horizon": 10,
             "passengers": [{"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0}]}]
    records = loader.load_training_dataset("unused", default_loader=_default_loader(rows))
    assert records[0]["scenario"] == "mixed"  # default fills missing scenario
    assert "prompt" in records[0]
