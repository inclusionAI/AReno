"""CPU tests for the elevator dispatch environment core (examples/agentic/elevator/game.py).

Covers legal state transitions, invalid actions, overload refusal, empty-door
invalid actions, termination by horizon, done action, FCFS baseline, and
deterministic state serialization. No GPU, network, or external services.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "elevator"


def _load_game():
    """Load the elevator game module without importing the areno package."""

    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    modname = "agentic_elevator_game_for_tests"
    try:
        spec = importlib.util.spec_from_file_location(modname, EXAMPLE_DIR / "game.py")
        module = importlib.util.module_from_spec(spec)
        # register before exec so dataclass decorators can resolve __module__
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


# [单测用例]测试场景：合法路径顺路接送 + 非法动作门未关时移动
def test_legal_path_delivers_two_passengers_and_rejects_move_with_door_open():
    game = _load_game()
    record = {
        "floors": 4,
        "capacity": 2,
        "horizon": 20,
        "scenario": "test",
        "passengers": [
            {"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0},
            {"pid": 1, "origin": 1, "dest": 3, "arrive_time": 0},
        ],
    }
    state = game.build_state(record)
    assert state.total_passengers() == 2
    assert state.floor == 0 and not state.door_open

    game.step(state, {"name": "open_door"})  # board pid0
    assert state.total_passengers() == 2  # total does not shrink on board
    # move while door open is invalid
    result = game.step(state, {"name": "move", "direction": 1})
    assert result["invalid"] is True
    assert result["error"] == "cannot move while door is open"
    assert state.invalid_actions == 1
    assert state.floor == 0  # unchanged

    game.step(state, {"name": "close_door"})
    game.step(state, {"name": "move", "direction": 1})  # floor 1
    assert state.floor == 1
    result = game.step(state, {"name": "open_door"})  # board pid1
    assert result["boarded"] == 1
    game.step(state, {"name": "close_door"})
    game.step(state, {"name": "move", "direction": 1})  # floor 2
    result = game.step(state, {"name": "open_door"})  # alight pid0 (dest=2)
    assert result["alighted"] == 1
    assert state.delivered == 1
    game.step(state, {"name": "close_door"})
    game.step(state, {"name": "move", "direction": 1})  # floor 3
    game.step(state, {"name": "open_door"})  # alight pid1 (dest=3)
    assert state.delivered == 2
    assert game.is_terminal(state)


# [单测用例]测试场景：过载防护，capacity 严格小于同时刻候梯人数
def test_overload_refuses_boarding_beyond_capacity():
    game = _load_game()
    record = {
        "floors": 3,
        "capacity": 1,
        "horizon": 30,
        "scenario": "overload",
        "passengers": [{"pid": i, "origin": 0, "dest": 2, "arrive_time": 0} for i in range(3)],
    }
    state = game.build_state(record)
    result = game.step(state, {"name": "open_door"})
    assert result["boarded"] == 1
    assert result["refused"] == 2
    assert state.overload_refused == 2
    assert state.delivered == 0


# [单测用例]测试场景：空门操作，door 已开时 open 非法、已关时 close 非法
def test_empty_door_state_rejects_redundant_open_and_close():
    game = _load_game()
    record = {
        "floors": 3,
        "capacity": 2,
        "horizon": 10,
        "door_open": True,
        "scenario": "empty_door",
        "passengers": [{"pid": 0, "origin": 0, "dest": 1, "arrive_time": 0}],
    }
    state = game.build_state(record)
    assert state.door_open is True

    # open while already open -> invalid
    result = game.step(state, {"name": "open_door"})
    assert result["invalid"] is True
    assert state.invalid_actions == 1

    # close, then close again -> invalid
    game.step(state, {"name": "close_door"})
    assert state.door_open is False
    result = game.step(state, {"name": "close_door"})
    assert result["invalid"] is True
    assert result["error"] == "door already closed"
    assert state.invalid_actions == 2


# [单测用例]测试场景：terminate 终止条件，horizon 极短强制结束
def test_horizon_termination_bounds_episode_length():
    game = _load_game()
    record = {
        "floors": 3,
        "capacity": 1,
        "horizon": 2,
        "scenario": "terminate",
        "passengers": [{"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0}],
    }
    state = game.build_state(record)
    game.step(state, {"name": "open_door"})
    assert not game.is_terminal(state)
    game.step(state, {"name": "close_door"})
    assert game.is_terminal(state)  # time reached horizon
    metrics = game.episode_metrics(state)
    assert metrics["time"] == 2
    assert metrics["delivered"] == 0  # could not deliver within 2 steps


# [单测用例]测试场景：done 动作主动终止
def test_done_action_terminates_episode_immediately():
    game = _load_game()
    record = {
        "floors": 3,
        "capacity": 1,
        "horizon": 10,
        "scenario": "test",
        "passengers": [{"pid": 0, "origin": 0, "dest": 1, "arrive_time": 0}],
    }
    state = game.build_state(record)
    result = game.step(state, {"name": "done"})
    assert result["done"] is True
    assert state.terminated is True
    assert game.is_terminal(state)


# [单测用例]测试场景：越界移动非法
def test_move_out_of_bounds_is_invalid():
    game = _load_game()
    state = game.build_state({"floors": 3, "capacity": 1, "horizon": 10,
                              "passengers": [{"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0}]})
    result = game.step(state, {"name": "move", "direction": -1})  # floor 0 -> -1
    assert result["invalid"] is True
    assert "out of range" in result["error"]
    assert state.invalid_actions == 1
    assert state.floor == 0


# [单测用例]测试场景：未知动作非法
def test_unknown_action_is_invalid():
    game = _load_game()
    state = game.build_state({"floors": 3, "capacity": 1, "horizon": 10,
                              "passengers": [{"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0}]})
    result = game.step(state, {"name": "fly"})
    assert result["invalid"] is True
    assert "unknown action" in result["error"]


# [单测用例]测试场景：配置校验拒绝非法 floors/capacity/horizon
def test_validate_config_rejects_invalid_values():
    game = _load_game()
    with pytest.raises(ValueError, match="floors"):
        game.validate_config(floors=1, capacity=1, horizon=10)
    with pytest.raises(ValueError, match="capacity"):
        game.validate_config(floors=3, capacity=0, horizon=10)
    with pytest.raises(ValueError, match="horizon"):
        game.validate_config(floors=3, capacity=1, horizon=0)


# [单测用例]测试场景：build_state 拒绝 origin==dest 与越界乘客
def test_build_state_rejects_bad_passengers():
    game = _load_game()
    with pytest.raises(ValueError, match="origin equals dest"):
        game.build_state({"floors": 3, "capacity": 1, "horizon": 5,
                          "passengers": [{"pid": 0, "origin": 1, "dest": 1, "arrive_time": 0}]})
    with pytest.raises(ValueError, match="out of range"):
        game.build_state({"floors": 3, "capacity": 1, "horizon": 5,
                          "passengers": [{"pid": 0, "origin": 5, "dest": 0, "arrive_time": 0}]})


# [单测用例]测试场景：FCFS 基线在简单场景送达所有乘客
def test_fcfs_baseline_delivers_all_on_simple_record():
    game = _load_game()
    record = {
        "floors": 4,
        "capacity": 2,
        "horizon": 20,
        "scenario": "test",
        "passengers": [
            {"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0},
            {"pid": 1, "origin": 1, "dest": 3, "arrive_time": 0},
        ],
    }
    metrics = game.run_fcfs_episode(record)
    assert metrics["delivered"] == 2
    assert metrics["total_passengers"] == 2
    assert metrics["invalid_actions"] == 0
    assert metrics["mean_wait"] >= 0


# [单测用例]测试场景：format_state 输出含关键字段且可读
def test_format_state_contains_key_fields():
    game = _load_game()
    state = game.build_state({"floors": 4, "capacity": 2, "horizon": 20,
                              "passengers": [{"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0}]})
    text = game.format_state(state)
    assert "floor=0/3" in text
    assert "door=CLOSED" in text
    assert "delivered=0/1" in text
    assert "time=0/20" in text


# [单测用例]测试场景：parse_action 从 OpenAI 消息对象提取动作
def test_parse_action_extracts_first_tool_call_from_message():
    game = _load_game()
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(
                id="c1",
                type="function",
                function=SimpleNamespace(name="move", arguments='{"direction": 1}'),
            )
        ]
    )
    action = game.parse_action(message)
    assert action == {"name": "move", "direction": 1}

    # empty tool calls -> None
    assert game.parse_action(SimpleNamespace(tool_calls=[])) is None
    assert game.parse_action(SimpleNamespace(tool_calls=None)) is None


# [单测用例]测试场景：parse_action 容忍非法 JSON 参数
def test_parse_action_tolerates_malformed_arguments():
    game = _load_game()
    message = SimpleNamespace(
        tool_calls=[
            SimpleNamespace(id="c1", type="function",
                            function=SimpleNamespace(name="move", arguments="{bad json"))
        ]
    )
    action = game.parse_action(message)
    assert action["name"] == "move"  # args default to {}
    assert action.get("direction") is None
