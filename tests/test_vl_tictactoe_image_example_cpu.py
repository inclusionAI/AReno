from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / "vl" / "tictactoe_image" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"vl_tictactoe_image_{name}_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vl_tictactoe_agent_declares_choose_square_tool():
    run_agent = _load_module("run_agent")

    tool = run_agent.CHOOSE_SQUARE_TOOL

    assert tool["type"] == "function"
    assert tool["function"]["name"] == "choose_square"
    assert tool["function"]["parameters"]["required"] == ["square"]


def test_vl_tictactoe_reward_scores_normalized_tool_square():
    reward = _load_module("reward")
    board = [["X", "X", "."], ["O", ".", "."], ["O", ".", "."]]
    record = SimpleNamespace(
        source_record={"board": board},
        completion="square 1",
        tool_calls=[{"name": "choose_square", "arguments": {"square": 3}}],
    )

    assert reward.reward_fn(record) == 1.0

    record.tool_calls = [{"name": "choose_square", "arguments": '{"square": 1}'}]
    assert reward.reward_fn(record) == -1.0
