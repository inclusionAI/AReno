"""CPU tests for the elevator run_agent loop (examples/agentic/elevator/run_agent.py).

Uses a fake OpenAI client to verify the variable-length episode loop: natural
termination on delivery, early stop on ``done``, horizon cap, graceful stop
when the model emits no tool call, concurrent multi-episode dispatch, and that
tool transcripts carry the environment state back into the message history.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "elevator"


def _load_module(name: str):
    """Load an elevator example module without importing the areno/torch stack."""

    previous_game = sys.modules.pop("game", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    sys.path.insert(0, str(EXAMPLE_DIR))
    modname = f"agentic_elevator_{name}_for_tests"
    try:
        if name == "run_agent":
            sys.modules["areno.api.agentic"] = SimpleNamespace(
                AgentTrajectory=type("AgentTrajectory", (), {"__init__": lambda self, turns=None: setattr(self, "turns", turns or [])}),
                AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
            )
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
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


def _fake_response(name: str | None, args: dict | None = None, cid: int = 0):
    """Build a fake OpenAI-style response carrying one tool call (or none)."""

    if name is None:
        message = SimpleNamespace(content="I have no action", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])
    call = SimpleNamespace(
        id=f"call-{cid}",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args or {})),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Replays a scripted sequence of fake responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.all_messages = []

    async def create(self, **kwargs):
        self.calls += 1
        self.all_messages.append(kwargs["messages"])
        if self.responses:
            return self.responses.pop(0)
        return _fake_response(None)


def _fake_client(responses):
    return SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(responses)))


def _item(record, prompt=None):
    game = _load_module("game")
    return SimpleNamespace(record=record, prompt=prompt or game.make_prompt(record))


BASE = {"floors": 4, "capacity": 2, "horizon": 20, "scenario": "test",
        "passengers": [{"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0}]}


# [单测用例]测试场景：送达后自然终止，turn 数等于动作数
def test_natural_termination_on_delivery_records_one_turn_per_action():
    run_agent = _load_module("run_agent")
    actions = [
        _fake_response("open_door", cid=0),
        _fake_response("close_door", cid=1),
        _fake_response("move", {"direction": 1}, cid=2),
        _fake_response("move", {"direction": 1}, cid=3),
        _fake_response("open_door", cid=4),  # alight pid0 -> terminal
    ]
    client = _fake_client(actions)
    turns = asyncio.run(run_agent._run_episode(_item(BASE), client))
    assert len(turns) == 5
    assert client.chat.completions.calls == 5


# [单测用例]测试场景：done 动作提前结束
def test_done_action_ends_episode_early():
    run_agent = _load_module("run_agent")
    actions = [_fake_response("open_door", cid=0), _fake_response("done", cid=1)]
    client = _fake_client(actions)
    turns = asyncio.run(run_agent._run_episode(_item(BASE), client))
    assert len(turns) == 2


# [单测用例]测试场景：horizon 上限防止无限循环
def test_horizon_caps_episode_length():
    run_agent = _load_module("run_agent")
    base = {"floors": 3, "capacity": 1, "horizon": 3, "scenario": "terminate",
            "passengers": [{"pid": 0, "origin": 0, "dest": 2, "arrive_time": 0}]}
    # keep moving legally to exhaust horizon
    actions = [_fake_response("open_door", cid=0), _fake_response("close_door", cid=1)] + [
        _fake_response("move", {"direction": 1}, cid=i) for i in range(2, 10)
    ]
    client = _fake_client(actions)
    turns = asyncio.run(run_agent._run_episode(_item(base), client))
    assert len(turns) <= base["horizon"] + 2


# [单测用例]测试场景：模型无 tool call 时优雅停止并记录 1 个 turn
def test_no_tool_call_stops_gracefully_with_one_turn():
    run_agent = _load_module("run_agent")
    actions = [_fake_response(None)]  # no tool call
    client = _fake_client(actions)
    turns = asyncio.run(run_agent._run_episode(_item(BASE), client))
    assert len(turns) == 1


# [单测用例]测试场景：tool result 把环境状态回灌到消息历史
def test_tool_result_feeds_state_back_into_messages():
    run_agent = _load_module("run_agent")
    actions = [_fake_response("open_door", cid=0), _fake_response("done", cid=1)]
    client = _fake_client(actions)
    asyncio.run(run_agent._run_episode(_item(BASE), client))
    # second create() call's messages should contain a tool message with state
    second_messages = client.chat.completions.all_messages[1]
    tool_msgs = [m for m in second_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert "state" in payload
    assert "floor=0" in payload["state"]


# [单测用例]测试场景：多个 episode 并发执行
def test_concurrent_episodes_dispatch_via_run_agent():
    run_agent = _load_module("run_agent")
    base2 = {"floors": 3, "capacity": 1, "horizon": 10, "scenario": "test",
             "passengers": [{"pid": 0, "origin": 0, "dest": 1, "arrive_time": 0}]}

    async def run_both():
        c1 = _fake_client([_fake_response("open_door", cid=0), _fake_response("done", cid=1)])
        c2 = _fake_client([_fake_response("open_door", cid=0), _fake_response("done", cid=1)])
        grouped = await asyncio.gather(
            run_agent._run_episode(_item(BASE), c1),
            run_agent._run_episode(_item(base2), c2),
        )
        # AgentTrajectory was imported into run_agent's namespace while the stub was active
        return run_agent.AgentTrajectory(turns=[t for ep in grouped for t in ep])

    traj = asyncio.run(run_both())
    assert len(traj.turns) == 4


# [单测用例]测试场景：turn 携带 response/tools/tool_choice 字段
def test_turn_preserves_response_tools_and_tool_choice():
    run_agent = _load_module("run_agent")
    actions = [_fake_response("open_door", cid=0), _fake_response("done", cid=1)]
    client = _fake_client(actions)
    turns = asyncio.run(run_agent._run_episode(_item(BASE), client))
    first = turns[0]
    assert hasattr(first, "response")
    assert hasattr(first, "tools")
    assert first.tool_choice == "auto"
