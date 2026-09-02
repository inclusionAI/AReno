from __future__ import annotations

import asyncio
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "terminal_hacking"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=lambda **kwargs: SimpleNamespace(**kwargs),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"terminal_hacking_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game
        if name == "run_agent":
            sys.modules.pop("areno.api.agentic", None)
            if previous_agentic is not None:
                sys.modules["areno.api.agentic"] = previous_agentic


def _record(seed: int = 9):
    generator = _load_module("dataset_generator")
    return generator.generate_records(1, seed=seed, workers=1)[0]


def test_generator_and_session_are_reproducible_with_probes():
    generator = _load_module("dataset_generator")
    game = _load_module("game")
    rows = generator.generate_records(32, seed=17, workers=1)
    assert rows == generator.generate_records(32, seed=17, workers=1)
    assert all(len(row["probes"]) == 3 for row in rows)
    for row in rows:
        first = game.TerminalHackingSession(row)
        second = game.TerminalHackingSession(row)
        assert first.public_state() == second.public_state()
        assert len(first.public_state()["available_probes"]) == 3
        assert row["password"] in first.consistent_candidates()


def test_candidate_tool_and_parser_accept_one_valid_candidate_set():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    active = ["ACCESS", "BINARY", "BRIDGE"]
    tool = game.candidate_tool(active)
    parameters = tool["function"]["parameters"]
    assert tool["function"]["name"] == "submit_candidates"
    assert parameters["properties"]["candidates"]["items"]["enum"] == active
    assert parameters["properties"]["candidates"]["uniqueItems"] is True
    assistant = {
        "tool_calls": [
            {
                "function": {
                    "name": "submit_candidates",
                    "arguments": json.dumps({"candidates": ["BINARY", "BRIDGE"]}),
                }
            }
        ]
    }
    assert run_agent._parse_candidates(assistant, active) == ["BINARY", "BRIDGE"]
    assert run_agent._parse_candidates({"tool_calls": []}, active) is None
    assistant["tool_calls"][0]["function"]["arguments"] = json.dumps({"candidates": ["UNKNOWN"]})
    assert run_agent._parse_candidates(assistant, active) is None


def test_reward_heavily_penalizes_over_inclusion_and_lightly_penalizes_shrinkage():
    reward = _load_module("reward")
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    chosen = None
    for row in generator.generate_records(256, seed=31, workers=1):
        session = game.candidate_filter_session(row)
        expected = session.consistent_candidates()
        extras = [word for word in session.public_state()["active_candidates"] if word not in expected]
        if len(expected) >= 2 and extras:
            chosen = row, expected, extras
            break
    assert chosen is not None
    row, expected, extras = chosen

    def score(candidate_sets):
        calls = [
            {"name": "submit_candidates", "arguments": json.dumps({"candidates": candidates})}
            for candidates in candidate_sets
        ]
        return reward.reward_fn(SimpleNamespace(source_record=row, tool_calls=calls))

    assert score([expected]) == 1.0
    assert 0.9 <= score([expected[:-1]]) < 1.0
    assert score([expected + extras[:1]]) <= -0.5


def test_agent_rollout_is_single_turn_state_to_candidates():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _record()
    session = game.candidate_filter_session(record)

    class FakeCompletions:
        def __init__(self):
            self.requests = []

        async def create(self, **kwargs):
            self.requests.append(kwargs)
            expected = session.consistent_candidates()
            call = SimpleNamespace(
                id=f"call-{len(self.requests)}",
                type="function",
                function=SimpleNamespace(
                    name="submit_candidates",
                    arguments=json.dumps({"candidates": expected}),
                ),
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    item = SimpleNamespace(prompt=game.make_filter_prompt(record), record=record)
    turn = asyncio.run(run_agent._run_filter(item, client))
    assert turn is not None
    assert len(completions.requests) == 1
    for request in completions.requests:
        assert request["tool_choice"] == "required"
        assert [message["role"] for message in request["messages"]] == ["system", "user", "user"]
        assert all(message["role"] != "assistant" for message in request["messages"])


def test_rlvr_and_webui_share_candidate_protocol_without_displaying_candidates():
    web = _load_module("web_ui")
    assert "submit_candidates" not in web.INDEX_HTML

    record = _record(91)
    server = SimpleNamespace(
        record=record,
        session=web.TerminalHackingSession(record),
        rng=random.Random(1),
        args=SimpleNamespace(base_url=None),
        error=None,
        events=[],
    )
    web._agent_step(server, mode="algo")
    assert len(server.session.guessed) == 1
    assert server.error is None
