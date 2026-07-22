from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "codebreaker"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    previous_agentic = sys.modules.get("areno.api.agentic")
    if name == "run_agent":
        sys.modules["areno.api.agentic"] = SimpleNamespace(
            AgentTrajectory=type("AgentTrajectory", (), {}),
            AgentTrajectoryTurn=lambda **kwargs: SimpleNamespace(**kwargs),
        )
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_codebreaker_{name}_for_tests", path)
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


def test_rules_validate_and_score_leading_zero_codes():
    game = _load_module("game")

    assert game.normalize_code("0123") == "0123"
    assert game.score_guess("0123", "0145") == {
        "valid": True,
        "guess": "0145",
        "exact": 2,
        "present": 0,
        "solved": False,
    }
    assert game.score_guess("0123", "3210")["present"] == 4
    assert game.score_guess("0123", "0012")["valid"] is False


def test_generator_is_reproducible_and_loader_hides_secret_from_prompt():
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")
    rows = generator.generate_records(8, seed=4)

    assert rows == generator.generate_records(8, seed=4)
    assert len({row["secret"] for row in rows}) == 8
    records = loader.load_training_dataset("unused", default_loader=lambda _: rows)
    assert all(record["secret"] not in record["prompt"] for record in records)
    assert all("at most 6 guesses" in record["prompt"] for record in records)


def test_agent_executes_strict_single_guess_and_does_not_fabricate_calls():
    run_agent = _load_module("run_agent")
    record = {"secret": "0123", "max_guesses": 6}
    valid = {
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "guess_code", "arguments": json.dumps({"code": "0145"})},
            }
        ]
    }

    assert run_agent._execute_guess(valid, record)["exact"] == 2
    assert run_agent._execute_guess({"tool_calls": []}, record) is None
    assert run_agent._execute_guess({"tool_calls": [valid["tool_calls"][0], valid["tool_calls"][0]]}, record) is None
    assert (
        run_agent._execute_guess(
            {"tool_calls": [{"function": {"name": "guess_code", "arguments": "not-json"}}]}, record
        )
        is None
    )


def test_reward_separates_invalid_partial_repeated_and_optimal_paths():
    reward = _load_module("reward")
    source = {"secret": "0123", "max_guesses": 6}

    def score(guesses):
        calls = [{"name": "guess_code", "arguments": json.dumps({"code": guess})} for guess in guesses]
        return reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=calls))

    assert score([]) == -1.0
    assert score(["0012"]) == -1.0
    assert score(["0145"]) > 0.0
    assert score(["0145", "0145"]) == -0.5
    assert score(["0123"]) == 1.0
    assert score(["4567", "0123"]) < 1.0


def test_tool_schema_is_closed_and_bounded_episode_defaults_to_six_turns():
    game = _load_module("game")
    parameters = game.GUESS_TOOL["function"]["parameters"]

    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["code"]
    assert parameters["properties"]["code"]["pattern"] == "^[0-9]{4}$"


def test_episode_preserves_tool_order_and_stops_on_success_or_parser_failure():
    run_agent = _load_module("run_agent")

    class FakeCompletions:
        def __init__(self, responses):
            self.responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self.responses)

    def response(code=None):
        calls = []
        if code is not None:
            calls = [
                SimpleNamespace(
                    id=f"call-{code}",
                    type="function",
                    function=SimpleNamespace(name="guess_code", arguments=json.dumps({"code": code})),
                )
            ]
        message = SimpleNamespace(content=None if calls else "I cannot guess", tool_calls=calls)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    item = SimpleNamespace(
        prompt="crack it",
        record={"secret": "0123", "max_guesses": 20},
    )
    completions = FakeCompletions([response("4567"), response("0123"), response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turns = asyncio.run(run_agent._run_episode(item, client))

    assert len(turns) == 3
    second_messages = completions.messages[1]
    assert second_messages[-3]["role"] == "assistant"
    assert second_messages[-2]["role"] == "tool"
    assert second_messages[-2]["tool_call_id"] == "call-4567"
    assert second_messages[-1]["role"] == "user"
    finish_messages = completions.messages[2]
    assert finish_messages[-3]["role"] == "assistant"
    assert finish_messages[-2]["role"] == "tool"
    assert finish_messages[-2]["tool_call_id"] == "call-0123"
    assert finish_messages[-1]["role"] == "user"

    failed = FakeCompletions([response()])
    failed_turns = asyncio.run(run_agent._run_episode(item, SimpleNamespace(chat=SimpleNamespace(completions=failed))))
    assert len(failed_turns) == 1


def test_tui_llm_mode_uses_openai_tool_call_and_solves(capsys):
    tui = _load_module("tui")
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            call = SimpleNamespace(
                id="call-win",
                type="function",
                function=SimpleNamespace(name="guess_code", arguments='{"code":"0123"}'),
            )
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[call]))])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    previous_openai = sys.modules.get("openai")
    sys.modules["openai"] = SimpleNamespace(OpenAI=FakeOpenAI)
    try:
        args = SimpleNamespace(
            max_guesses=6,
            base_url="http://127.0.0.1:8000/v1",
            api_key="token",
            model="policy",
        )
        tui._run_llm("0123", args)
    finally:
        sys.modules.pop("openai", None)
        if previous_openai is not None:
            sys.modules["openai"] = previous_openai

    assert captured["tool_choice"]["function"]["name"] == "guess_code"
    assert captured["tools"] == [tui.GUESS_TOOL]
    assert "ACCESS GRANTED" in capsys.readouterr().out
