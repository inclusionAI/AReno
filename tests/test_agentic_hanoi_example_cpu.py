from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "hanoi"


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
        spec = importlib.util.spec_from_file_location(f"agentic_hanoi_{name}_for_tests", path)
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


def test_optimal_calculator_and_trace_replay_complete_every_disk_count():
    game = _load_module("game")

    for n in (3, 4, 5, 6):
        assert game.optimal_steps(n) == 2 ** n - 1
        moves = game.optimal_moves(n)
        assert len(moves) == game.optimal_steps(n)
        result = game.score_episode(n, moves)
        assert result["completed"] is True
        assert result["illegal"] is False
        assert result["excess"] == 0
        assert result["efficiency"] == 1.0
        assert result["reward"] == 1.0


def test_legal_moves_reject_empty_source_and_larger_on_smaller():
    game = _load_module("game")
    state = game.initial_state(3)

    # Pegs B and C are empty at the start: B -> C is an empty-source move.
    assert game.is_legal_move(state, "B", "C") is False
    assert game.is_legal_move(state, "A", "B") is True

    # A -> C moves disk 1, then A -> B moves disk 2. State: A=[3], B=[2], C=[1].
    state = game.apply_move(state, "A", "C")
    assert game.top_disk(state, "A") == 2
    state = game.apply_move(state, "A", "B")
    assert game.top_disk(state, "C") == 1
    # Placing disk 1 (C) on disk 2 (B) is legal (smaller on larger).
    assert game.is_legal_move(state, "C", "B") is True
    # Placing disk 2 (B) on disk 1 (C) is illegal (larger on smaller).
    assert game.is_legal_move(state, "B", "C") is False
    assert "smaller disk" in game.illegal_reason(state, "B", "C")
    with pytest.raises(ValueError):
        game.apply_move(state, "B", "C")


def test_score_completion_efficiency_and_illegal_termination():
    game = _load_module("game")
    n = 3
    optimal = game.optimal_moves(n)

    # Illegal move before completion terminates and scores 0.0.
    assert game.score_episode(n, [("B", "C")])["reward"] == 0.0
    assert game.score_episode(n, [("B", "C")])["illegal"] is True

    # One move short of completion: completed=False, reward 0.0, not illegal.
    short = game.score_episode(n, optimal[: len(optimal) - 1])
    assert short["completed"] is False
    assert short["illegal"] is False
    assert short["reward"] == 0.0

    # Optimal path scores 1.0; one extra redundant move still completes but scores less.
    assert game.score_episode(n, optimal)["reward"] == 1.0
    # Insert a legal redundant cycle A->B, B->A after the first optimal move.
    extended = [optimal[0], ("A", "B"), ("B", "A"), *optimal[1:]]
    result = game.score_episode(n, extended)
    assert result["completed"] is True
    assert result["excess"] == 2
    assert 0.5 < result["reward"] < 1.0


def test_generator_is_reproducible_and_within_disk_range():
    generator = _load_module("dataset_generator")

    records = generator.generate_records(32, seed=7)
    assert records == generator.generate_records(32, seed=7)
    assert len(records) == 32
    for record in records:
        assert record["n"] in (3, 4, 5, 6)
        assert record["max_moves"] >= game_optimal_steps(record["n"]) * 2
        assert "optimal" not in record  # the answer must not live on the record


def game_optimal_steps(n: int) -> int:
    game = _load_module("game")
    return game.optimal_steps(n)


def test_loader_validates_disks_and_attaches_prompt_without_leaking_solution():
    loader = _load_module("dataset_loader")
    game = _load_module("game")

    raw = [{"id": "t1", "n": 3}, {"id": "t2", "n": 5}]
    records = loader.load_training_dataset("unused", default_loader=lambda _: raw)
    assert [r["id"] for r in records] == ["t1", "t2"]
    for record in records:
        assert record["n"] in (3, 5)
        assert record["max_moves"] > 0
        assert f"{record['n']} disks" in record["prompt"]
        assert "Peg A" in record["prompt"]
        assert "2 ** n - 1" not in record["prompt"]

    # Out-of-range disk counts are rejected before any backend work.
    with pytest.raises(ValueError):
        loader.load_training_dataset("unused", default_loader=lambda _: [{"n": 2}])
    with pytest.raises(ValueError):
        loader.load_training_dataset("unused", default_loader=lambda _: [{"n": 7}])


def test_reward_replays_tool_calls_and_scores_illegal_and_malformed():
    reward = _load_module("reward")
    game = _load_module("game")
    n = 3
    optimal = game.optimal_moves(n)

    def score(moves):
        calls = [
            {"name": "move", "arguments": json.dumps({"source": src, "target": tgt})}
            for src, tgt in moves
        ]
        return reward.reward_fn(
            SimpleNamespace(source_record={"n": n}, tool_calls=calls)
        )

    assert score([]) == 0.0
    assert score(optimal) == 1.0
    assert score([("B", "C")]) == 0.0  # empty source is illegal
    assert score(optimal[:2]) == 0.0  # incomplete without illegal

    malformed = reward.reward_fn(
        SimpleNamespace(
            source_record={"n": n},
            tool_calls=[{"name": "move", "arguments": "not-json"}],
        )
    )
    assert malformed == 0.0


def test_tool_schema_is_closed_and_bounded_to_three_pegs():
    game = _load_module("game")
    parameters = game.MOVE_TOOL["function"]["parameters"]

    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["source", "target"]
    assert parameters["properties"]["source"]["enum"] == ["A", "B", "C"]
    assert parameters["properties"]["target"]["enum"] == ["A", "B", "C"]


def _move_response(source: str, target: str):
    call = SimpleNamespace(
        id=f"call-{source}{target}",
        type="function",
        function=SimpleNamespace(
            name="move", arguments=json.dumps({"source": source, "target": target})
        ),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _summary_response():
    message = SimpleNamespace(content="solved", tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeCompletions:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs["messages"])
        return next(self.responses)


def test_episode_replays_optimal_path_and_appends_clean_summary():
    run_agent = _load_module("run_agent")
    game = _load_module("game")
    n = 3
    optimal = game.optimal_moves(n)

    responses = [_move_response(src, tgt) for src, tgt in optimal]
    responses.append(_summary_response())
    completions = _FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    item = SimpleNamespace(
        prompt="solve it",
        record={"n": n, "max_moves": game.default_max_moves(n)},
    )

    turns = asyncio.run(run_agent._run_episode(item, client))

    # One turn per optimal move, plus the final non-tool summary turn.
    assert len(turns) == len(optimal) + 1
    assert all(getattr(turn, "tool_choice", None) is not None for turn in turns[:-1])
    assert getattr(turns[-1], "tool_choice", None) is None  # summary turn carries no tool
    assert getattr(turns[-1], "tools", None) in (None, [])

    finish_messages = completions.calls[-1]
    tool_results = [m for m in finish_messages if m.get("role") == "tool"]
    assert '"ok": true' in tool_results[-1]["content"]
    assert '"completed": true' in tool_results[-1]["content"]
    # The model was asked for exactly one move per turn plus the summary.
    assert len(completions.calls) == len(optimal) + 1


def test_episode_terminates_on_illegal_move_with_rejection_message():
    run_agent = _load_module("run_agent")
    game = _load_module("game")
    n = 3

    # From the start state, B is empty: move(B, C) is an illegal empty-source move.
    responses = [_move_response("B", "C"), _summary_response()]
    completions = _FakeCompletions(responses)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    item = SimpleNamespace(
        prompt="solve it",
        record={"n": n, "max_moves": game.default_max_moves(n)},
    )

    turns = asyncio.run(run_agent._run_episode(item, client))
    assert len(turns) == 2  # illegal move turn + summary turn
    tool_results = [m for m in completions.calls[-1] if m.get("role") == "tool"]
    assert '"ok": false' in tool_results[-1]["content"]
    assert "empty" in tool_results[-1]["content"]
    # Replaying the single emitted move through the reward scores 0.0.
    reward = _load_module("reward")
    score = reward.reward_fn(
        SimpleNamespace(
            source_record={"n": n},
            tool_calls=[{"name": "move", "arguments": json.dumps({"source": "B", "target": "C"})}],
        )
    )
    assert score == 0.0


def test_episode_breaks_when_model_returns_no_tool_call():
    run_agent = _load_module("run_agent")
    game = _load_module("game")
    n = 3

    completions = _FakeCompletions([_summary_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    item = SimpleNamespace(
        prompt="solve it",
        record={"n": n, "max_moves": game.default_max_moves(n)},
    )

    turns = asyncio.run(run_agent._run_episode(item, client))
    assert len(turns) == 1  # no executable move -> stop without a summary turn


def test_state_text_is_deterministic_and_reads_top_at_glance():
    game = _load_module("game")
    text = game.state_to_text(game.initial_state(3))
    assert text == "Peg A (bottom->top): 3 2 1\nPeg B (bottom->top): (empty)\nPeg C (bottom->top): (empty)"
