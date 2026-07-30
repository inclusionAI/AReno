from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from collections import deque
from pathlib import Path
from types import ModuleType, SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "warehouse"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    module_name = f"agentic_warehouse_{name}_for_tests"
    saved_modules = {key: sys.modules.get(key) for key in ("game", "areno", "areno.api", "areno.api.agentic")}
    if name == "run_agent":
        areno_module = ModuleType("areno")
        api_module = ModuleType("areno.api")

        class AgentTrajectory:
            def __init__(self, *, turns):
                self.turns = turns

        agentic_module = ModuleType("areno.api.agentic")
        agentic_module.AgentTrajectory = AgentTrajectory
        agentic_module.AgentTrajectoryTurn = lambda **kwargs: SimpleNamespace(**kwargs)
        sys.modules["areno"] = areno_module
        sys.modules["areno.api"] = api_module
        sys.modules["areno.api.agentic"] = agentic_module

    sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        sys.modules.pop(module_name, None)
        for key, previous in saved_modules.items():
            sys.modules.pop(key, None)
            if previous is not None:
                sys.modules[key] = previous


def _assert_value_error(fn, text: str) -> None:
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc)
    else:
        raise AssertionError("expected ValueError")


def _records(count: int = 3, *, seed: int = 2026) -> list[dict]:
    return _load_module("dataset_generator").generate_records(count, seed=seed)


def _path_to(state, target: str) -> list[str]:
    if state.agent_pos == target:
        return []
    visited = {state.agent_pos}
    queue = deque([(state.agent_pos, [])])
    while queue:
        node, path = queue.popleft()
        for neighbor in state.adjacency[node]:
            if neighbor == target:
                return [*path, neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, [*path, neighbor]))
    raise AssertionError(f"unreachable test shelf: {target}")


def _move_to(game, state, target: str) -> None:
    for shelf_id in _path_to(state, target):
        assert game.move(state, shelf_id).success


def _response(*calls, content=None):
    tool_calls = [
        SimpleNamespace(
            id=call["id"],
            type="function",
            function=SimpleNamespace(
                name=call["name"],
                arguments=json.dumps(call.get("arguments", {})),
            ),
        )
        for call in calls
    ]
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return next(self._responses)


def _fake_client(responses):
    completions = _FakeCompletions(responses)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        completions=completions,
    )


def _assistant_call(call_id: str, name: str, arguments: object) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Generator tests
# ---------------------------------------------------------------------------


def test_warehouse_generator_is_balanced_deterministic_and_satisfiable():
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(9, seed=7)

    assert records == generator.generate_records(9, seed=7)
    assert [record["difficulty"] for record in records[:3]] == [
        "small",
        "medium",
        "hard",
    ]
    for record in records:
        state = game.build_state(record)
        assert game.baseline_distance(state) >= 0
        assert state.target_shelf != ""
        for item in record["order"]:
            available = sum(shelf.stock.get(item["sku"], 0) for shelf in state.shelves.values())
            assert available >= item["qty"]


def test_warehouse_generator_rejects_non_positive_count():
    generator = _load_module("dataset_generator")
    _assert_value_error(lambda: generator.generate_records(0), "count must be")
    _assert_value_error(lambda: generator.generate_records(-1), "count must be")


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------


def test_warehouse_loader_uses_shared_defaults_and_target_shelf():
    loader = _load_module("dataset_loader")
    source = {
        "id": 1,
        "difficulty": "small",
        "seed": 42,
        "order": [{"sku": "S1", "qty": 1}],
    }

    record = loader.load_training_dataset(
        "unused",
        default_loader=lambda _: [source],
    )[0]

    assert record["rows"] == 2
    assert record["sku_pool"] == ["S1", "S2", "S3", "S4"]
    assert record["target_shelf"] != ""
    assert "move_to" in record["prompt"]
    assert "submit_order" in record["prompt"]
    assert record["target_shelf"] in record["prompt"]


def test_warehouse_loader_reports_row_and_invalid_field():
    loader = _load_module("dataset_loader")
    source = _records(1, seed=12)[0]
    source["start_shelf"] = "Z9"

    _assert_value_error(
        lambda: loader.load_training_dataset(
            "unused",
            default_loader=lambda _: [source],
        ),
        "warehouse dataset row 0: start_shelf",
    )


def test_warehouse_loader_rejects_an_unsatisfiable_order():
    loader = _load_module("dataset_loader")
    source = _records(1, seed=121)[0]
    source["order"][0]["qty"] = 10_000

    _assert_value_error(
        lambda: loader.load_training_dataset(
            "unused",
            default_loader=lambda _: [source],
        ),
        "warehouse dataset row 0: order SKU",
    )


# ---------------------------------------------------------------------------
# State tests
# ---------------------------------------------------------------------------


def test_warehouse_build_state_returns_independent_mutable_states():
    game = _load_module("game")
    record = _records(1, seed=13)[0]
    first = game.build_state(record)
    second = game.build_state(record)

    assert first is not second
    assert first.shelves is not second.shelves
    assert first.target_shelf == second.target_shelf
    assert first.completed is False
    assert second.completed is False


def test_warehouse_grid_is_connected_and_adjacency_is_symmetric():
    game = _load_module("game")
    state = game.build_state(_records(3, seed=14)[2])
    visited = {state.agent_pos}
    queue = deque([state.agent_pos])
    while queue:
        shelf_id = queue.popleft()
        for neighbor in state.adjacency[shelf_id]:
            assert shelf_id in state.adjacency[neighbor]
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    assert visited == set(state.shelves)


def test_warehouse_move_validates_unknown_and_unreachable_shelves():
    game = _load_module("game")
    state = game.build_state(_records(1, seed=15)[0])
    neighbor = state.adjacency[state.agent_pos][0]

    assert game.move(state, neighbor).success
    assert state.total_distance == 1
    assert not game.move(state, "Z9").success
    assert state.invalid_actions == 1

    non_neighbor = next(
        shelf_id
        for shelf_id in state.shelves
        if shelf_id != state.agent_pos and shelf_id not in state.adjacency[state.agent_pos]
    )
    assert not game.move(state, non_neighbor).success
    assert state.invalid_actions == 2


def _find_remote_record(seed: int = 999, count: int = 30) -> dict:
    """Find a record where the agent does not start on the target shelf."""
    game = _load_module("game")
    for candidate in _records(count, seed=seed):
        state = game.build_state(candidate)
        if state.agent_pos != state.target_shelf:
            return candidate
    raise AssertionError("no record with agent_pos != target_shelf found")


def test_warehouse_submit_requires_target_position():
    game = _load_module("game")
    record = _find_remote_record()
    state = game.build_state(record)
    baseline = game.baseline_distance(state)
    assert baseline > 0

    early = game.submit_order(state)
    assert not early.success
    assert early.data["stage"] == "completion_validation"
    assert state.invalid_actions == 1

    _move_to(game, state, state.target_shelf)
    completed = game.submit_order(state)
    assert completed.success
    assert state.completed is True
    assert state.invalid_actions == 1

    metrics = game.state_metrics(state, baseline=baseline)
    assert metrics["complete_orders"] == 1
    assert metrics["remaining_distance"] == 0
    assert set(metrics) == {
        "complete_orders",
        "invalid_actions",
        "distance",
        "baseline_distance",
        "remaining_distance",
        "progress",
    }


def test_warehouse_baseline_distance_is_bfs_to_target():
    game = _load_module("game")
    state = game.build_state(_records(1, seed=18)[0])

    expected = _bfs(state.adjacency, state.agent_pos, state.target_shelf)
    assert game.baseline_distance(state) == expected
    assert game.baseline_action_count(state) == expected + 1


def _bfs(adjacency, start, target):
    if start == target:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        for neighbor in adjacency.get(node, []):
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, dist + 1))
    return -1


# ---------------------------------------------------------------------------
# Execute action tests
# ---------------------------------------------------------------------------


def test_warehouse_execute_action_rejects_malformed_arguments_without_raising():
    game = _load_module("game")
    state = game.build_state(_records(1, seed=19)[0])

    array_result = game.execute_action(state, "move_to", [])
    missing_shelf = game.execute_action(state, "move_to", {"shelf_id": 123})
    extra_arg = game.execute_action(state, "submit_order", {"foo": "bar"})
    unknown_result = game.execute_action(state, "fly", {})

    assert not array_result.success
    assert array_result.data["input"] == "arguments"
    assert not missing_shelf.success
    assert missing_shelf.data["input"] == "shelf_id"
    assert not extra_arg.success
    assert not unknown_result.success
    assert state.invalid_actions == 4


# ---------------------------------------------------------------------------
# Agent tests
# ---------------------------------------------------------------------------


def test_warehouse_agent_tool_schemas_are_closed():
    run_agent = _load_module("run_agent")

    assert {tool["function"]["name"] for tool in run_agent.TOOLS} == {
        "move_to",
        "submit_order",
    }
    for tool in run_agent.TOOLS:
        assert tool["function"]["parameters"]["additionalProperties"] is False


def test_warehouse_agent_invalid_json_is_a_structured_failure():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    state = game.build_state(_records(1, seed=20)[0])
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "bad-json",
                "function": {
                    "name": "move_to",
                    "arguments": "{not-json}",
                },
            }
        ],
    }

    result = run_agent._execute_tool_call(assistant_message, state)

    assert not result.success
    assert result.data == {
        "stage": "tool_validation",
        "input": "arguments",
    }
    assert state.invalid_actions == 1


def test_warehouse_agent_tool_messages_match_call_ids():
    run_agent = _load_module("run_agent")
    assistant = {
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "move_to"},
            },
        ]
    }
    result = {"success": True, "message": "moved", "data": {}}

    messages = run_agent._tool_messages(assistant, result)

    assert len(messages) == 2
    assert messages[0] is assistant
    assert messages[1]["tool_call_id"] == "call-1"
    assert messages[1]["name"] == "move_to"


def test_warehouse_state_prompt_includes_target_and_turn():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    state = game.build_state(_records(1, seed=21)[0])

    move_prompt = run_agent.make_state_prompt(
        state, turn_number=2, turn_limit=5, is_submit_turn=False,
    )
    submit_prompt = run_agent.make_state_prompt(
        state, turn_number=5, turn_limit=5, is_submit_turn=True,
    )

    assert "Turn 2 of 5" in move_prompt
    assert state.target_shelf in move_prompt
    assert state.agent_pos in move_prompt
    assert "move_to" in move_prompt
    assert "Turn 5 of 5" in submit_prompt
    assert state.target_shelf in submit_prompt
    assert "submit_order" in submit_prompt


def test_warehouse_episode_navigates_and_submits():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _records(1, seed=22)[0]
    state = game.build_state(record)
    baseline = game.baseline_distance(state)
    turn_limit = baseline + 1

    path = _path_to(state, state.target_shelf)
    move_responses = [
        _response({"id": f"move-{i}", "name": "move_to", "arguments": {"shelf_id": shelf_id}})
        for i, shelf_id in enumerate(path)
    ]
    submit_response = [_response({"id": "submit", "name": "submit_order"})]
    item = SimpleNamespace(
        prompt="navigate to target",
        record=record,
        prompt_index=0,
        sample_index=0,
    )
    client = _fake_client([*move_responses, *submit_response])

    turns = asyncio.run(run_agent._run_episode(item, state, client))

    assert state.completed
    assert len(turns) == turn_limit

    for i, request in enumerate(client.completions.requests):
        if i < baseline:
            assert request["tool_choice"] == {"type": "function", "function": {"name": "move_to"}}
            assert len(request["tools"]) == 1
            assert request["tools"][0]["function"]["name"] == "move_to"
        else:
            assert request["tool_choice"] == {"type": "function", "function": {"name": "submit_order"}}
            assert len(request["tools"]) == 1
            assert request["tools"][0]["function"]["name"] == "submit_order"


def test_warehouse_episode_creates_dummy_on_missing_tool_call():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _find_remote_record(seed=23)
    state = game.build_state(record)
    baseline = game.baseline_distance(state)
    assert baseline > 0
    item = SimpleNamespace(
        prompt="navigate",
        record=record,
        prompt_index=0,
        sample_index=0,
    )
    responses = [_response(content="I cannot act.") for _ in range(baseline + 1)]
    client = _fake_client(responses)

    turns = asyncio.run(run_agent._run_episode(item, state, client))

    assert len(turns) == baseline + 1
    assert state.completed is False
    assert state.invalid_actions >= 1


def test_warehouse_concurrent_episodes_keep_state_isolated():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _records(1, seed=231)[0]
    first_state = game.build_state(record)
    second_state = game.build_state(record)
    first_baseline = game.baseline_distance(first_state)
    second_baseline = game.baseline_distance(second_state)
    item = SimpleNamespace(
        prompt="navigate",
        record=record,
        prompt_index=0,
        sample_index=0,
    )

    def make_responses():
        return [_response(content="no action") for _ in range(first_baseline + 1)]

    async def run_both():
        return await asyncio.gather(
            run_agent._run_episode(item, first_state, _fake_client(make_responses())),
            run_agent._run_episode(item, second_state, _fake_client(make_responses())),
        )

    asyncio.run(run_both())

    assert first_state.completed is False
    assert second_state.completed is False
    assert first_state.agent_pos == second_state.agent_pos


# ---------------------------------------------------------------------------
# Reward tests
# ---------------------------------------------------------------------------


def test_warehouse_reward_replays_optimal_partial_and_invalid_paths():
    game = _load_module("game")
    reward = _load_module("reward")
    record = _records(1, seed=25)[0]
    state = game.build_state(record)
    target = state.target_shelf
    path = _path_to(state, target)

    def move_calls(prefix):
        return [
            _assistant_call(f"{prefix}-{i}", "move_to", {"shelf_id": shelf_id})
            for i, shelf_id in enumerate(path)
        ]

    submit_call = _assistant_call("submit", "submit_order", {})
    optimal = SimpleNamespace(
        source_record=record,
        messages=[*move_calls("opt"), submit_call],
        tool_calls=[],
    )
    partial = SimpleNamespace(
        source_record=record,
        messages=[_assistant_call("p0", "move_to", {"shelf_id": path[0]})] if path else [],
        tool_calls=[],
    )
    invalid = SimpleNamespace(
        source_record=record,
        messages=[_assistant_call("inv", "move_to", {"shelf_id": "Z9"})],
        tool_calls=[],
    )

    assert reward.reward_fn(optimal) == 1.0
    assert reward.reward_fn(partial) < 0.0
    assert reward.reward_fn(invalid) < 0.0
    assert reward.reward_fn(optimal) > reward.reward_fn(partial)
    assert reward.reward_fn(partial) >= reward.reward_fn(invalid)


def test_warehouse_reward_grades_by_remaining_distance():
    game = _load_module("game")
    reward = _load_module("reward")
    record = _find_remote_record(seed=260)
    state = game.build_state(record)
    path = _path_to(state, state.target_shelf)
    assert len(path) >= 1

    wrong_neighbor = next(
        shelf_id
        for shelf_id in state.adjacency[state.agent_pos]
        if shelf_id != path[0]
    ) if len(state.adjacency[state.agent_pos]) > 1 else state.adjacency[state.agent_pos][0]

    def move_calls(prefix, shelves):
        return [
            _assistant_call(f"{prefix}-{i}", "move_to", {"shelf_id": shelf_id})
            for i, shelf_id in enumerate(shelves)
        ]

    submit_call = _assistant_call("submit", "submit_order", {})

    shortest = SimpleNamespace(
        source_record=record,
        messages=[*move_calls("short", path), submit_call],
        tool_calls=[],
    )
    close_no_submit = SimpleNamespace(
        source_record=record,
        messages=[*move_calls("close", path[:1])],
        tool_calls=[],
    )
    wrong_direction = SimpleNamespace(
        source_record=record,
        messages=[*_move_call_list("wrong", [wrong_neighbor])],
        tool_calls=[],
    )
    at_target_no_submit = SimpleNamespace(
        source_record=record,
        messages=[*move_calls("nosub", path)],
        tool_calls=[],
    )

    shortest_score = reward.reward_fn(shortest)
    close_score = reward.reward_fn(close_no_submit)
    wrong_score = reward.reward_fn(wrong_direction)
    nosub_score = reward.reward_fn(at_target_no_submit)

    assert shortest_score == 1.0
    assert nosub_score < shortest_score
    assert nosub_score > wrong_score
    assert close_score >= wrong_score
    assert all(-1.0 <= s <= 1.0 for s in [shortest_score, close_score, wrong_score, nosub_score])


def _move_call_list(prefix, shelves):
    return [
        _assistant_call(f"{prefix}-{i}", "move_to", {"shelf_id": shelf_id})
        for i, shelf_id in enumerate(shelves)
    ]