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


def _fulfill_order(game, state) -> None:
    for item in state.order:
        remaining = item["qty"]
        for shelf_id, shelf in state.shelves.items():
            available = shelf.stock.get(item["sku"], 0)
            if available <= 0:
                continue
            _move_to(game, state, shelf_id)
            quantity = min(available, remaining)
            assert game.pick(state, item["sku"], quantity).success
            remaining -= quantity
            if remaining == 0:
                break
        assert remaining == 0


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
        for item in record["order"]:
            available = sum(shelf.stock.get(item["sku"], 0) for shelf in state.shelves.values())
            assert available >= item["qty"]


def test_warehouse_generator_rejects_non_positive_count():
    generator = _load_module("dataset_generator")
    _assert_value_error(lambda: generator.generate_records(0), "count must be")
    _assert_value_error(lambda: generator.generate_records(-1), "count must be")


def test_warehouse_loader_uses_shared_defaults_and_current_tool_names():
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
    assert "check_shelf" in record["prompt"]
    assert "pick_item" in record["prompt"]
    assert "pick_from_shelf" not in record["prompt"]


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


def test_warehouse_build_state_returns_independent_mutable_states():
    game = _load_module("game")
    record = _records(1, seed=13)[0]
    first = game.build_state(record)
    second = game.build_state(record)
    sku, quantity = next(iter(first.shelves["A1"].stock.items()))

    assert first is not second
    assert first.shelves is not second.shelves
    assert game.pick(first, sku, 1).success
    assert first.cart == {sku: 1}
    assert second.cart == {}
    assert second.shelves["A1"].stock[sku] == quantity


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


def test_warehouse_pick_tracks_wrong_item_and_out_of_stock():
    game = _load_module("game")
    state = game.build_state(_records(1, seed=16)[0])
    order_skus = {item["sku"] for item in state.order}
    wrong = next(
        (shelf_id, sku, quantity)
        for shelf_id, shelf in state.shelves.items()
        for sku, quantity in shelf.stock.items()
        if sku not in order_skus
    )
    _move_to(game, state, wrong[0])

    result = game.pick(state, wrong[1], 1)
    assert result.success
    assert result.data["mistake"] is True
    assert state.picking_errors == 1

    result = game.pick(state, wrong[1], wrong[2] + 1)
    assert not result.success
    assert result.data["stage"] == "stock_validation"
    assert state.picking_errors == 2


def test_warehouse_submit_requires_exact_cart_and_emits_metrics():
    game = _load_module("game")
    state = game.build_state(_records(1, seed=17)[0])
    baseline = game.baseline_distance(state)

    early = game.submit_order(state)
    assert not early.success
    assert early.data["stage"] == "completion_validation"
    assert state.invalid_actions == 1

    _fulfill_order(game, state)
    completed = game.submit_order(state)
    metrics = game.state_metrics(state, baseline=baseline)
    assert completed.success
    assert metrics["complete_orders"] == 1
    assert metrics["baseline_distance"] == baseline
    assert set(metrics) == {
        "complete_orders",
        "picking_mistakes",
        "invalid_actions",
        "distance",
        "baseline_distance",
        "distance_ratio",
        "distance_efficiency",
        "cart_progress",
    }


def test_warehouse_baseline_supports_quantities_split_across_shelves():
    game = _load_module("game")
    state = game.build_state(_records(1, seed=18)[0])
    for shelf in state.shelves.values():
        shelf.stock.pop("SPLIT", None)
    state.shelves["A1"].stock["SPLIT"] = 1
    state.shelves["B2"].stock["SPLIT"] = 1
    state.order = [{"sku": "SPLIT", "qty": 2}]

    assert game.baseline_distance(state) == 2


def test_warehouse_execute_action_rejects_malformed_arguments_without_raising():
    game = _load_module("game")
    state = game.build_state(_records(1, seed=19)[0])

    array_result = game.execute_action(state, "pick_item", [])
    string_qty_result = game.execute_action(
        state,
        "pick_item",
        {"sku": "S1", "qty": "one"},
    )
    unknown_result = game.execute_action(state, "fly", {})

    assert not array_result.success
    assert array_result.data["input"] == "arguments"
    assert not string_qty_result.success
    assert string_qty_result.data["input"] == "qty"
    assert not unknown_result.success
    assert state.invalid_actions == 3


def test_warehouse_agent_tool_schemas_are_closed():
    run_agent = _load_module("run_agent")

    assert {tool["function"]["name"] for tool in run_agent.TOOLS} == {
        "move_to",
        "check_shelf",
        "pick_item",
        "submit_order",
    }
    for tool in run_agent.TOOLS:
        assert tool["function"]["parameters"]["additionalProperties"] is False


def test_warehouse_agent_invalid_json_is_a_structured_failure():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    state = game.build_state(_records(1, seed=20)[0])
    call = {
        "id": "bad-json",
        "function": {
            "name": "pick_item",
            "arguments": "{not-json}",
        },
    }

    result = run_agent._execute_tool_call(call, state)

    assert not result.success
    assert result.data == {
        "stage": "tool_validation",
        "input": "arguments",
    }
    assert state.invalid_actions == 1


def test_warehouse_agent_tool_results_match_every_call_id():
    run_agent = _load_module("run_agent")
    assistant = {
        "tool_calls": [
            {
                "id": "call-1",
                "function": {"name": "check_shelf"},
            },
            {
                "id": "call-2",
                "function": {"name": "submit_order"},
            },
        ]
    }
    results = [{"success": False}, {"success": False}]

    messages = run_agent._tool_messages(assistant, results)

    assert [message["tool_call_id"] for message in messages] == [
        "call-1",
        "call-2",
    ]
    _assert_value_error(
        lambda: run_agent._tool_messages(assistant, results[:1]),
        "does not match",
    )


def test_warehouse_state_prompt_tracks_partial_quantities():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    state = game.build_state(_records(1, seed=21)[0])
    item = state.order[0]
    state.cart[item["sku"]] = item["qty"] - 1

    prompt = run_agent.make_state_prompt(state, turn_number=2)

    assert f"{item['sku']} x1" in prompt
    assert "Action turn 2 of 6" in prompt


def test_warehouse_episode_preserves_history_and_final_tool_result():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _records(1, seed=22)[0]
    state = game.build_state(record)
    sku = next(iter(state.shelves["A1"].stock))
    state.order = [{"sku": sku, "qty": 1}]
    item = SimpleNamespace(
        prompt="pick one item",
        record=record,
        prompt_index=0,
        sample_index=0,
    )
    client = _fake_client(
        [
            _response({"id": "check", "name": "check_shelf"}),
            _response(
                {
                    "id": "pick",
                    "name": "pick_item",
                    "arguments": {"sku": sku, "qty": 1},
                }
            ),
            _response({"id": "submit", "name": "submit_order"}),
            _response(content="The order was completed."),
        ]
    )

    turns = asyncio.run(run_agent._run_episode(item, state, client))

    assert state.completed
    assert len(turns) == 4
    second_messages = client.completions.requests[1]["messages"]
    assert [message["role"] for message in second_messages[-4:]] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    final_messages = client.completions.requests[-1]["messages"]
    assert [message["role"] for message in final_messages[-4:]] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    submit_result = json.loads(final_messages[-2]["content"])
    assert submit_result["data"]["completed"] is True
    assert submit_result["data"]["metrics"]["complete_orders"] == 1


def test_warehouse_episode_does_not_fabricate_a_missing_tool_call():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _records(1, seed=23)[0]
    state = game.build_state(record)
    item = SimpleNamespace(
        prompt="pick an order",
        record=record,
        prompt_index=0,
        sample_index=0,
    )
    client = _fake_client([_response(content="I cannot act.")])

    turns = asyncio.run(run_agent._run_episode(item, state, client))

    assert len(turns) == 1
    assert len(client.completions.requests) == 1
    assert turns[0].response.choices[0].message.tool_calls == []
    assert state.cart == {}


def test_warehouse_concurrent_episodes_keep_state_isolated():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _records(1, seed=231)[0]
    first_state = game.build_state(record)
    second_state = game.build_state(record)
    sku, original_stock = next(iter(first_state.shelves["A1"].stock.items()))
    first_state.order = [{"sku": sku, "qty": 2}]
    second_state.order = [{"sku": sku, "qty": 2}]
    item = SimpleNamespace(
        prompt="pick an order",
        record=record,
        prompt_index=0,
        sample_index=0,
    )

    def client(call_id):
        return _fake_client(
            [
                _response(
                    {
                        "id": call_id,
                        "name": "pick_item",
                        "arguments": {"sku": sku, "qty": 1},
                    }
                ),
                _response(content="I cannot continue."),
            ]
        )

    async def run_both():
        return await asyncio.gather(
            run_agent._run_episode(item, first_state, client("first")),
            run_agent._run_episode(item, second_state, client("second")),
        )

    asyncio.run(run_both())

    assert first_state.cart == {sku: 1}
    assert second_state.cart == {sku: 1}
    assert first_state.shelves["A1"].stock.get(sku, 0) == original_stock - 1
    assert second_state.shelves["A1"].stock.get(sku, 0) == original_stock - 1


def test_warehouse_episode_rejects_multiple_calls_with_matched_results():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _records(1, seed=24)[0]
    state = game.build_state(record)
    item = SimpleNamespace(
        prompt="pick an order",
        record=record,
        prompt_index=0,
        sample_index=0,
    )
    client = _fake_client(
        [
            _response(
                {"id": "call-1", "name": "check_shelf"},
                {"id": "call-2", "name": "submit_order"},
            ),
            _response(content="The tool protocol was invalid."),
        ]
    )

    turns = asyncio.run(run_agent._run_episode(item, state, client))

    assert len(turns) == 2
    assert state.invalid_actions == 1
    final_messages = client.completions.requests[-1]["messages"]
    tool_messages = [message for message in final_messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == [
        "call-1",
        "call-2",
    ]
    assert all(json.loads(message["content"])["data"]["stage"] == "tool_protocol" for message in tool_messages)


def test_warehouse_reward_replays_optimal_partial_and_invalid_paths():
    game = _load_module("game")
    reward = _load_module("reward")
    record = _records(1, seed=25)[0]
    state = game.build_state(record)
    sku = next(iter(state.shelves["A1"].stock))
    record["order"] = [{"sku": sku, "qty": 1}]

    pick_call = _assistant_call(
        "pick",
        "pick_item",
        {"sku": sku, "qty": 1},
    )
    submit_call = _assistant_call("submit", "submit_order", {})
    optimal = SimpleNamespace(
        source_record=record,
        messages=[pick_call, submit_call],
        tool_calls=[],
    )
    partial = SimpleNamespace(
        source_record=record,
        messages=[pick_call],
        tool_calls=[],
    )
    repeated = SimpleNamespace(
        source_record=record,
        messages=[
            _assistant_call("check-1", "check_shelf", {}),
            _assistant_call("check-2", "check_shelf", {}),
        ],
        tool_calls=[],
    )
    multiple = SimpleNamespace(
        source_record=record,
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    pick_call["tool_calls"][0],
                    submit_call["tool_calls"][0],
                ],
            }
        ],
        tool_calls=[],
    )

    assert reward.reward_fn(optimal) == 1.0
    assert reward.reward_fn(partial) == 0.0
    assert reward.reward_fn(repeated) < -0.5
    assert reward.reward_fn(multiple) < -0.5
