from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "warehouse"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    mod_name = f"agentic_warehouse_{name}_for_tests"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        sys.modules.pop(mod_name, None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


def _load_module_without_sys_path(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    mod_name = f"agentic_warehouse_{name}_without_path_for_tests"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("game", None)
        sys.modules.pop(mod_name, None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


def _make_record(**overrides):
    record = {
        "rows": 2,
        "cols": 2,
        "sku_pool": ["S1", "S2", "S3", "S4"],
        "max_stock": 5,
        "order_size": 2,
        "seed": 42,
        "start_shelf": "A1",
        "order": [{"sku": "S1", "qty": 1}, {"sku": "S2", "qty": 1}],
    }
    record.update(overrides)
    return record


# ── Data generation & loading ──


def test_warehouse_generator_produces_promptable_records():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(12, seed=7)

    assert len(records) == 12
    for record in records:
        assert record["difficulty"] in {"small", "medium", "hard"}
        assert "order" in record
        assert isinstance(record["order"], list)
        assert len(record["order"]) == record["order_size"]
        assert game.make_prompt(record)


def test_warehouse_generator_is_deterministic():
    _load_module("game")
    gen1 = _load_module("dataset_generator")
    gen2 = _load_module("dataset_generator")

    r1 = gen1.generate_records(8, seed=2026)
    r2 = gen2.generate_records(8, seed=2026)

    assert r1 == r2


def test_warehouse_loader_imports_from_file_path_without_sys_path():
    generator = _load_module("dataset_generator")
    loader = _load_module_without_sys_path("dataset_loader")
    reward = _load_module_without_sys_path("reward")

    source = generator.generate_records(1, seed=9)[0]

    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])

    assert records[0]["prompt"].startswith("You are in a warehouse")
    assert reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=[], messages=[])) == -0.7


def test_warehouse_loader_fills_missing_fields_from_difficulty():
    loader = _load_module_without_sys_path("dataset_loader")
    source = {
        "id": 1,
        "difficulty": "small",
        "seed": 42,
        "order": [{"sku": "S1", "qty": 1}],
    }

    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])

    assert records[0]["rows"] == 2
    assert records[0]["cols"] == 2
    assert records[0]["sku_pool"] == ["S1", "S2", "S3", "S4"]
    assert records[0]["start_shelf"] == "A1"
    assert "prompt" in records[0]


# ── Environment core logic ──


def test_warehouse_game_build_state_generates_valid_layout():
    game = _load_module("game")
    record = _make_record()
    state = game.build_state(record)

    assert len(state.shelves) == 4
    assert state.agent_pos == "A1"
    assert state.order == record["order"]
    assert state.cart == {}
    assert state.completed is False


def test_warehouse_game_adjacency_is_symmetric():
    game = _load_module("game")
    record = _make_record()
    state = game.build_state(record)

    for shelf_id, neighbors in state.adjacency.items():
        for neighbor in neighbors:
            assert shelf_id in state.adjacency[neighbor], f"{shelf_id} -> {neighbor} not symmetric"


def test_warehouse_game_query_inventory_valid():
    game = _load_module("game")
    state = game.build_state(_make_record())

    result = game.query_inventory(state, "A1")

    assert result.success
    assert result.data["shelf_id"] == "A1"
    assert isinstance(result.data["stock"], dict)


def test_warehouse_game_query_inventory_unknown_shelf():
    game = _load_module("game")
    state = game.build_state(_make_record())

    result = game.query_inventory(state, "Z9")

    assert not result.success
    assert "unknown shelf" in result.message


def test_warehouse_game_move_to_adjacent_shelf():
    game = _load_module("game")
    state = game.build_state(_make_record())

    neighbors = state.adjacency["A1"]
    target = neighbors[0]
    result = game.move(state, target)

    assert result.success
    assert state.agent_pos == target
    assert state.total_distance == 1


def test_warehouse_game_move_to_non_adjacent_shelf():
    game = _load_module("game")
    state = game.build_state(_make_record())

    # A1 -> A2 is adjacent in 2x2, but A1 -> B2 is diagonal (not adjacent)
    result = game.move(state, "B2")

    assert not result.success
    assert "unreachable" in result.message
    assert state.invalid_actions == 1


def test_warehouse_game_move_to_unknown_shelf():
    game = _load_module("game")
    state = game.build_state(_make_record())

    result = game.move(state, "Z9")

    assert not result.success
    assert "unknown shelf" in result.message
    assert state.invalid_actions == 1


def test_warehouse_game_pick_valid():
    game = _load_module("game")
    state = game.build_state(_make_record())

    # Find a SKU on the starting shelf and pick it
    shelf = state.shelves["A1"]
    sku = next(iter(shelf.stock))
    qty = min(shelf.stock[sku], 2)

    result = game.pick(state, sku, qty)

    assert result.success
    assert state.cart[sku] == qty


def test_warehouse_game_pick_wrong_shelf():
    game = _load_module("game")
    state = game.build_state(_make_record())

    # Find a SKU that is NOT on A1
    shelf_a1 = state.shelves["A1"]
    all_skus = set()
    for s in state.shelves.values():
        all_skus.update(s.stock.keys())
    missing = all_skus - set(shelf_a1.stock.keys())
    if missing:
        sku = next(iter(missing))
        result = game.pick(state, sku, 1)

        assert not result.success
        assert "not on shelf" in result.message
        assert state.picking_errors == 1


def test_warehouse_game_pick_insufficient_stock():
    game = _load_module("game")
    state = game.build_state(_make_record())

    shelf = state.shelves["A1"]
    sku = next(iter(shelf.stock))
    excess_qty = shelf.stock[sku] + 1

    result = game.pick(state, sku, excess_qty)

    assert not result.success
    assert "insufficient stock" in result.message
    assert state.picking_errors == 1


def test_warehouse_game_pick_zero_qty():
    game = _load_module("game")
    state = game.build_state(_make_record())

    result = game.pick(state, "S1", 0)

    assert not result.success
    assert "invalid qty" in result.message
    assert state.picking_errors == 1


def test_warehouse_game_submit_complete_order():
    game = _load_module("game")
    record = _make_record()
    state = game.build_state(record)

    # Pick all order items from wherever they are
    for item in state.order:
        sku, qty = item["sku"], item["qty"]
        # Find which shelf has this SKU
        for shelf_id, shelf in state.shelves.items():
            if shelf.stock.get(sku, 0) >= qty:
                if state.agent_pos != shelf_id:
                    # Move step by step to the shelf (brute force for test)
                    _force_move(game, state, shelf_id)
                game.pick(state, sku, qty)
                break

    result = game.submit_order(state)

    assert result.success
    assert state.completed is True


def test_warehouse_game_submit_incomplete_order():
    game = _load_module("game")
    state = game.build_state(_make_record())

    # Don't pick anything, submit directly
    result = game.submit_order(state)

    assert not result.success
    assert "order incomplete" in result.message
    assert state.completed is False


def test_warehouse_game_submit_extra_items():
    game = _load_module("game")
    record = _make_record(order=[{"sku": "S1", "qty": 1}])
    state = game.build_state(record)

    # Pick more than needed
    shelf = state.shelves["A1"]
    for sku, available in shelf.stock.items():
        game.pick(state, sku, 1)
        break

    # Pick an extra item if available
    for shelf_id, shelf in state.shelves.items():
        if shelf_id == state.agent_pos:
            continue
        for sku in shelf.stock:
            _force_move(game, state, shelf_id)
            game.pick(state, sku, 1)
            break
        break

    result = game.submit_order(state)

    assert not result.success
    assert state.completed is False


def test_warehouse_game_submit_empty_cart():
    game = _load_module("game")
    state = game.build_state(_make_record())

    result = game.submit_order(state)

    assert not result.success
    assert "order incomplete" in result.message


# ── Scoring ──


def test_warehouse_baseline_distance_returns_positive():
    game = _load_module("game")
    state = game.build_state(_make_record())

    baseline = game.baseline_distance(state)

    assert isinstance(baseline, int)
    assert baseline >= 0


def test_warehouse_score_task_completed_correct_sequence():
    game = _load_module("game")
    record = _make_record()
    state = game.build_state(record)
    baseline = game.baseline_distance(state)

    trajectory_data = {
        "completed": True,
        "distance": baseline,
        "picking_errors": 0,
        "invalid_actions": 0,
        "baseline_distance": baseline,
        "tool_names": ["pick_from_shelf", "submit_order"],
    }

    score = game.score_task(record, trajectory_data)

    assert score > 1.0  # 1.0 base + 0.2 bonus


def test_warehouse_score_task_not_completed():
    game = _load_module("game")
    record = _make_record()

    # No tools called, no cart, no distance → base -0.5 - 0.2 (no pick) = -0.7
    score = game.score_task(record, {"completed": False, "tool_names": []})

    assert score == -0.7


def test_warehouse_score_task_with_errors():
    game = _load_module("game")
    record = _make_record()
    state = game.build_state(record)
    baseline = game.baseline_distance(state)

    trajectory_data = {
        "completed": True,
        "distance": baseline,
        "picking_errors": 2,
        "invalid_actions": 1,
        "baseline_distance": baseline,
        "tool_names": ["query_inventory", "move", "pick", "submit_order"],
    }

    score = game.score_task(record, trajectory_data)

    # 1.0 - 0.2 - 0.05 = 0.75, * efficiency, + 0.2
    assert score < 1.2
    assert score > 0.5


# ── Agent entrypoint ──


def test_warehouse_agent_tools_use_record_shape():
    run_agent = _load_module_without_sys_path("run_agent")
    run_agent.reset_state_cache()
    record = _make_record()

    # Build state to find a valid SKU on A1
    game = _load_module("game")
    state = game.build_state(record)
    shelf = state.shelves["A1"]
    sku = next(iter(shelf.stock))
    qty = min(shelf.stock[sku], 1)

    assistant_message = {
        "tool_calls": [
            {
                "function": {
                    "name": "pick_from_shelf",
                    "arguments": json.dumps({"shelf_id": "A1", "sku": sku, "qty": qty}),
                }
            }
        ]
    }

    result = run_agent._run_tool(assistant_message, record)

    assert result["success"] is True
    assert result["data"]["cart"][sku] == qty


def test_warehouse_agent_missing_tool_call_returns_error():
    run_agent = _load_module_without_sys_path("run_agent")
    run_agent.reset_state_cache()

    assistant_message = {"role": "assistant", "content": "plain text", "tool_calls": []}

    result = run_agent._run_tool(assistant_message, _make_record())

    assert result == {"error": "missing tool call"}


def test_warehouse_agent_invalid_json_arguments():
    run_agent = _load_module_without_sys_path("run_agent")
    run_agent.reset_state_cache()

    assistant_message = {
        "tool_calls": [
            {
                "function": {
                    "name": "query_inventory",
                    "arguments": "{invalid json}",
                }
            }
        ]
    }

    result = run_agent._run_tool(assistant_message, _make_record())

    assert result == {"error": "invalid JSON arguments"}


def test_warehouse_agent_unknown_tool_returns_error():
    run_agent = _load_module_without_sys_path("run_agent")
    run_agent.reset_state_cache()

    assistant_message = {
        "tool_calls": [
            {
                "function": {
                    "name": "fly",
                    "arguments": "{}",
                }
            }
        ]
    }

    result = run_agent._run_tool(assistant_message, _make_record())

    assert result == {"error": "unknown tool: fly"}


# ── Reward function ──


def test_warehouse_reward_no_submit_returns_negative():
    reward = _load_module_without_sys_path("reward")
    record = _make_record()
    reward_record = SimpleNamespace(
        source_record=record,
        tool_calls=[
            {"name": "query_inventory", "arguments": json.dumps({"shelf_id": "A1"})},
            {"name": "move", "arguments": json.dumps({"target_shelf_id": "A2"})},
            {"name": "pick", "arguments": json.dumps({"sku": "S1", "qty": 1})},
        ],
        messages=[],
    )

    # No pick_from_shelf or submit_order in tool_calls → -0.5 - 0.2 (no pick) = -0.7
    assert reward.reward_fn(reward_record) == -0.7


def test_warehouse_reward_full_correct_sequence():
    game = _load_module("game")
    reward = _load_module_without_sys_path("reward")
    record = _make_record()
    state = game.build_state(record)
    baseline = game.baseline_distance(state)

    # Simulate a successful complete trajectory (2-turn: pick_from_shelf + submit_order)
    messages = [
        {"role": "tool", "content": json.dumps({"success": True, "message": "picked", "data": {"cart": {}, "distance": baseline}})},
        {"role": "tool", "content": json.dumps({"success": True, "message": "order completed", "data": {"completed": True, "distance": baseline}})},
    ]
    reward_record = SimpleNamespace(
        source_record=record,
        tool_calls=[
            {"name": "pick_from_shelf", "arguments": "{}"},
            {"name": "submit_order", "arguments": "{}"},
        ],
        messages=messages,
    )

    score = reward.reward_fn(reward_record)

    assert score > 1.0  # completed + correct sequence + optimal distance


def test_warehouse_reward_wrong_sequence():
    game = _load_module("game")
    reward = _load_module_without_sys_path("reward")
    record = _make_record()
    state = game.build_state(record)
    baseline = game.baseline_distance(state)

    messages = [
        {"role": "tool", "content": json.dumps({"success": True, "message": "picked", "data": {"distance": baseline}})},
        {"role": "tool", "content": json.dumps({"success": True, "message": "order completed", "data": {"completed": True, "distance": baseline}})},
    ]
    reward_record = SimpleNamespace(
        source_record=record,
        tool_calls=[
            {"name": "submit_order", "arguments": "{}"},
            {"name": "pick_from_shelf", "arguments": "{}"},
        ],
        messages=messages,
    )

    score = reward.reward_fn(reward_record)

    # No sequence bonus (0.2 less than correct sequence)
    correct_names = ["pick_from_shelf", "submit_order"]
    correct_reward_record = SimpleNamespace(
        source_record=record,
        tool_calls=[{"name": n, "arguments": "{}"} for n in correct_names],
        messages=messages,
    )
    correct_score = reward.reward_fn(correct_reward_record)

    assert score < correct_score


# ── Helpers ──


def _force_move(game, state, target_shelf_id):
    """Move agent to target shelf by any path, picking adjacent moves."""

    import collections

    if state.agent_pos == target_shelf_id:
        return
    # BFS path
    visited = {state.agent_pos}
    queue = collections.deque([(state.agent_pos, [])])
    while queue:
        node, path = queue.popleft()
        for neighbor in state.adjacency.get(node, []):
            if neighbor == target_shelf_id:
                for step in path + [neighbor]:
                    game.move(state, step)
                return
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))