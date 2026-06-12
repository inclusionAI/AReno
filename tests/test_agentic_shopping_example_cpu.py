from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "shopping"


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_shopping_{name}_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXAMPLE_DIR))
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


def _load_module_without_sys_path(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_shopping_{name}_without_path_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


def test_shopping_generator_produces_promptable_records():
    game = _load_module("game")
    generator = _load_module("dataset_generator")

    records = generator.generate_records(12, seed=7)

    assert len(records) == 12
    for record in records:
        assert set(record["categories"]).issubset({"jacket", "shoes", "bottle"})
        assert game.best_bundle(record)
        assert "submit the final item ids" in game.make_prompt(record)


def test_shopping_loader_and_reward_import_from_file_path_without_sys_path():
    generator = _load_module("dataset_generator")
    loader = _load_module_without_sys_path("dataset_loader")
    reward = _load_module_without_sys_path("reward")

    source = generator.generate_records(1, seed=9)[0]

    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])

    assert records[0]["prompt"].startswith("Build a ")
    assert reward.reward_fn(SimpleNamespace(source_record=source, tool_calls=[])) == -1.0


def test_shopping_tools_filter_inspect_and_check_catalog():
    game = _load_module("game")

    results = game.search_catalog("jacket", query="waterproof", max_price=100)

    assert results == [{"id": "packable-rain-shell", "name": "Packable Rain Shell", "price": 89, "rating": 4.6}]
    grouped = game.search_catalog_many(["jacket", "bottle"], max_price=140)
    assert {item["id"] for item in grouped["bottle"]} == {"insulated-bottle-750", "collapsible-bottle"}
    items = game.inspect_items(["packable-rain-shell", "insulated-bottle-750"])
    assert items[0]["features"] == ["waterproof", "packable", "breathable"]
    record = {
        "kit_name": "rain commute",
        "categories": ["jacket", "bottle"],
        "budget": 140,
        "required_features_by_category": {
            "jacket": ["waterproof", "packable"],
            "bottle": ["insulated", "leakproof"],
        },
    }
    assert game.check_kit(record, ["packable-rain-shell", "insulated-bottle-750"])["valid"] is True


def test_shopping_agent_tools_use_agent_item_record_shape():
    run_agent = _load_module_without_sys_path("run_agent")
    record = {
        "kit_name": "rain commute",
        "categories": ["jacket", "bottle"],
        "budget": 140,
        "required_features_by_category": {
            "jacket": ["waterproof", "packable"],
            "bottle": ["insulated", "leakproof"],
        },
    }
    assistant_message = {
        "tool_calls": [
            {
                "function": {
                    "name": "check_kit",
                    "arguments": json.dumps({"item_ids": ["packable-rain-shell", "insulated-bottle-750"]}),
                }
            }
        ]
    }

    result = run_agent._run_tool(assistant_message, record)

    assert result["kit"]["valid"] is True


def test_shopping_agent_missing_tool_call_returns_error_tool_result():
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {
        "role": "assistant",
        "content": "plain text",
        "tool_calls": [
            {
                "id": "missing_submit_bundle",
                "type": "function",
                "function": {"name": "submit_bundle", "arguments": "{}"},
            }
        ],
    }

    messages = run_agent._tool_messages(assistant_message, run_agent._run_tool(assistant_message, {}))

    assert messages[0]["tool_calls"][0]["function"]["name"] == "submit_bundle"
    assert messages[1]["role"] == "tool"
    assert json.loads(messages[1]["content"])["submitted"] == []


def test_shopping_reward_requires_final_submit_bundle():
    game = _load_module("game")
    reward = _load_module("reward")
    record = {
        "kit_name": "rain commute",
        "categories": ["jacket", "bottle"],
        "budget": 140,
        "required_features_by_category": {
            "jacket": ["waterproof", "packable"],
            "bottle": ["insulated", "leakproof"],
        },
    }
    best = game.best_bundle(record)
    reward_record = SimpleNamespace(
        source_record=record,
        tool_calls=[
            {"name": "search_catalog", "arguments": json.dumps({"categories": ["jacket", "bottle"], "max_price": 140})},
            {"name": "inspect_items", "arguments": json.dumps({"item_ids": best})},
            {"name": "check_kit", "arguments": json.dumps({"item_ids": best})},
        ],
    )

    assert reward.reward_fn(reward_record) == -1.0

    reward_record.tool_calls.append({"name": "submit_bundle", "arguments": json.dumps({"item_ids": best})})
    assert reward.reward_fn(reward_record) == 1.0
