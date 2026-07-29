from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "multitool"


def _areno_importable() -> bool:
    """Check whether the areno package can be imported (requires torch)."""

    try:
        import areno.api.agentic  # noqa: F401
    except Exception:
        return False
    return True


_HAS_ARENO = _areno_importable()
_requires_areno = pytest.mark.skipif(not _HAS_ARENO, reason="areno package not importable (torch missing)")


def _load_module(name: str):
    path = EXAMPLE_DIR / f"{name}.py"
    previous_game = sys.modules.pop("game", None)
    sys.path.insert(0, str(EXAMPLE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"agentic_multitool_{name}_for_tests", path)
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
        spec = importlib.util.spec_from_file_location(f"agentic_multitool_{name}_without_path_for_tests", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("game", None)
        if previous_game is not None:
            sys.modules["game"] = previous_game


# ---------------------------------------------------------------------------
# Tool logic tests
# ---------------------------------------------------------------------------


def test_lookup_contact_finds_alice():
    game = _load_module("game")
    result = game.lookup_contact("Alice")
    assert result is not None
    assert result["name"] == "Alice Chen"
    assert result["phone"] == "13800001111"


def test_lookup_contact_partial_match_case_insensitive():
    game = _load_module("game")
    result = game.lookup_contact("carol")
    assert result is not None
    assert result["name"] == "Carol Lee"


def test_lookup_contact_not_found_returns_none():
    game = _load_module("game")
    assert game.lookup_contact("Zzz") is None


def test_read_note_returns_content():
    game = _load_module("game")
    result = game.read_note("meeting")
    assert result is not None
    assert "Team sync" in result["content"]


def test_read_note_unknown_key_returns_none():
    game = _load_module("game")
    assert game.read_note("nonexistent") is None


def test_calculator_basic_arithmetic():
    game = _load_module("game")
    assert game.calculate("3 * 15")["result"] == 45.0
    assert game.calculate("10 + 5")["result"] == 15.0
    assert game.calculate("20 / 4")["result"] == 5.0
    assert game.calculate("(2 + 3) * 4")["result"] == 20.0


def test_calculator_division_by_zero_returns_error():
    game = _load_module("game")
    result = game.calculate("10 / 0")
    assert "error" in result
    assert "division by zero" in result["error"]


def test_calculator_empty_expression_returns_error():
    game = _load_module("game")
    result = game.calculate("")
    assert "error" in result


def test_calculator_invalid_characters_returns_error():
    game = _load_module("game")
    result = game.calculate("import os")
    assert "error" in result


def test_unit_convert_cm_to_m():
    game = _load_module("game")
    result = game.unit_convert(100, "cm", "m")
    assert result["result"] == 1.0


def test_unit_convert_kg_to_g():
    game = _load_module("game")
    result = game.unit_convert(2.5, "kg", "g")
    assert result["result"] == 2500.0


def test_unit_convert_unsupported_returns_error():
    game = _load_module("game")
    result = game.unit_convert(1, "cm", "kg")
    assert "error" in result


def test_lookup_parcel_existing():
    game = _load_module("game")
    result = game.lookup_parcel("P002")
    assert result is not None
    assert result["status"] == "in_transit"
    assert result["address"] == "Beijing"


def test_lookup_parcel_not_found():
    game = _load_module("game")
    assert game.lookup_parcel("P999") is None


# ---------------------------------------------------------------------------
# Dataset generator and loader tests
# ---------------------------------------------------------------------------


def test_generator_produces_deterministic_records():
    game = _load_module("game")
    records = game.generate_records(10, seed=42)
    assert len(records) == 10
    # Deterministic: same seed produces same output
    records2 = game.generate_records(10, seed=42)
    assert records == records2
    for record in records:
        assert "description" in record
        assert "required_tools" in record
        assert len(record["required_tools"]) >= 2
        assert "prompt" not in record  # prompt is added by loader


def test_loader_adds_prompt():
    loader = _load_module_without_sys_path("dataset_loader")
    source = {"id": "test-0", "description": "Do something", "required_tools": ["calculate", "read_note"]}
    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])
    assert records[0]["prompt"].startswith("Task: Do something")


# ---------------------------------------------------------------------------
# Scoring tests — success path
# ---------------------------------------------------------------------------


def test_score_contact_meeting_success():
    game = _load_module("game")
    record = {
        "id": "contact-meeting-0",
        "description": "Find Alice's phone number, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["tool_selection"] == 1.0
    assert score["arguments"] == 1.0
    assert score["order"] == 1.0
    assert score["final_answer"] == 1.0
    assert score["failures"] == []


def test_score_budget_shipping_success():
    game = _load_module("game")
    record = {
        "id": "budget-shipping-0",
        "description": "Read the budget note, then read the shipping note.",
        "required_tools": ["read_note", "read_note"],
        "expected_note_keys": ["budget", "shipping"],
    }
    tool_calls = [
        {"name": "read_note", "arguments": json.dumps({"note_key": "budget"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "shipping"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["failures"] == []


def test_score_parcel_city_success():
    game = _load_module("game")
    record = {
        "id": "parcel-city-0",
        "description": "Look up parcel P002, then find a contact in the same city.",
        "required_tools": ["lookup_parcel", "lookup_contact"],
        "expected_parcel": "P002",
        "expected_contact_city": "Beijing",
    }
    tool_calls = [
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P002"})},
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Bob"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["failures"] == []


def test_score_calc_shipping_success():
    game = _load_module("game")
    record = {
        "id": "calc-shipping-0",
        "description": "Calculate 3 * 15, then read the shipping note.",
        "required_tools": ["calculate", "read_note"],
        "expected_expression": "3 * 15",
        "expected_note_key": "shipping",
    }
    tool_calls = [
        {"name": "calculate", "arguments": json.dumps({"expression": "3 * 15"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "shipping"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["failures"] == []


def test_score_convert_parcel_success():
    game = _load_module("game")
    record = {
        "id": "convert-parcel-0",
        "description": "Convert 100 cm to m, then look up parcel P003.",
        "required_tools": ["unit_convert", "lookup_parcel"],
        "expected_value": 100,
        "expected_from_unit": "cm",
        "expected_to_unit": "m",
        "expected_parcel": "P003",
    }
    tool_calls = [
        {"name": "unit_convert", "arguments": json.dumps({"value": 100, "from_unit": "cm", "to_unit": "m"})},
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P003"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["failures"] == []


# ---------------------------------------------------------------------------
# Scoring tests — failure paths
# ---------------------------------------------------------------------------


def test_score_wrong_tool_order():
    game = _load_module("game")
    record = {
        "id": "contact-meeting-1",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["order"] < 1.0
    assert "order" in score["failures"]


def test_score_wrong_arguments():
    game = _load_module("game")
    record = {
        "id": "contact-meeting-2",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Bob"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "budget"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0
    assert "arguments" in score["failures"]


def test_score_missing_tool():
    game = _load_module("game")
    record = {
        "id": "contact-meeting-3",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["tool_selection"] < 1.0
    assert "tool_selection" in score["failures"]


def test_score_empty_trajectory():
    game = _load_module("game")
    record = {
        "id": "contact-meeting-4",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    score = game.score_task(record, [])
    assert score["overall"] == -1.0
    assert "tool_selection" in score["failures"]


def test_score_invalid_json_arguments():
    game = _load_module("game")
    record = {
        "id": "contact-meeting-5",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "lookup_contact", "arguments": "not-json"},
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0


# ---------------------------------------------------------------------------
# Reward function tests
# ---------------------------------------------------------------------------


def test_reward_fn_returns_positive_on_success():
    reward = _load_module("reward")
    record = {
        "id": "contact-meeting-0",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
    ]
    reward_record = SimpleNamespace(source_record=record, tool_calls=tool_calls)
    assert reward.reward_fn(reward_record) == 1.0


def test_reward_fn_returns_negative_on_empty_trajectory():
    reward = _load_module("reward")
    record = {
        "id": "contact-meeting-0",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    reward_record = SimpleNamespace(source_record=record, tool_calls=[])
    assert reward.reward_fn(reward_record) == -1.0


def test_reward_fn_partial_credit_on_wrong_order():
    reward = _load_module("reward")
    record = {
        "id": "contact-meeting-0",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
    ]
    reward_record = SimpleNamespace(source_record=record, tool_calls=tool_calls)
    result = reward.reward_fn(reward_record)
    assert -1.0 < result < 1.0


# ---------------------------------------------------------------------------
# run_agent tool dispatch tests
# ---------------------------------------------------------------------------


@_requires_areno
def test_run_agent_dispatches_lookup_contact():
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
            }
        ]
    }
    result = run_agent._run_tool(assistant_message)
    assert result["name"] == "Alice Chen"


@_requires_areno
def test_run_agent_dispatches_calculate():
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculate", "arguments": json.dumps({"expression": "2 + 3"})},
            }
        ]
    }
    result = run_agent._run_tool(assistant_message)
    assert result["result"] == 5.0


@_requires_areno
def test_run_agent_dispatches_unit_convert():
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "unit_convert",
                    "arguments": json.dumps({"value": 100, "from_unit": "cm", "to_unit": "m"}),
                },
            }
        ]
    }
    result = run_agent._run_tool(assistant_message)
    assert result["result"] == 1.0


@_requires_areno
def test_run_agent_missing_tool_call_returns_error():
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {"tool_calls": []}
    result = run_agent._run_tool(assistant_message)
    assert "error" in result


@_requires_areno
def test_run_agent_unknown_tool_returns_error():
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "nonexistent_tool", "arguments": "{}"},
            }
        ]
    }
    result = run_agent._run_tool(assistant_message)
    assert "error" in result


@_requires_areno
def test_run_agent_tool_messages_format():
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
            }
        ],
    }
    tool_result = run_agent._run_tool(assistant_message)
    messages = run_agent._tool_messages(assistant_message, tool_result)
    assert len(messages) == 2
    assert messages[0]["role"] == "assistant"
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == "call_1"
    assert messages[1]["name"] == "lookup_contact"


# ---------------------------------------------------------------------------
# Boundary tests
# ---------------------------------------------------------------------------


def test_score_task_with_no_required_tools():
    game = _load_module("game")
    record = {"id": "empty-0", "description": "No tools needed.", "required_tools": []}
    score = game.score_task(record, [])
    assert score["tool_selection"] == 1.0
    assert score["order"] == 1.0


def test_generator_seed_reproducibility():
    game = _load_module("game")
    r1 = game.generate_records(5, seed=100)
    r2 = game.generate_records(5, seed=100)
    assert r1 == r2
    r3 = game.generate_records(5, seed=999)
    assert r1 != r3