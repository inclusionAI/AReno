from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Path to the multitool example directory, used to load game.py, reward.py, etc.
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "multitool"


def _areno_importable() -> bool:
    """Check whether the areno package can be imported (requires torch).

    Some tests (e.g. run_agent dispatch) depend on the areno package being
    installed. This function probes importability so those tests can be
    conditionally skipped when torch is not available (e.g. in CI without GPU).

    Returns:
        True if `areno.api.agentic` can be imported, False otherwise.
    """

    try:
        import areno.api.agentic  # noqa: F401
    except Exception:
        return False
    return True


# Cached flag: whether the areno package is importable in this environment.
_HAS_ARENO = _areno_importable()
# Pytest marker: skip a test if areno (and thus torch) is not installed.
_requires_areno = pytest.mark.skipif(not _HAS_ARENO, reason="areno package not importable (torch missing)")


def _load_module(name: str):
    """Dynamically load a Python module from the multitool example directory.

    Temporarily adds the example directory to sys.path so that the module
    can import its sibling modules (e.g. game.py imports from game). The
    path and any 'game' module entry are cleaned up in the finally block
    to avoid polluting other tests.

    Args:
        name: The module file name without .py extension (e.g. "game", "reward").

    Returns:
        The loaded module object.
    """

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
    """Load a module from the example directory WITHOUT adding it to sys.path.

    Used for modules (e.g. dataset_loader.py, run_agent.py) that import
    'game' via a sys.path manipulation at their own file level, so we
    must NOT pre-add the directory — otherwise duplicate path entries
    or import conflicts may occur.

    Args:
        name: The module file name without .py extension.

    Returns:
        The loaded module object.
    """

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
# Tool logic tests — verify each fake tool returns correct results
# ---------------------------------------------------------------------------


def test_lookup_contact_finds_alice():
    """Verify lookup_contact finds Alice by partial name match."""

    game = _load_module("game")
    result = game.lookup_contact("Alice")
    assert result is not None
    assert result["name"] == "Alice Chen"
    assert result["phone"] == "13800001111"


def test_lookup_contact_partial_match_case_insensitive():
    """Verify lookup_contact matches case-insensitively with a partial name."""

    game = _load_module("game")
    result = game.lookup_contact("carol")
    assert result is not None
    assert result["name"] == "Carol Lee"


def test_lookup_contact_not_found_returns_none():
    """Verify lookup_contact returns None for a name that doesn't exist."""

    game = _load_module("game")
    assert game.lookup_contact("Zzz") is None


def test_read_note_returns_content():
    """Verify read_note returns the note content for a valid key."""

    game = _load_module("game")
    result = game.read_note("meeting")
    assert result is not None
    assert "Team sync" in result["content"]


def test_read_note_unknown_key_returns_none():
    """Verify read_note returns None for a non-existent note key."""

    game = _load_module("game")
    assert game.read_note("nonexistent") is None


def test_calculator_basic_arithmetic():
    """Verify calculate handles +, -, *, / and parentheses correctly."""

    game = _load_module("game")
    assert game.calculate("3 * 15")["result"] == 45.0
    assert game.calculate("10 + 5")["result"] == 15.0
    assert game.calculate("20 / 4")["result"] == 5.0
    assert game.calculate("(2 + 3) * 4")["result"] == 20.0


def test_calculator_division_by_zero_returns_error():
    """Verify calculate returns an error dict (not an exception) for division by zero."""

    game = _load_module("game")
    result = game.calculate("10 / 0")
    assert "error" in result
    assert "division by zero" in result["error"]


def test_calculator_empty_expression_returns_error():
    """Verify calculate returns an error for an empty string input."""

    game = _load_module("game")
    result = game.calculate("")
    assert "error" in result


def test_calculator_invalid_characters_returns_error():
    """Verify calculate rejects non-arithmetic input (e.g. code injection attempts)."""

    game = _load_module("game")
    result = game.calculate("import os")
    assert "error" in result


def test_unit_convert_cm_to_m():
    """Verify unit_convert correctly converts 100 cm to 1 m."""

    game = _load_module("game")
    result = game.unit_convert(100, "cm", "m")
    assert result["result"] == 1.0


def test_unit_convert_kg_to_g():
    """Verify unit_convert correctly converts 2.5 kg to 2500 g."""

    game = _load_module("game")
    result = game.unit_convert(2.5, "kg", "g")
    assert result["result"] == 2500.0


def test_unit_convert_unsupported_returns_error():
    """Verify unit_convert returns an error when converting across categories (length to weight)."""

    game = _load_module("game")
    result = game.unit_convert(1, "cm", "kg")
    assert "error" in result


def test_lookup_parcel_existing():
    """Verify lookup_parcel returns tracking info for a valid tracking id."""

    game = _load_module("game")
    result = game.lookup_parcel("P002")
    assert result is not None
    assert result["status"] == "in_transit"
    assert result["address"] == "Beijing"


def test_lookup_parcel_not_found():
    """Verify lookup_parcel returns None for a non-existent tracking id."""

    game = _load_module("game")
    assert game.lookup_parcel("P999") is None


def test_search_notes_finds_matching_keyword():
    """Verify search_notes returns matching notes for a keyword in content."""

    game = _load_module("game")
    results = game.search_notes("Team")
    assert len(results) > 0
    assert results[0]["key"] == "meeting"


def test_search_notes_case_insensitive():
    """Verify search_notes matches case-insensitively."""

    game = _load_module("game")
    results = game.search_notes("BUDGET")
    assert len(results) > 0
    assert results[0]["key"] == "budget"


def test_search_notes_no_match_returns_empty():
    """Verify search_notes returns empty list for a non-matching keyword."""

    game = _load_module("game")
    assert game.search_notes("nonexistent") == []


def test_search_notes_empty_keyword_returns_empty():
    """Verify search_notes returns empty list for an empty keyword."""

    game = _load_module("game")
    assert game.search_notes("") == []


def test_list_contacts_by_city_returns_shanghai():
    """Verify list_contacts_by_city returns contacts in Shanghai."""

    game = _load_module("game")
    results = game.list_contacts_by_city("Shanghai")
    assert len(results) >= 2
    for contact in results:
        assert contact["city"] == "Shanghai"


def test_list_contacts_by_city_case_insensitive():
    """Verify list_contacts_by_city matches case-insensitively."""

    game = _load_module("game")
    results = game.list_contacts_by_city("beijing")
    assert len(results) == 1
    assert results[0]["name"] == "Bob Smith"


def test_list_contacts_by_city_no_match_returns_empty():
    """Verify list_contacts_by_city returns empty list for a city with no contacts."""

    game = _load_module("game")
    assert game.list_contacts_by_city("Tokyo") == []


# ---------------------------------------------------------------------------
# Dataset generator and loader tests — verify data pipeline correctness
# ---------------------------------------------------------------------------


def test_generator_produces_deterministic_records():
    """Verify generate_records produces the correct count and is deterministic by seed.

    Checks: record count matches request, same seed yields identical output,
    each record has the required fields, and no pre-built prompt leaks in.
    """

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
    """Verify the dataset loader injects a 'prompt' field into each record.

    The loader calls make_prompt() on the record's description and stores
    the result as record["prompt"], which is what the agent sees during rollout.
    """

    loader = _load_module_without_sys_path("dataset_loader")
    source = {"id": "test-0", "description": "Do something", "required_tools": ["calculate", "read_note"]}
    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])
    assert records[0]["prompt"].startswith("Task: Do something")


def test_loader_validates_known_tools():
    """Verify the loader rejects records with unknown tool names before model init."""

    loader = _load_module_without_sys_path("dataset_loader")
    source = {"id": "test-0", "description": "Bad tools", "required_tools": ["nonexistent_tool", "read_note"]}
    with pytest.raises(ValueError, match="unknown tool"):
        loader.load_training_dataset("unused", default_loader=lambda _: [source])


def test_loader_validates_required_fields_present():
    """Verify the loader rejects records missing 'id' or 'description'."""

    loader = _load_module_without_sys_path("dataset_loader")
    # Missing id
    with pytest.raises(ValueError, match="missing required field 'id'"):
        loader.load_training_dataset("unused", default_loader=lambda _: [{"description": "x", "required_tools": ["read_note", "read_note"]}])
    # Missing description
    with pytest.raises(ValueError, match="missing 'description'"):
        loader.load_training_dataset("unused", default_loader=lambda _: [{"id": "test-0", "required_tools": ["read_note", "read_note"]}])


def test_loader_validates_min_two_required_tools():
    """Verify the loader rejects records with fewer than 2 required tools."""

    loader = _load_module_without_sys_path("dataset_loader")
    source = {"id": "test-0", "description": "One tool", "required_tools": ["read_note"]}
    with pytest.raises(ValueError, match="at least 2"):
        loader.load_training_dataset("unused", default_loader=lambda _: [source])


def test_loader_validates_expected_fields_per_task_type():
    """Verify the loader rejects records missing expected_* fields for their task type."""

    loader = _load_module_without_sys_path("dataset_loader")
    # contact task missing expected_contact
    source = {"id": "contact-meeting-0", "description": "Find Alice", "required_tools": ["lookup_contact", "read_note"]}
    with pytest.raises(ValueError, match="missing expected fields"):
        loader.load_training_dataset("unused", default_loader=lambda _: [source])
    # calc task missing expected_expression
    source2 = {"id": "calc-shipping-0", "description": "Calculate", "required_tools": ["calculate", "read_note"],
               "expected_note_key": "shipping"}
    with pytest.raises(ValueError, match="missing expected fields"):
        loader.load_training_dataset("unused", default_loader=lambda _: [source2])


def test_loader_passes_valid_contact_task():
    """Verify the loader accepts a well-formed contact task with all expected fields."""

    loader = _load_module_without_sys_path("dataset_loader")
    source = {"id": "contact-meeting-0", "description": "Find Alice's phone, then check meeting note.",
              "required_tools": ["lookup_contact", "read_note"],
              "expected_contact": "Alice Chen", "expected_note_key": "meeting"}
    records = loader.load_training_dataset("unused", default_loader=lambda _: [source])
    assert len(records) == 1
    assert "prompt" in records[0]


# ---------------------------------------------------------------------------
# Scoring tests — success path: verify perfect trajectories score 1.0
# ---------------------------------------------------------------------------


def test_score_contact_meeting_success():
    """Verify a correct contact-meeting trajectory (lookup_contact then read_note) gets a perfect score."""

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
    """Verify a correct budget-shipping trajectory (two read_note calls) gets a perfect score."""

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
    """Verify a correct parcel-city trajectory (lookup_parcel then lookup_contact by city) gets a perfect score."""

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
    """Verify a correct calc-shipping trajectory (calculate then read_note) gets a perfect score."""

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
    """Verify a correct convert-parcel trajectory (unit_convert then lookup_parcel) gets a perfect score."""

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


def test_score_search_meeting_contact_success():
    """Verify a correct 3-step search-meeting-contact trajectory gets a perfect score."""

    game = _load_module("game")
    record = {
        "id": "search-meeting-contact-0",
        "description": "Search notes for 'Team', read the meeting note, then list contacts in Shanghai.",
        "required_tools": ["search_notes", "read_note", "list_contacts_by_city"],
        "expected_search_keyword": "Team",
        "expected_note_key": "meeting",
        "expected_city": "Shanghai",
    }
    tool_calls = [
        {"name": "search_notes", "arguments": json.dumps({"keyword": "Team"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
        {"name": "list_contacts_by_city", "arguments": json.dumps({"city": "Shanghai"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["failures"] == []


def test_score_parcel_calc_note_success():
    """Verify a correct 3-step parcel-calc-note trajectory gets a perfect score."""

    game = _load_module("game")
    record = {
        "id": "parcel-calc-note-0",
        "description": "Look up parcel P002, calculate 7 - 6, then read the shipping note.",
        "required_tools": ["lookup_parcel", "calculate", "read_note"],
        "expected_parcel": "P002",
        "expected_expression": "7 - 6",
        "expected_note_key": "shipping",
    }
    tool_calls = [
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P002"})},
        {"name": "calculate", "arguments": json.dumps({"expression": "7 - 6"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "shipping"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["failures"] == []


def test_score_convert_search_contact_parcel_success():
    """Verify a correct 4-step convert-search-contact-parcel trajectory gets a perfect score."""

    game = _load_module("game")
    record = {
        "id": "convert-search-contact-parcel-0",
        "description": "Convert 1000 mm to m, search notes for 'shipping', list contacts in Shanghai, then look up parcel P001.",
        "required_tools": ["unit_convert", "search_notes", "list_contacts_by_city", "lookup_parcel"],
        "expected_value": 1000,
        "expected_from_unit": "mm",
        "expected_to_unit": "m",
        "expected_search_keyword": "shipping",
        "expected_city": "Shanghai",
        "expected_parcel": "P001",
    }
    tool_calls = [
        {"name": "unit_convert", "arguments": json.dumps({"value": 1000, "from_unit": "mm", "to_unit": "m"})},
        {"name": "search_notes", "arguments": json.dumps({"keyword": "shipping"})},
        {"name": "list_contacts_by_city", "arguments": json.dumps({"city": "Shanghai"})},
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P001"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["overall"] == 1.0
    assert score["failures"] == []


# ---------------------------------------------------------------------------
# Scoring tests — failure paths: verify incorrect trajectories are penalized
# ---------------------------------------------------------------------------


def test_score_wrong_tool_order():
    """Verify that calling tools in the wrong order lowers the 'order' score."""

    game = _load_module("game")
    record = {
        "id": "contact-meeting-1",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    # Intentionally reversed order: read_note before lookup_contact
    tool_calls = [
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["order"] < 1.0
    assert "order" in score["failures"]


def test_score_wrong_arguments():
    """Verify that passing incorrect tool arguments lowers the 'arguments' score."""

    game = _load_module("game")
    record = {
        "id": "contact-meeting-2",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    # Wrong: looking up Bob instead of Alice, reading budget instead of meeting
    tool_calls = [
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Bob"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "budget"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0
    assert "arguments" in score["failures"]


def test_score_missing_tool():
    """Verify that omitting a required tool lowers the 'tool_selection' score."""

    game = _load_module("game")
    record = {
        "id": "contact-meeting-3",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    # Only one of two required tools is called
    tool_calls = [
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["tool_selection"] < 1.0
    assert "tool_selection" in score["failures"]


def test_score_empty_trajectory():
    """Verify that an empty tool-call trajectory gets the worst overall score (-1.0)."""

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
    """Verify that malformed JSON arguments are treated as a failed argument check."""

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


def test_score_search_meeting_contact_wrong_order():
    """Verify that wrong tool order in a 3-step search-meeting-contact task lowers the 'order' score."""

    game = _load_module("game")
    record = {
        "id": "search-meeting-contact-1",
        "description": "Search notes for 'Team', read the meeting note, then list contacts in Shanghai.",
        "required_tools": ["search_notes", "read_note", "list_contacts_by_city"],
        "expected_search_keyword": "Team",
        "expected_note_key": "meeting",
        "expected_city": "Shanghai",
    }
    # Wrong order: read_note before search_notes
    tool_calls = [
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
        {"name": "search_notes", "arguments": json.dumps({"keyword": "Team"})},
        {"name": "list_contacts_by_city", "arguments": json.dumps({"city": "Shanghai"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["order"] < 1.0
    assert "order" in score["failures"]


def test_score_parcel_calc_note_wrong_arguments():
    """Verify that wrong arguments in a 3-step parcel-calc-note task lowers the 'arguments' score."""

    game = _load_module("game")
    record = {
        "id": "parcel-calc-note-1",
        "description": "Look up parcel P002, calculate 7 - 6, then read the shipping note.",
        "required_tools": ["lookup_parcel", "calculate", "read_note"],
        "expected_parcel": "P002",
        "expected_expression": "7 - 6",
        "expected_note_key": "shipping",
    }
    # Wrong: looking up P001 instead of P002, wrong expression, wrong note key
    tool_calls = [
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P001"})},
        {"name": "calculate", "arguments": json.dumps({"expression": "3 + 3"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "budget"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0
    assert "arguments" in score["failures"]


def test_score_convert_search_contact_parcel_missing_tool():
    """Verify that omitting a tool in a 4-step task lowers the 'tool_selection' score."""

    game = _load_module("game")
    record = {
        "id": "convert-search-contact-parcel-1",
        "description": "Convert 1000 mm to m, search notes for 'shipping', list contacts in Shanghai, then look up parcel P001.",
        "required_tools": ["unit_convert", "search_notes", "list_contacts_by_city", "lookup_parcel"],
        "expected_value": 1000,
        "expected_from_unit": "mm",
        "expected_to_unit": "m",
        "expected_search_keyword": "shipping",
        "expected_city": "Shanghai",
        "expected_parcel": "P001",
    }
    # Missing list_contacts_by_city
    tool_calls = [
        {"name": "unit_convert", "arguments": json.dumps({"value": 1000, "from_unit": "mm", "to_unit": "m"})},
        {"name": "search_notes", "arguments": json.dumps({"keyword": "shipping"})},
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P001"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["tool_selection"] < 1.0
    assert "tool_selection" in score["failures"]


def test_reward_fn_returns_positive_on_success():
    """Verify reward_fn returns 1.0 when the agent's tool calls are fully correct."""

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
    """Verify reward_fn returns -1.0 when the agent made no tool calls at all."""

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
    """Verify reward_fn returns a partial score (between -1 and 1) for wrong tool order."""

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
# run_agent tool dispatch tests — verify _run_tool routes to the correct game tool
# ---------------------------------------------------------------------------


@_requires_areno
def test_run_agent_dispatches_lookup_contact():
    """Verify _run_tool dispatches a lookup_contact call and returns the contact dict."""
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
    """Verify _run_tool dispatches a calculate call and returns the numeric result."""
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
    """Verify _run_tool dispatches a unit_convert call and returns the converted value."""
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
    """Verify _run_tool returns an error dict when the assistant message has no tool calls."""
    run_agent = _load_module_without_sys_path("run_agent")
    assistant_message = {"tool_calls": []}
    result = run_agent._run_tool(assistant_message)
    assert "error" in result


@_requires_areno
def test_run_agent_unknown_tool_returns_error():
    """Verify _run_tool returns an error dict when an unrecognized tool name is called."""
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
    """Verify _tool_messages produces the correct OpenAI-style message pair (assistant + tool result)."""
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
# Boundary tests — edge cases and reproducibility checks
# ---------------------------------------------------------------------------


def test_score_task_with_no_required_tools():
    """Verify score_task gives perfect tool_selection and order when no tools are required."""

    game = _load_module("game")
    record = {"id": "empty-0", "description": "No tools needed.", "required_tools": []}
    score = game.score_task(record, [])
    assert score["tool_selection"] == 1.0
    assert score["order"] == 1.0


def test_generator_seed_reproducibility():
    """Verify that the same seed always produces the same records, and different seeds differ."""

    game = _load_module("game")
    r1 = game.generate_records(5, seed=100)
    r2 = game.generate_records(5, seed=100)
    assert r1 == r2
    r3 = game.generate_records(5, seed=999)
    assert r1 != r3


# ---------------------------------------------------------------------------
# Robustness tests — verify score_task does not crash on malformed agent output
# ---------------------------------------------------------------------------


def test_score_unit_convert_string_value_does_not_crash():
    """Verify _score_arguments does not crash when value is a string expression (e.g. '100*10')."""

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
        {"name": "unit_convert", "arguments": json.dumps({"value": "100*10", "from_unit": "cm", "to_unit": "m"})},
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P003"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0


def test_score_unit_convert_none_value_does_not_crash():
    """Verify _score_arguments does not crash when value is None."""

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
        {"name": "unit_convert", "arguments": json.dumps({"value": None, "from_unit": "cm", "to_unit": "m"})},
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P003"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0


def test_score_unit_convert_text_value_does_not_crash():
    """Verify _score_arguments does not crash when value is a non-numeric string (e.g. 'abc')."""

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
        {"name": "unit_convert", "arguments": json.dumps({"value": "abc", "from_unit": "cm", "to_unit": "m"})},
        {"name": "lookup_parcel", "arguments": json.dumps({"tracking_id": "P003"})},
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0


def test_score_read_note_none_expected_keys_does_not_crash():
    """Verify _score_arguments does not crash when expected_note_keys is None and read_note is called."""

    game = _load_module("game")
    record = {
        "id": "contact-meeting-0",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_keys": None,
    }
    tool_calls = [
        {"name": "lookup_contact", "arguments": json.dumps({"name": "Alice"})},
        {"name": "read_note", "arguments": json.dumps({"note_key": "meeting"})},
    ]
    score = game.score_task(record, tool_calls)
    # Should not crash; arguments may be < 1.0 because expected_note_key is missing
    assert "arguments" in score


def test_score_missing_arguments_key_does_not_crash():
    """Verify _score_arguments does not crash when a tool call has no 'arguments' key at all."""

    game = _load_module("game")
    record = {
        "id": "contact-meeting-0",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "lookup_contact"},  # no "arguments" key
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0


def test_score_non_dict_arguments_does_not_crash():
    """Verify _score_arguments does not crash when arguments is a list instead of dict/JSON."""

    game = _load_module("game")
    record = {
        "id": "contact-meeting-0",
        "description": "Find Alice's phone, then check the meeting note.",
        "required_tools": ["lookup_contact", "read_note"],
        "expected_contact": "Alice Chen",
        "expected_note_key": "meeting",
    }
    tool_calls = [
        {"name": "lookup_contact", "arguments": [1, 2, 3]},  # list, not dict or JSON string
    ]
    score = game.score_task(record, tool_calls)
    assert score["arguments"] < 1.0