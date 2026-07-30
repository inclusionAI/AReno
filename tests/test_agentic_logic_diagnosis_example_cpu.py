"""CPU tests for the logic-circuit diagnosis agentic example."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "logic_diagnosis"


def _load_module(name: str):
    """Dynamically import a module from the example directory."""
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
        spec = importlib.util.spec_from_file_location(
            f"agentic_logic_diag_{name}_for_tests", path
        )
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


# ---------------------------------------------------------------------------
# Circuit generation tests
# ---------------------------------------------------------------------------
def test_circuit_generation_is_reproducible():
    game = _load_module("game")
    a = game.generate_circuit(4, 8, seed=42)
    b = game.generate_circuit(4, 8, seed=42)
    assert a == b


def test_circuit_is_acyclic():
    game = _load_module("game")
    for seed in range(20):
        nodes = game.generate_circuit(4, 8, seed=seed)
        # All edges must go from smaller id to larger id
        for node in nodes:
            for inp in node["inputs"]:
                assert inp < node["id"], f"back-edge: {inp} → {node['id']}"


def test_circuit_has_output():
    game = _load_module("game")
    for seed in range(20):
        nodes = game.generate_circuit(4, 8, seed=seed)
        outputs = [n for n in nodes if n["type"] == "output"]
        assert len(outputs) == 1


def test_circuit_types_and_arity():
    game = _load_module("game")
    nodes = game.generate_circuit(4, 8, seed=42)
    for node in nodes:
        if node["type"] == "not":
            assert len(node["inputs"]) == 1, f"NOT gate {node['id']} has {len(node['inputs'])} inputs"
        elif node["type"] in ("and", "or"):
            assert len(node["inputs"]) >= 1, f"{node['type']} gate {node['id']} has no inputs"
        elif node["type"] == "input":
            assert node["inputs"] == []
        elif node["type"] == "output":
            assert len(node["inputs"]) >= 1


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------
def test_evaluate_and_or_not():
    game = _load_module("game")
    # Circuit: IN0 AND IN1 → OUT
    nodes = [
        {"id": 0, "type": "input", "inputs": []},
        {"id": 1, "type": "input", "inputs": []},
        {"id": 2, "type": "and", "inputs": [0, 1]},
        {"id": 3, "type": "output", "inputs": [2]},
    ]
    v = game.evaluate(nodes, [False, False])
    assert v[3] is False
    v = game.evaluate(nodes, [True, True])
    assert v[3] is True

    # NOT gate
    nodes = [
        {"id": 0, "type": "input", "inputs": []},
        {"id": 1, "type": "not", "inputs": [0]},
        {"id": 2, "type": "output", "inputs": [1]},
    ]
    assert game.evaluate(nodes, [False])[2] is True
    assert game.evaluate(nodes, [True])[2] is False


def test_fault_overrides_output():
    game = _load_module("game")
    nodes = [
        {"id": 0, "type": "input", "inputs": []},
        {"id": 1, "type": "input", "inputs": []},
        {"id": 2, "type": "and", "inputs": [0, 1]},
        {"id": 3, "type": "output", "inputs": [2]},
    ]
    # Without fault: True AND True = True
    assert game.evaluate(nodes, [True, True])[3] is True
    # With stuck-at-0 on the AND gate
    fault = {"node": 2, "stuck_value": 0}
    assert game.evaluate(nodes, [True, True], fault)[3] is False
    # With stuck-at-1 on the AND gate
    fault = {"node": 2, "stuck_value": 1}
    assert game.evaluate(nodes, [False, False], fault)[3] is True


# ---------------------------------------------------------------------------
# Diagnosis verification
# ---------------------------------------------------------------------------
def test_verify_diagnosis_correct_and_wrong():
    game = _load_module("game")
    nodes: list = []
    fault = {"node": 5, "stuck_value": 0}
    assert game.verify_diagnosis(nodes, fault, 5, "stuck_at_0") is True
    assert game.verify_diagnosis(nodes, fault, 5, "stuck_at_1") is False
    assert game.verify_diagnosis(nodes, fault, 3, "stuck_at_0") is False


def test_brute_force_finds_unique():
    game = _load_module("game")
    # Try multiple seeds — small circuits may be genuinely ambiguous;
    # the generator's retry loop handles this, and brute_force correctly
    # detects ambiguity. We search for a seed that produces a unique circuit.
    for seed in range(50):
        nodes = game.generate_circuit(4, 6, seed=seed)
        gate_count = sum(1 for n in nodes if n["type"] in ("and", "or", "not"))
        if gate_count < 2:
            continue
        fault = game.inject_fault(nodes, seed=seed + 100)
        if game.brute_force_verify(nodes, fault):
            return  # success
    raise AssertionError("no unique circuit found in 50 seeds")


# ---------------------------------------------------------------------------
# Circuit pruning
# ---------------------------------------------------------------------------
def test_prune_removes_dead_nodes():
    game = _load_module("game")
    # Circuit with a dead gate chain (nodes 3→4, but output only from node 2)
    nodes = [
        {"id": 0, "type": "input", "inputs": []},
        {"id": 1, "type": "input", "inputs": []},
        {"id": 2, "type": "and", "inputs": [0, 1]},
        {"id": 3, "type": "not", "inputs": [0]},     # dead — output not reachable from here
        {"id": 4, "type": "or", "inputs": [3]},       # dead — depends on dead node
        {"id": 5, "type": "output", "inputs": [2]},
    ]
    pruned = game._prune_unreachable(nodes)
    # After renumbering, live nodes should be: 2 inputs + 1 AND + 1 output = 4 nodes
    # Dead NOT (old 3) and dead OR (old 4) should be removed
    types = {t: sum(1 for node in pruned if node["type"] == t) for t in ("input", "and", "or", "not", "output")}
    assert types.get("not", 0) == 0, "dead NOT gate should be removed"
    assert types.get("or", 0) == 0, "dead OR gate should be removed"
    assert types.get("input", 0) == 2, "both inputs should be kept"
    assert types.get("and", 0) == 1, "live AND gate should be kept"
    assert types.get("output", 0) == 1, "output should be kept"


# ---------------------------------------------------------------------------
# Generator and loader
# ---------------------------------------------------------------------------
def test_generator_is_reproducible():
    generator = _load_module("dataset_generator")
    a = generator.generate_records(8, seed=4)
    b = generator.generate_records(8, seed=4)
    assert a == b


def test_generator_produces_valid_records():
    game = _load_module("game")
    generator = _load_module("dataset_generator")
    records = generator.generate_records(24, seed=7)
    assert len(records) == 24
    record_keys = set()
    input_counts = {}
    gate_counts = {}
    fault_counts = {}
    for r in records:
        assert "nodes" in r
        assert "fault" in r
        assert "n_inputs" in r
        assert "n_gates" in r
        assert "prompt" in r
        assert r["fault"]["node"] >= 0
        gate_nodes = [node for node in r["nodes"] if node["type"] in ("and", "or", "not")]
        assert r["n_gates"] == len(gate_nodes)
        assert generator.DATASET_MIN_GATES <= r["n_gates"] <= generator.DATASET_MAX_GATES
        assert game.brute_force_verify(r["nodes"], r["fault"])
        assert generator._fault_changes_output(r["nodes"], r["fault"])

        record_key = json.dumps({"nodes": r["nodes"], "fault": r["fault"]}, sort_keys=True)
        assert record_key not in record_keys
        record_keys.add(record_key)
        input_counts[r["n_inputs"]] = input_counts.get(r["n_inputs"], 0) + 1
        gate_counts[r["n_gates"]] = gate_counts.get(r["n_gates"], 0) + 1
        fault_type = next(node["type"] for node in gate_nodes if node["id"] == r["fault"]["node"])
        fault_class = (fault_type, r["fault"]["stuck_value"])
        fault_counts[fault_class] = fault_counts.get(fault_class, 0) + 1

    assert max(input_counts.values()) - min(input_counts.values()) <= 1
    assert max(gate_counts.values()) - min(gate_counts.values()) <= 1
    assert max(fault_counts.values()) - min(fault_counts.values()) <= 1


def test_generator_rejects_negative_record_count():
    generator = _load_module("dataset_generator")
    try:
        generator.generate_records(-1)
    except ValueError as exc:
        assert str(exc) == "count must be non-negative"
    else:
        raise AssertionError("negative record counts must be rejected")


def test_loader_hides_fault_from_prompt():
    generator = _load_module("dataset_generator")
    loader = _load_module("dataset_loader")
    rows = generator.generate_records(8, seed=4)
    records = loader.load_training_dataset("unused", default_loader=lambda _: rows)
    for record in records:
        # The specific fault node id must not appear in the prompt
        # (circuit topology does contain node IDs, but the fault annotation should not)
        prompt = record["prompt"]
        fault_node = record["fault"]["node"]
        fault_type = "stuck_at_0" if record["fault"]["stuck_value"] == 0 else "stuck_at_1"
        # Check that the prompt doesn't explicitly say which node is faulty
        assert f"node {fault_node} is faulty" not in prompt.lower()
        assert f"faulty gate is {fault_node}" not in prompt.lower()
        assert f"fault at node {fault_node}" not in prompt.lower()
        # The specific fault_type should not appear as a diagnosis
        assert f"the fault is {fault_type}" not in prompt.lower()


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------
def test_tool_schemas_are_closed():
    game = _load_module("game")
    for tool in [game.SET_INPUT_VECTOR_TOOL, game.INSPECT_NODE_TOOL, game.SUBMIT_DIAGNOSIS_TOOL]:
        params = tool["function"]["parameters"]
        assert params["additionalProperties"] is False


def test_tool_schemas_have_required_fields():
    game = _load_module("game")
    assert game.SET_INPUT_VECTOR_TOOL["function"]["parameters"]["required"] == ["inputs"]
    assert game.INSPECT_NODE_TOOL["function"]["parameters"]["required"] == ["node_id"]
    assert game.SUBMIT_DIAGNOSIS_TOOL["function"]["parameters"]["required"] == ["node_id", "fault_type"]


# ---------------------------------------------------------------------------
# Tool execution (unit tests on helpers)
# ---------------------------------------------------------------------------
def _make_record(nodes=None, fault=None):
    if nodes is None:
        nodes = [
            {"id": 0, "type": "input", "inputs": []},
            {"id": 1, "type": "input", "inputs": []},
            {"id": 2, "type": "and", "inputs": [0, 1]},
            {"id": 3, "type": "output", "inputs": [2]},
        ]
    if fault is None:
        fault = {"node": 2, "stuck_value": 0}
    return {"nodes": nodes, "fault": fault, "max_probes": 10}


def test_set_input_vector_valid_and_invalid():
    game = _load_module("game")
    run_agent = _load_module("run_agent")
    record = _make_record()
    state: dict = {"input_vector": None, "probes_used": 0}

    # Valid
    result = run_agent._execute_tool(
        _fake_assistant("set_input_vector", {"inputs": [True, False]}),
        record["nodes"], record["fault"], state,
    )
    assert result is not None
    assert "error" not in result
    assert result["output_value"] is False  # stuck-at-0 AND
    assert state["input_vector"] == [True, False]

    # Invalid
    result = run_agent._execute_tool(
        _fake_assistant("set_input_vector", {"inputs": "not-a-list"}),
        record["nodes"], record["fault"], {"input_vector": None, "probes_used": 0},
    )
    assert result is not None and "error" in result


def test_inspect_node_valid_and_invalid():
    run_agent = _load_module("run_agent")
    record = _make_record()
    state = {"input_vector": [True, True], "probes_used": 0}

    # Valid probe
    result = run_agent._execute_tool(
        _fake_assistant("inspect_node", {"node_id": 2}),
        record["nodes"], record["fault"], state,
    )
    assert result is not None
    assert result["probed_value"] is False  # stuck-at-0
    assert state["probes_used"] == 1

    # Invalid node id
    result = run_agent._execute_tool(
        _fake_assistant("inspect_node", {"node_id": 99}),
        record["nodes"], record["fault"], {"input_vector": [True, True], "probes_used": 1},
    )
    assert result is not None and "error" in result

    # Probe without setting input first
    result = run_agent._execute_tool(
        _fake_assistant("inspect_node", {"node_id": 2}),
        record["nodes"], record["fault"], {"input_vector": None, "probes_used": 0},
    )
    assert result is not None and "error" in result


def test_submit_diagnosis_result():
    run_agent = _load_module("run_agent")
    record = _make_record()
    state: dict = {"input_vector": None, "probes_used": 3, "diagnosis_submitted": False}

    # Correct
    result = run_agent._execute_tool(
        _fake_assistant("submit_diagnosis", {"node_id": 2, "fault_type": "stuck_at_0"}),
        record["nodes"], record["fault"], state,
    )
    assert result is not None
    assert result["correct"] is True

    # Wrong
    state["diagnosis_submitted"] = False
    result = run_agent._execute_tool(
        _fake_assistant("submit_diagnosis", {"node_id": 2, "fault_type": "stuck_at_1"}),
        record["nodes"], record["fault"], state,
    )
    assert result is not None
    assert result["correct"] is False


def test_unknown_tool_returns_error():
    run_agent = _load_module("run_agent")
    record = _make_record()
    result = run_agent._execute_tool(
        _fake_assistant("nonexistent_tool", {}),
        record["nodes"], record["fault"], {"input_vector": None, "probes_used": 0},
    )
    assert result is not None and "error" in result


def test_missing_tool_call_returns_none():
    run_agent = _load_module("run_agent")
    record = _make_record()
    result = run_agent._execute_tool(
        {"tool_calls": []},
        record["nodes"], record["fault"], {"input_vector": None, "probes_used": 0},
    )
    assert result is None


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
def test_reward_correct_diagnosis():
    reward = _load_module("reward")
    fault = {"node": 5, "stuck_value": 0}
    record = SimpleNamespace(
        source_record={"fault": fault},
        tool_calls=[
            {"name": "set_input_vector", "arguments": json.dumps({"inputs": [True, False]})},
            {"name": "submit_diagnosis", "arguments": json.dumps({"node_id": 5, "fault_type": "stuck_at_0"})},
        ],
    )
    score = reward.reward_fn(record)
    assert score >= 0.99, f"expected ~1.0, got {score}"  # 0 probes


def test_reward_wrong_diagnosis():
    reward = _load_module("reward")
    fault = {"node": 5, "stuck_value": 0}
    record = SimpleNamespace(
        source_record={"fault": fault},
        tool_calls=[
            {"name": "submit_diagnosis", "arguments": json.dumps({"node_id": 3, "fault_type": "stuck_at_1"})},
        ],
    )
    assert reward.reward_fn(record) == 0.0


def test_reward_no_submission():
    reward = _load_module("reward")
    # Empty completion → -1.0
    record = SimpleNamespace(
        source_record={"fault": {"node": 5, "stuck_value": 0}},
        tool_calls=[],
        completion="",
    )
    assert reward.reward_fn(record) == -1.0
    # Some text but no tool calls → between -0.5 and -0.3
    record_text = SimpleNamespace(
        source_record={"fault": {"node": 5, "stuck_value": 0}},
        tool_calls=[],
        completion='{"inputs": [true, false]}',
    )
    assert -0.5 <= reward.reward_fn(record_text) <= -0.3


def test_reward_probes_reduce_score():
    reward = _load_module("reward")
    fault = {"node": 5, "stuck_value": 0}

    # Correct with 10 probes → min score for correct diagnosis
    record_many = SimpleNamespace(
        source_record={"fault": fault},
        tool_calls=[
            {"name": "inspect_node", "arguments": "{}"},
        ] * 10
        + [{"name": "submit_diagnosis", "arguments": json.dumps({"node_id": 5, "fault_type": "stuck_at_0"})}],
    )
    score_many = reward.reward_fn(record_many)

    # Correct with 0 probes → max score
    record_few = SimpleNamespace(
        source_record={"fault": fault},
        tool_calls=[
            {"name": "submit_diagnosis", "arguments": json.dumps({"node_id": 5, "fault_type": "stuck_at_0"})},
        ],
    )
    score_few = reward.reward_fn(record_few)

    assert score_few > score_many, f"{score_few} should be > {score_many}"


# ---------------------------------------------------------------------------
# Episode flow (async test with fakes)
# ---------------------------------------------------------------------------
def test_episode_stops_on_submit_diagnosis():
    run_agent = _load_module("run_agent")

    nodes = [
        {"id": 0, "type": "input", "inputs": []},
        {"id": 1, "type": "input", "inputs": []},
        {"id": 2, "type": "and", "inputs": [0, 1]},
        {"id": 3, "type": "output", "inputs": [2]},
    ]
    fault = {"node": 2, "stuck_value": 0}

    class FakeCompletions:
        def __init__(self, responses):
            self._responses = iter(responses)
            self.messages = []

        async def create(self, **kwargs):
            self.messages.append(kwargs["messages"])
            return next(self._responses)

    def make_response(tool_name, arguments):
        call = SimpleNamespace(
            id=f"call-{tool_name}",
            type="function",
            function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments)),
        )
        msg = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    def make_text_response(text):
        msg = SimpleNamespace(content=text, tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    item = SimpleNamespace(
        prompt="Diagnose the circuit.",
        record={"nodes": nodes, "fault": fault, "max_probes": 10},
    )

    completions = FakeCompletions([
        make_response("set_input_vector", {"inputs": [True, True]}),
        make_response("inspect_node", {"node_id": 2}),
        make_response("submit_diagnosis", {"node_id": 2, "fault_type": "stuck_at_0"}),
        make_text_response("I found the fault at node 2."),
    ])
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turns = asyncio.run(run_agent._run_episode(item, fake_client))

    # Should have: set_input, inspect, submit, summary = 4 turns
    assert len(turns) == 4
    # Verify tool messages were accumulated correctly
    assert len(completions.messages) >= 3


def test_brute_force_baseline_on_small_circuit():
    game = _load_module("game")

    # Build a simple 1-gate circuit and verify brute force works
    nodes = [
        {"id": 0, "type": "input", "inputs": []},
        {"id": 1, "type": "input", "inputs": []},
        {"id": 2, "type": "or", "inputs": [0, 1]},
        {"id": 3, "type": "output", "inputs": [2]},
    ]
    fault = {"node": 2, "stuck_value": 0}
    assert game.brute_force_verify(nodes, fault) is True

    # Verify the OR gate stuck-at-0 means output is always 0
    for bits in [(False, False), (False, True), (True, False), (True, True)]:
        values = game.evaluate(nodes, list(bits), fault)
        assert values[3] is False, f"stuck-at-0 OR should output 0 for {bits}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fake_assistant(tool_name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": f"call-{tool_name}",
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(arguments)},
            }
        ],
    }
