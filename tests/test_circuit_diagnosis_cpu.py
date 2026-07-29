"""CPU tests for the logic-circuit diagnosis demo (issue #193).

Tests cover:
- Circuit generation (seeded, deterministic, valid topology).
- Gate evaluation (AND, OR, NOT, INPUT).
- Fault injection (stuck-at-0, stuck-at-1, only non-INPUT gates).
- Faulty simulation (differs from reference).
- Diagnosis session (probe, submit, action limits, invalid nodes).
- Scoring (correct, incorrect, efficiency).
- Brute-force baseline.
- Prompt formatting.
- Dataset generation and loading.
- Reward function.
- Invalid inputs and boundary values.
- Deterministic output.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add example directory to path.
_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "circuit"

# Check if torch is available (run_agent imports areno.api which needs torch).
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
sys.path.insert(0, str(_EXAMPLE_DIR))

import circuit  # noqa: E402
import dataset_generator  # noqa: E402
import reward  # noqa: E402

# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------


class TestCircuitGeneration(unittest.TestCase):
    """generate_circuit should produce valid, deterministic circuits."""

    def test_generates_correct_number_of_gates(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        self.assertEqual(circ.num_gates, 6)
        self.assertEqual(circ.num_inputs, 3)

    def test_first_gates_are_inputs(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        for i in range(3):
            self.assertEqual(circ.gates[i].gate_type, circuit.GateType.INPUT)

    def test_non_input_gates_are_not_input(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        for gate in circ.gates[3:]:
            self.assertIn(gate.gate_type, (circuit.GateType.AND, circuit.GateType.OR, circuit.GateType.NOT))

    def test_deterministic_with_same_seed(self):
        c1 = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        c2 = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        self.assertEqual(c1.gates, c2.gates)

    def test_different_seeds_different_circuits(self):
        c1 = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        c2 = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=100)
        self.assertNotEqual(c1.gates, c2.gates)

    def test_invalid_num_inputs(self):
        with self.assertRaises(ValueError):
            circuit.generate_circuit(num_inputs=1, num_gates=5)

    def test_invalid_num_gates(self):
        with self.assertRaises(ValueError):
            circuit.generate_circuit(num_inputs=3, num_gates=3)

    def test_input_gate_ids(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        self.assertEqual(circ.input_gate_ids, [0, 1, 2])


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


class TestGateEvaluation(unittest.TestCase):
    """Gate.evaluate should compute correct logic."""

    def test_and_gate(self):
        gate = circuit.Gate(gate_id=2, gate_type=circuit.GateType.AND, inputs=(0, 1))
        wires = [False, True, None]
        self.assertFalse(gate.evaluate(wires))

    def test_or_gate(self):
        gate = circuit.Gate(gate_id=2, gate_type=circuit.GateType.OR, inputs=(0, 1))
        wires = [False, True, None]
        self.assertTrue(gate.evaluate(wires))

    def test_not_gate(self):
        gate = circuit.Gate(gate_id=1, gate_type=circuit.GateType.NOT, inputs=(0,))
        wires = [False, None]
        self.assertTrue(gate.evaluate(wires))


# ---------------------------------------------------------------------------
# Circuit simulation
# ---------------------------------------------------------------------------


class TestCircuitSimulation(unittest.TestCase):
    """Circuit.simulate should produce correct wire values."""

    def setUp(self):
        # Build a simple circuit: inputs A, B; gate2 = A AND B; gate3 = NOT gate2
        self.circ = circuit.Circuit(
            gates=[
                circuit.Gate(0, circuit.GateType.INPUT),
                circuit.Gate(1, circuit.GateType.INPUT),
                circuit.Gate(2, circuit.GateType.AND, (0, 1)),
                circuit.Gate(3, circuit.GateType.NOT, (2,)),
            ],
            num_inputs=2,
            num_outputs=1,
        )

    def test_simulate_and_not(self):
        result = self.circ.simulate([True, True])
        self.assertTrue(result[2])  # A AND B = True
        self.assertFalse(result[3])  # NOT(A AND B) = False

    def test_simulate_false_and_true(self):
        result = self.circ.simulate([False, True])
        self.assertFalse(result[2])  # False AND True = False
        self.assertTrue(result[3])  # NOT False = True

    def test_wrong_input_count(self):
        with self.assertRaises(ValueError):
            self.circ.simulate([True])

    def test_get_wire_value(self):
        val = self.circ.get_wire_value([True, True], 3)
        self.assertFalse(val)


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


class TestFaultInjection(unittest.TestCase):
    """inject_fault should create a faulty circuit with a non-INPUT gate."""

    def setUp(self):
        self.circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)

    def test_faulty_gate_is_not_input(self):
        faulty = circuit.inject_fault(self.circ, seed=0)
        gate = self.circ.gates[faulty.faulty_gate_id]
        self.assertNotEqual(gate.gate_type, circuit.GateType.INPUT)

    def test_fault_type_is_valid(self):
        faulty = circuit.inject_fault(self.circ, seed=0)
        self.assertIn(faulty.fault_type, ("stuck_at_0", "stuck_at_1"))

    def test_deterministic_fault(self):
        f1 = circuit.inject_fault(self.circ, seed=5)
        f2 = circuit.inject_fault(self.circ, seed=5)
        self.assertEqual(f1.faulty_gate_id, f2.faulty_gate_id)
        self.assertEqual(f1.fault_type, f2.fault_type)

    def test_faulty_differs_from_reference(self):
        """At least one input vector should produce different outputs."""
        faulty = circuit.inject_fault(self.circ, seed=0)
        from itertools import product as iter_product

        found_diff = False
        for vec in iter_product([False, True], repeat=self.circ.num_inputs):
            ref = self.circ.simulate(list(vec))
            faulty_out = faulty.simulate_faulty(list(vec))
            if ref != faulty_out:
                found_diff = True
                break
        self.assertTrue(found_diff, "Faulty circuit should differ from reference")

    def test_verify_diagnosis_correct(self):
        faulty = circuit.inject_fault(self.circ, seed=0)
        self.assertTrue(faulty.verify_diagnosis(faulty.faulty_gate_id))

    def test_verify_diagnosis_incorrect(self):
        faulty = circuit.inject_fault(self.circ, seed=0)
        wrong_id = faulty.faulty_gate_id + 1 if faulty.faulty_gate_id + 1 < self.circ.num_gates else 0
        self.assertFalse(faulty.verify_diagnosis(wrong_id))


# ---------------------------------------------------------------------------
# Diagnosis session
# ---------------------------------------------------------------------------


class TestDiagnosisSession(unittest.TestCase):
    """DiagnosisSession should track probes and submissions with limits."""

    def setUp(self):
        self.circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        self.faulty = circuit.inject_fault(self.circ, seed=0)
        self.session = circuit.DiagnosisSession(faulty_circuit=self.faulty, max_probes=5, max_submissions=2)

    def test_initial_state(self):
        self.assertEqual(self.session.num_probes, 0)
        self.assertEqual(self.session.num_submissions, 0)
        self.assertFalse(self.session.solved)
        self.assertEqual(self.session.probes_remaining, 5)
        self.assertEqual(self.session.submissions_remaining, 2)

    def test_probe_returns_value(self):
        result = self.session.probe([True, False, True], wire_id=5)
        self.assertIsInstance(result.value, bool)
        self.assertEqual(result.wire_id, 5)
        self.assertEqual(self.session.num_probes, 1)

    def test_probe_invalid_wire(self):
        with self.assertRaises(ValueError):
            self.session.probe([True, False, True], wire_id=99)

    def test_probe_negative_wire(self):
        with self.assertRaises(ValueError):
            self.session.probe([True, False, True], wire_id=-1)

    def test_probe_limit(self):
        for _ in range(5):
            self.session.probe([True, False, True], wire_id=5)
        with self.assertRaises(ValueError):
            self.session.probe([True, False, True], wire_id=5)

    def test_submit_correct(self):
        result = self.session.submit(self.faulty.faulty_gate_id)
        self.assertTrue(result)
        self.assertTrue(self.session.solved)

    def test_submit_incorrect(self):
        result = self.session.submit(0)
        self.assertFalse(result)
        self.assertFalse(self.session.solved)

    def test_submission_limit(self):
        self.session.submit(0)
        self.session.submit(1)
        with self.assertRaises(ValueError):
            self.session.submit(2)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring(unittest.TestCase):
    """score_diagnosis should reward correct diagnosis with efficiency bonus."""

    def setUp(self):
        self.circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        self.faulty = circuit.inject_fault(self.circ, seed=0)

    def test_solved_few_probes_high_score(self):
        session = circuit.DiagnosisSession(faulty_circuit=self.faulty, max_probes=20)
        session.probe([True, False, True], wire_id=5)
        session.probe([False, True, False], wire_id=4)
        session.submit(self.faulty.faulty_gate_id)
        score = circuit.score_diagnosis(session)
        self.assertGreater(score, 0.9)
        self.assertLessEqual(score, 1.0)

    def test_solved_many_probes_lower_score(self):
        session = circuit.DiagnosisSession(faulty_circuit=self.faulty, max_probes=20)
        for _ in range(18):
            session.probe([True, False, True], wire_id=5)
        session.submit(self.faulty.faulty_gate_id)
        score = circuit.score_diagnosis(session)
        self.assertLess(score, 0.6)

    def test_not_solved_zero_score(self):
        session = circuit.DiagnosisSession(faulty_circuit=self.faulty, max_probes=20)
        session.submit(0)
        score = circuit.score_diagnosis(session)
        self.assertEqual(score, 0.0)


# ---------------------------------------------------------------------------
# Brute-force baseline
# ---------------------------------------------------------------------------


class TestBruteForceBaseline(unittest.TestCase):
    """brute_force_baseline should find the faulty gate."""

    def test_finds_faulty_gate(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        faulty = circuit.inject_fault(circ, seed=0)
        guessed, probes = circuit.brute_force_baseline(faulty, max_probes=50)
        self.assertEqual(guessed, faulty.faulty_gate_id)
        self.assertGreater(probes, 0)

    def test_within_probe_limit(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=10)
        faulty = circuit.inject_fault(circ, seed=5)
        _, probes = circuit.brute_force_baseline(faulty, max_probes=100)
        self.assertLessEqual(probes, 100)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


class TestPromptFormatting(unittest.TestCase):
    """format_prompt should describe the circuit structure and turn limit."""

    def test_prompt_contains_circuit_info(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        prompt = circuit.format_prompt(circ)
        self.assertIn("diagnosing", prompt)
        self.assertIn("3 inputs", prompt)
        self.assertIn("6 gates", prompt)
        self.assertIn("INPUT", prompt)
        self.assertIn("faulty", prompt)
        self.assertIn("probe", prompt)
        self.assertIn("submit", prompt)

    def test_prompt_says_10_turns(self):
        """Prompt should say '10 turns', not '20 probes / 3 submissions'."""
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        prompt = circuit.format_prompt(circ)
        self.assertIn("10 turns", prompt)
        self.assertNotIn("20 probes", prompt)
        self.assertNotIn("3 submissions", prompt)

    def test_prompt_custom_turns(self):
        circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        prompt = circuit.format_prompt(circ, max_turns=5)
        self.assertIn("5 turns", prompt)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


class TestDatasetGeneration(unittest.TestCase):
    """dataset_generator should produce valid, reproducible records."""

    def test_generates_correct_count(self):
        records = dataset_generator.generate_records(count=10, seed=42, num_inputs=3, num_gates=6)
        self.assertEqual(len(records), 10)

    def test_records_have_required_fields(self):
        records = dataset_generator.generate_records(count=5, seed=42)
        for r in records:
            self.assertIn("id", r)
            self.assertIn("num_inputs", r)
            self.assertIn("num_gates", r)
            self.assertIn("gates", r)
            self.assertIn("faulty_gate_id", r)
            self.assertIn("fault_type", r)
            self.assertIn("prompt", r)

    def test_deterministic(self):
        r1 = dataset_generator.generate_records(count=5, seed=42)
        r2 = dataset_generator.generate_records(count=5, seed=42)
        self.assertEqual(r1, r2)

    def test_jsonl_roundtrip(self):
        records = dataset_generator.generate_records(count=5, seed=42)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            dataset_generator.write_jsonl(records, f)
            path = f.name
        try:
            loaded = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        loaded.append(json.loads(line))
            self.assertEqual(len(loaded), 5)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------


class TestRewardFunction(unittest.TestCase):
    """reward_fn should return float 1.0 for correct, 0.0 otherwise."""

    def test_correct_diagnosis(self):
        record = type(
            "R",
            (),
            {
                "source_record": {"faulty_gate_id": 4},
                "tool_calls": [{"name": "submit", "arguments": {"gate_id": 4}}],
                "completion": "",
            },
        )()
        self.assertEqual(reward.reward_fn(record), 1.0)

    def test_incorrect_diagnosis(self):
        record = type(
            "R",
            (),
            {
                "source_record": {"faulty_gate_id": 4},
                "tool_calls": [{"name": "submit", "arguments": {"gate_id": 3}}],
                "completion": "",
            },
        )()
        self.assertEqual(reward.reward_fn(record), 0.0)

    def test_no_submit_call(self):
        record = type(
            "R",
            (),
            {
                "source_record": {"faulty_gate_id": 4},
                "tool_calls": [{"name": "probe", "arguments": {"wire_id": 5}}],
                "completion": "",
            },
        )()
        self.assertEqual(reward.reward_fn(record), 0.0)

    def test_string_arguments(self):
        record = type(
            "R",
            (),
            {
                "source_record": {"faulty_gate_id": 4},
                "tool_calls": [{"name": "submit", "arguments": '{"gate_id": 4}'}],
                "completion": "",
            },
        )()
        self.assertEqual(reward.reward_fn(record), 1.0)

    def test_no_tool_calls(self):
        record = type(
            "R",
            (),
            {
                "source_record": {"faulty_gate_id": 4},
                "tool_calls": [],
                "completion": "I don't know",
            },
        )()
        self.assertEqual(reward.reward_fn(record), 0.0)

    def test_returns_float_not_dict(self):
        """reward_fn must return float, not dict (AReno trainer does float(reward_fn(record)))."""
        record = type(
            "R",
            (),
            {
                "source_record": {"faulty_gate_id": 4},
                "tool_calls": [{"name": "submit", "arguments": {"gate_id": 4}}],
                "completion": "",
            },
        )()
        result = reward.reward_fn(record)
        self.assertIsInstance(result, float)

    def test_analyze_tool_calls_helper(self):
        """analyze_tool_calls is a standalone helper that returns a dict."""
        record = type(
            "R",
            (),
            {
                "source_record": {"faulty_gate_id": 4},
                "tool_calls": [
                    {"name": "probe", "arguments": {"wire_id": 5, "inputs": [True, False, True]}},
                    {"name": "probe", "arguments": {"wire_id": 3, "inputs": [False, True, False]}},
                    {"name": "submit", "arguments": {"gate_id": 4}},
                ],
                "completion": "",
            },
        )()
        info = reward.analyze_tool_calls(record)
        self.assertEqual(info["probes_used"], 2)
        self.assertTrue(info["submitted"])
        self.assertEqual(info["guessed_gate_id"], 4)


# ---------------------------------------------------------------------------
# Boundary values
# ---------------------------------------------------------------------------


class TestBoundaryValues(unittest.TestCase):
    """Edge cases: minimum circuit, single non-input gate."""

    def test_minimum_circuit(self):
        circ = circuit.generate_circuit(num_inputs=2, num_gates=3, seed=1)
        self.assertEqual(circ.num_gates, 3)
        self.assertEqual(circ.num_inputs, 2)

    def test_fault_on_single_gate(self):
        circ = circuit.Circuit(
            gates=[
                circuit.Gate(0, circuit.GateType.INPUT),
                circuit.Gate(1, circuit.GateType.INPUT),
                circuit.Gate(2, circuit.GateType.AND, (0, 1)),
            ],
            num_inputs=2,
            num_outputs=1,
        )
        faulty = circuit.inject_fault(circ, seed=0)
        self.assertEqual(faulty.faulty_gate_id, 2)

    def test_probe_empty_circuit_error(self):
        """inject_fault on a circuit with only INPUT gates should raise."""
        circ = circuit.Circuit(
            gates=[
                circuit.Gate(0, circuit.GateType.INPUT),
                circuit.Gate(1, circuit.GateType.INPUT),
            ],
            num_inputs=2,
            num_outputs=0,
        )
        with self.assertRaises(ValueError):
            circuit.inject_fault(circ, seed=0)


# ---------------------------------------------------------------------------
# Probe parameter validation (run_agent._execute_probe)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_HAS_TORCH, "torch not available; run_agent imports areno.api")
class TestExecuteProbeValidation(unittest.TestCase):
    """_execute_probe should strictly validate inputs and wire_id."""

    def setUp(self):
        self.circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        self.faulty = circuit.inject_fault(self.circ, seed=0)
        # Import _execute_probe from run_agent module.
        sys.path.insert(0, str(_EXAMPLE_DIR))
        import run_agent  # noqa: E402

        self._execute_probe = run_agent._execute_probe

    def test_valid_probe(self):
        result = self._execute_probe(self.faulty, {"inputs": [True, False, True], "wire_id": 5})
        self.assertIn("value", result)
        self.assertNotIn("error", result)

    def test_string_false_not_accepted(self):
        """bool('false') is True in Python — must reject non-bool inputs."""
        result = self._execute_probe(self.faulty, {"inputs": ["false", "true", "false"], "wire_id": 5})
        self.assertIn("error", result)
        self.assertNotIn("value", result)

    def test_wrong_input_length(self):
        result = self._execute_probe(self.faulty, {"inputs": [True, False], "wire_id": 5})
        self.assertIn("error", result)
        self.assertIn("length", result["error"])

    def test_inputs_not_list(self):
        result = self._execute_probe(self.faulty, {"inputs": True, "wire_id": 5})
        self.assertIn("error", result)

    def test_wire_id_out_of_range(self):
        result = self._execute_probe(self.faulty, {"inputs": [True, False, True], "wire_id": 99})
        self.assertIn("error", result)
        self.assertIn("out of range", result["error"])

    def test_wire_id_negative(self):
        result = self._execute_probe(self.faulty, {"inputs": [True, False, True], "wire_id": -1})
        self.assertIn("error", result)

    def test_wire_id_string(self):
        result = self._execute_probe(self.faulty, {"inputs": [True, False, True], "wire_id": "5"})
        self.assertIn("value", result)
        self.assertNotIn("error", result)

    def test_wire_id_invalid_string(self):
        result = self._execute_probe(self.faulty, {"inputs": [True, False, True], "wire_id": "abc"})
        self.assertIn("error", result)


# ---------------------------------------------------------------------------
# Run agent multi-turn tests with fake OpenAI client
# ---------------------------------------------------------------------------


class _FakeToolCall:
    """Mimics openai ToolCall object."""

    def __init__(self, name, arguments, call_id="call_0"):
        self.id = call_id
        self.type = "function"
        self.function = MagicMock()
        self.function.name = name
        self.function.arguments = arguments


class _FakeMessage:
    """Mimics openai Message object."""

    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    """Mimics openai Choice object."""

    def __init__(self, message):
        self.message = message
        self.finish_reason = "tool_calls" if message.tool_calls else "stop"


class _FakeResponse:
    """Mimics openai ChatCompletion response with Areno trajectory metadata."""

    def __init__(self, message):
        self.choices = [_FakeChoice(message)]
        # AgentTrajectoryTurn.__post_init__ requires response.areno metadata
        # with response_tokens and response_logprobs lists.
        self.areno = {
            "response_tokens": [1, 2, 3],
            "response_logprobs": [-0.1, -0.2, -0.3],
        }


class _FakeOpenAIClient:
    """Fake OpenAI client that returns pre-scripted responses.

    Usage: pass a list of _FakeResponse objects to cycle through.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    class _Completions:
        def __init__(self, parent):
            self.parent = parent

        async def create(self, **kwargs):
            resp = self.parent._responses.pop(0)
            self.parent.calls += 1
            return resp

    class _Chat:
        def __init__(self, parent):
            self.completions = _FakeOpenAIClient._Completions(parent)

    @property
    def chat(self):
        return self._Chat(self)

    async def close(self):
        pass


# Need to also define _FakeChoice finish_reason correctly
def _make_response(content=None, tool_calls=None):
    """Create a fake response with optional tool calls."""
    msg = _FakeMessage(content=content, tool_calls=tool_calls)
    resp = _FakeResponse(msg)
    return resp


@unittest.skipUnless(_HAS_TORCH, "torch not available; run_agent imports areno.api")
class TestRunAgentMultiTurn(unittest.TestCase):
    """Test run_agent multi-turn interaction with fake OpenAI client."""

    def setUp(self):
        self.circ = circuit.generate_circuit(num_inputs=3, num_gates=6, seed=42)
        self.faulty = circuit.inject_fault(self.circ, seed=0)
        self.source_record = {
            "num_inputs": 3,
            "num_gates": 6,
            "gates": [
                {"gate_id": g.gate_id, "gate_type": g.gate_type.value, "inputs": list(g.inputs)}
                for g in self.circ.gates
            ],
            "faulty_gate_id": self.faulty.faulty_gate_id,
            "fault_type": self.faulty.fault_type,
            "prompt": circuit.format_prompt(self.circ),
        }

    def _make_item(self):
        """Create a fake batch item."""
        item = MagicMock()
        item.prompt = self.source_record["prompt"]
        item.source_record = self.source_record
        return item

    def _make_ctx(self, client):
        """Create a fake agent context."""
        ctx = MagicMock()
        ctx.max_running_prompts = 1
        ctx.get_base_url.return_value = "http://fake/v1"
        ctx.api_key = "fake"
        return ctx

    def _make_batch(self, items):
        batch = MagicMock()
        batch.iter_samples.return_value = iter(items)
        return batch

    def test_probe_then_submit_flow(self):
        """Turn 1: probe → Turn 2: submit → ends."""
        import run_agent  # noqa: E402

        item = self._make_item()
        responses = [
            _make_response(
                tool_calls=[
                    _FakeToolCall("probe", '{"inputs": [true, false, true], "wire_id": 5}', "c1"),
                ]
            ),
            _make_response(
                tool_calls=[
                    _FakeToolCall("submit", f'{{"gate_id": {self.faulty.faulty_gate_id}}}', "c2"),
                ]
            ),
        ]
        client = _FakeOpenAIClient(responses)
        batch = self._make_batch([item])
        ctx = self._make_ctx(client)

        with patch("httpx.AsyncClient"), patch("openai.AsyncOpenAI", return_value=client):
            traj = asyncio.run(run_agent.run_agent(ctx, batch))

        # Should have 2 turns (probe + submit).
        self.assertEqual(len(traj.turns), 2)
        # Turn 1 messages should not contain tool result yet.
        # Turn 2 messages should contain the probe tool result.
        turn2_messages = traj.turns[1].messages
        tool_msgs = [m for m in turn2_messages if m.get("role") == "tool"]
        self.assertTrue(len(tool_msgs) >= 1)
        self.assertEqual(tool_msgs[0]["name"], "probe")

    def test_no_tool_call_nudge(self):
        """Model returns no tool call → nudge message appended → next turn works."""
        import run_agent  # noqa: E402

        item = self._make_item()
        responses = [
            _make_response(content="I need to think..."),  # No tool call
            _make_response(
                tool_calls=[
                    _FakeToolCall("submit", f'{{"gate_id": {self.faulty.faulty_gate_id}}}', "c1"),
                ]
            ),
        ]
        client = _FakeOpenAIClient(responses)
        batch = self._make_batch([item])
        ctx = self._make_ctx(client)

        with patch("httpx.AsyncClient"), patch("openai.AsyncOpenAI", return_value=client):
            traj = asyncio.run(run_agent.run_agent(ctx, batch))

        # 2 turns: no-tool + submit.
        self.assertEqual(len(traj.turns), 2)
        # Turn 2 messages should have the nudge user message.
        turn2_messages = traj.turns[1].messages
        user_msgs = [m for m in turn2_messages if m.get("role") == "user"]
        self.assertTrue(any("did not include a tool call" in m.get("content", "") for m in user_msgs))

    def test_multiple_tool_calls_only_first_executed(self):
        """If model returns 2 tool calls, only first is executed."""
        import run_agent  # noqa: E402

        item = self._make_item()
        # Model returns probe + submit in same response.
        # Only probe should execute; submit should NOT be recorded.
        responses = [
            _make_response(
                tool_calls=[
                    _FakeToolCall("probe", '{"inputs": [true, false, true], "wire_id": 5}', "c1"),
                    _FakeToolCall("submit", f'{{"gate_id": {self.faulty.faulty_gate_id}}}', "c2"),
                ]
            ),
            _make_response(
                tool_calls=[
                    _FakeToolCall("submit", f'{{"gate_id": {self.faulty.faulty_gate_id}}}', "c3"),
                ]
            ),
        ]
        client = _FakeOpenAIClient(responses)
        batch = self._make_batch([item])
        ctx = self._make_ctx(client)

        with patch("httpx.AsyncClient"), patch("openai.AsyncOpenAI", return_value=client):
            traj = asyncio.run(run_agent.run_agent(ctx, batch))

        # Turn 1: probe executed, submit ignored. Turn 2: submit.
        self.assertEqual(len(traj.turns), 2)
        # Turn 1 parsed_tool_calls should only contain probe, NOT submit.
        turn1_parsed = traj.turns[0].parsed_tool_calls
        self.assertEqual(len(turn1_parsed), 1)
        self.assertEqual(turn1_parsed[0]["function"]["name"], "probe")
        # After turn 1, we should see a probe tool message, not a submit.
        tool_msgs = []
        for m in traj.turns[1].messages:  # Turn 2's messages include turn 1 history
            if m.get("role") == "tool":
                tool_msgs.append(m)
        # First tool message should be probe.
        self.assertEqual(tool_msgs[0]["name"], "probe")

    def test_invalid_probe_returns_error_not_crash(self):
        """Invalid probe params should return error tool message, not crash."""
        import run_agent  # noqa: E402

        item = self._make_item()
        responses = [
            _make_response(
                tool_calls=[
                    _FakeToolCall("probe", '{"inputs": [true, false], "wire_id": 5}', "c1"),
                ]
            ),
            _make_response(
                tool_calls=[
                    _FakeToolCall("submit", f'{{"gate_id": {self.faulty.faulty_gate_id}}}', "c2"),
                ]
            ),
        ]
        client = _FakeOpenAIClient(responses)
        batch = self._make_batch([item])
        ctx = self._make_ctx(client)

        with patch("httpx.AsyncClient"), patch("openai.AsyncOpenAI", return_value=client):
            traj = asyncio.run(run_agent.run_agent(ctx, batch))

        # Should not crash; turn 2 messages should contain error tool message.
        tool_msgs = [m for m in traj.turns[1].messages if m.get("role") == "tool"]
        self.assertTrue(len(tool_msgs) >= 1)
        # The probe result should contain an error.
        probe_result = json.loads(tool_msgs[0]["content"])
        self.assertIn("error", probe_result)

    def test_submit_ends_conversation(self):
        """submit on turn 1 should end immediately with 1 turn."""
        import run_agent  # noqa: E402

        item = self._make_item()
        responses = [
            _make_response(
                tool_calls=[
                    _FakeToolCall("submit", f'{{"gate_id": {self.faulty.faulty_gate_id}}}', "c1"),
                ]
            ),
        ]
        client = _FakeOpenAIClient(responses)
        batch = self._make_batch([item])
        ctx = self._make_ctx(client)

        with patch("httpx.AsyncClient"), patch("openai.AsyncOpenAI", return_value=client):
            traj = asyncio.run(run_agent.run_agent(ctx, batch))

        # Only 1 turn (submit ends immediately).
        self.assertEqual(len(traj.turns), 1)
        # Client should only be called once.
        self.assertEqual(client.calls, 1)

    def test_max_turns_exhausted(self):
        """If model never calls submit, should stop after MAX_TURNS."""
        import run_agent  # noqa: E402

        item = self._make_item()
        # 10 probe responses, no submit.
        responses = [
            _make_response(
                tool_calls=[
                    _FakeToolCall("probe", '{"inputs": [true, false, true], "wire_id": 5}', f"c{i}"),
                ]
            )
            for i in range(15)  # More than MAX_TURNS
        ]
        client = _FakeOpenAIClient(responses)
        batch = self._make_batch([item])
        ctx = self._make_ctx(client)

        with patch("httpx.AsyncClient"), patch("openai.AsyncOpenAI", return_value=client):
            traj = asyncio.run(run_agent.run_agent(ctx, batch))

        # Should stop at MAX_TURNS=10.
        self.assertEqual(len(traj.turns), 10)
        self.assertEqual(client.calls, 10)
