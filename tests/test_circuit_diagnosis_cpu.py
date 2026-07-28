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

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add example directory to path.
_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "agentic" / "circuit"
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
    """format_prompt should describe the circuit structure."""

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
    """reward_fn should return 1.0 for correct diagnosis, 0.0 otherwise."""

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


if __name__ == "__main__":
    unittest.main()
