"""Logic-circuit diagnosis game engine (issue #193).

Generates small AND/OR/NOT circuits with one injected faulty gate. The agent
can set input vectors, inspect selected node outputs, and submit the faulty
gate, while the reference (correct) circuit is kept hidden.

Circuits are represented as a list of gates, each with a type (AND, OR, NOT,
INPUT) and input wire indices. Wires are indexed 0..N-1; gate i produces
wire i. INPUT gates have no inputs and are set by the agent.

Fault injection replaces one gate's output with its complement (stuck-at
fault), and the agent must identify which gate is faulty by probing outputs.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Gate types
# ---------------------------------------------------------------------------


class GateType(str, Enum):
    """Supported gate types for circuit generation."""

    INPUT = "INPUT"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"


# Gate truth tables: input tuple -> output.
_GATE_LOGIC: dict[GateType, Any] = {
    GateType.AND: lambda a, b: a and b,
    GateType.OR: lambda a, b: a or b,
    GateType.NOT: lambda a: not a,
}


# ---------------------------------------------------------------------------
# Circuit representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """A single logic gate in the circuit.

    Attributes:
        gate_id: Wire index (0-based) that this gate drives.
        gate_type: The logic function (AND, OR, NOT, INPUT).
        inputs: Tuple of input wire indices. INPUT gates have empty inputs.
    """

    gate_id: int
    gate_type: GateType
    inputs: tuple[int, ...] = ()

    def evaluate(self, wire_values: list[bool | None]) -> bool:
        """Evaluate this gate's output given current wire values.

        Args:
            wire_values: List of wire values (None = not yet computed).

        Returns:
            The boolean output of this gate.

        Raises:
            ValueError: If input wire values are None (not yet computed).
        """

        if self.gate_type is GateType.INPUT:
            # INPUT value should already be set in wire_values.
            val = wire_values[self.gate_id]
            if val is None:
                raise ValueError(f"INPUT gate {self.gate_id} has no value set")
            return val
        # Get input values.
        input_vals = []
        for inp in self.inputs:
            v = wire_values[inp]
            if v is None:
                raise ValueError(f"wire {inp} has no value (needed by gate {self.gate_id})")
            input_vals.append(v)
        if self.gate_type is GateType.NOT:
            return _GATE_LOGIC[GateType.NOT](input_vals[0])
        return _GATE_LOGIC[self.gate_type](input_vals[0], input_vals[1])


@dataclass
class Circuit:
    """A logic circuit composed of gates.

    Attributes:
        gates: Ordered list of Gate objects (index = wire index).
        num_inputs: Number of INPUT gates.
        num_outputs: Number of output wires (last N wires).
    """

    gates: list[Gate]
    num_inputs: int
    num_outputs: int = 1

    @property
    def num_gates(self) -> int:
        return len(self.gates)

    @property
    def input_gate_ids(self) -> list[int]:
        """Wire indices of all INPUT gates."""

        return [g.gate_id for g in self.gates if g.gate_type is GateType.INPUT]

    @property
    def output_gate_ids(self) -> list[int]:
        """Wire indices of output wires (last num_outputs gates)."""

        return list(range(len(self.gates) - self.num_outputs, len(self.gates)))

    def simulate(self, input_values: list[bool]) -> list[bool]:
        """Simulate the circuit with given input values.

        Args:
            input_values: Boolean values for each INPUT gate, in order.

        Returns:
            List of all wire values (index = wire index).

        Raises:
            ValueError: If input count doesn't match.
        """

        if len(input_values) != self.num_inputs:
            raise ValueError(f"expected {self.num_inputs} inputs, got {len(input_values)}")
        wires: list[bool | None] = [None] * self.num_gates
        # Set input values.
        for i, gate_id in enumerate(self.input_gate_ids):
            wires[gate_id] = input_values[i]
        # Evaluate gates in topological order (they are already ordered).
        for gate in self.gates:
            if gate.gate_type is GateType.INPUT:
                continue
            wires[gate.gate_id] = gate.evaluate(wires)
        return [w for w in wires if w is not None]  # type: ignore[list-item]

    def get_wire_value(self, input_values: list[bool], wire_id: int) -> bool:
        """Get the value of a specific wire for given inputs.

        Args:
            input_values: Boolean values for each INPUT gate.
            wire_id: The wire index to inspect.

        Returns:
            The boolean value of the wire.

        Raises:
            ValueError: If wire_id is out of range.
        """

        if wire_id < 0 or wire_id >= self.num_gates:
            raise ValueError(f"wire_id {wire_id} out of range [0, {self.num_gates})")
        all_values = self.simulate(input_values)
        return all_values[wire_id]


# ---------------------------------------------------------------------------
# Fault injection
# ---------------------------------------------------------------------------


@dataclass
class FaultyCircuit:
    """A circuit with one injected stuck-at fault.

    The reference (correct) circuit is hidden from the agent. The agent
    observes outputs from the faulty circuit and must identify the faulty
    gate.

    Attributes:
        reference: The correct circuit (hidden from agent).
        faulty_gate_id: The gate index that is faulty.
        fault_type: "stuck_at_0" or "stuck_at_1".
    """

    reference: Circuit
    faulty_gate_id: int
    fault_type: str = "stuck_at_0"

    def simulate_faulty(self, input_values: list[bool]) -> list[bool]:
        """Simulate the faulty circuit.

        The faulty gate's output is forced to the stuck-at value.

        Args:
            input_values: Boolean values for each INPUT gate.

        Returns:
            List of all wire values from the faulty simulation.
        """

        if len(input_values) != self.reference.num_inputs:
            raise ValueError(f"expected {self.reference.num_inputs} inputs, got {len(input_values)}")
        wires: list[bool | None] = [None] * self.reference.num_gates
        for i, gate_id in enumerate(self.reference.input_gate_ids):
            wires[gate_id] = input_values[i]
        stuck_value = self.fault_type == "stuck_at_1"
        for gate in self.reference.gates:
            if gate.gate_type is GateType.INPUT:
                continue
            if gate.gate_id == self.faulty_gate_id:
                wires[gate.gate_id] = stuck_value
            else:
                wires[gate.gate_id] = gate.evaluate(wires)
        return [w for w in wires if w is not None]  # type: ignore[list-item]

    def get_faulty_wire_value(self, input_values: list[bool], wire_id: int) -> bool:
        """Get a wire value from the faulty circuit."""

        if wire_id < 0 or wire_id >= self.reference.num_gates:
            raise ValueError(f"wire_id {wire_id} out of range [0, {self.reference.num_gates})")
        all_values = self.simulate_faulty(input_values)
        return all_values[wire_id]

    def verify_diagnosis(self, guessed_gate_id: int) -> bool:
        """Check if the agent's diagnosis is correct.

        Args:
            guessed_gate_id: The gate index the agent thinks is faulty.

        Returns:
            True if the guess matches the faulty gate.
        """

        return guessed_gate_id == self.faulty_gate_id


# ---------------------------------------------------------------------------
# Circuit generation
# ---------------------------------------------------------------------------


def generate_circuit(
    num_inputs: int = 3,
    num_gates: int = 6,
    *,
    seed: int = 42,
) -> Circuit:
    """Generate a random logic circuit with AND/OR/NOT gates.

    Args:
        num_inputs: Number of INPUT gates (must be >= 2).
        num_gates: Total number of gates including inputs (must be > num_inputs).
        seed: Random seed for reproducibility.

    Returns:
        A :class:`Circuit` with topologically ordered gates.

    Raises:
        ValueError: If parameters are invalid.
    """

    if num_inputs < 2:
        raise ValueError("num_inputs must be >= 2")
    if num_gates <= num_inputs:
        raise ValueError("num_gates must be > num_inputs")
    rng = random.Random(seed)
    gates: list[Gate] = []
    # First num_inputs gates are INPUT gates.
    for i in range(num_inputs):
        gates.append(Gate(gate_id=i, gate_type=GateType.INPUT, inputs=()))
    # Remaining gates are AND/OR/NOT, each taking inputs from earlier gates.
    for i in range(num_inputs, num_gates):
        gate_type = rng.choice([GateType.AND, GateType.OR, GateType.NOT])
        if gate_type is GateType.NOT:
            inp = rng.randint(0, i - 1)
            gates.append(Gate(gate_id=i, gate_type=gate_type, inputs=(inp,)))
        else:
            inp_a = rng.randint(0, i - 1)
            inp_b = rng.randint(0, i - 1)
            gates.append(Gate(gate_id=i, gate_type=gate_type, inputs=(inp_a, inp_b)))
    return Circuit(gates=gates, num_inputs=num_inputs, num_outputs=1)


def inject_fault(circuit: Circuit, *, seed: int = 0) -> FaultyCircuit:
    """Inject a random stuck-at fault into a non-INPUT gate.

    Args:
        circuit: The reference circuit.
        seed: Random seed for fault selection.

    Returns:
        A :class:`FaultyCircuit` with one faulty gate.

    Raises:
        ValueError: If the circuit has no non-INPUT gates.
    """

    rng = random.Random(seed)
    # Only non-INPUT gates can be faulty.
    candidate_ids = [g.gate_id for g in circuit.gates if g.gate_type is not GateType.INPUT]
    if not candidate_ids:
        raise ValueError("circuit has no non-INPUT gates to fault")
    faulty_gate_id = rng.choice(candidate_ids)
    fault_type = rng.choice(["stuck_at_0", "stuck_at_1"])
    return FaultyCircuit(reference=circuit, faulty_gate_id=faulty_gate_id, fault_type=fault_type)


# ---------------------------------------------------------------------------
# Agent interaction
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of probing a wire with given inputs.

    Attributes:
        wire_id: The wire that was probed.
        input_vector: The input values used.
        value: The observed output value (from faulty circuit).
    """

    wire_id: int
    input_vector: list[bool]
    value: bool


@dataclass
class DiagnosisSession:
    """A diagnosis session tracking the agent's probes and submissions.

    Attributes:
        faulty_circuit: The faulty circuit being diagnosed.
        max_probes: Maximum number of probes allowed.
        max_submissions: Maximum number of diagnosis submissions.
        probes: List of probes made so far.
        submissions: List of submitted gate IDs.
        solved: Whether the correct gate was identified.
    """

    faulty_circuit: FaultyCircuit
    max_probes: int = 20
    max_submissions: int = 3
    probes: list[ProbeResult] = field(default_factory=list)
    submissions: list[int] = field(default_factory=list)
    solved: bool = False

    @property
    def num_probes(self) -> int:
        return len(self.probes)

    @property
    def num_submissions(self) -> int:
        return len(self.submissions)

    @property
    def probes_remaining(self) -> int:
        return max(0, self.max_probes - self.num_probes)

    @property
    def submissions_remaining(self) -> int:
        return max(0, self.max_submissions - self.num_submissions)

    def probe(self, input_vector: list[bool], wire_id: int) -> ProbeResult:
        """Probe a wire with the given input vector.

        Args:
            input_vector: Boolean values for each INPUT gate.
            wire_id: The wire index to inspect.

        Returns:
            A :class:`ProbeResult` with the observed value.

        Raises:
            ValueError: If no probes remaining or wire_id is invalid.
        """

        if self.probes_remaining <= 0:
            raise ValueError(f"no probes remaining (max={self.max_probes})")
        circuit = self.faulty_circuit.reference
        if wire_id < 0 or wire_id >= circuit.num_gates:
            raise ValueError(f"wire_id {wire_id} out of range [0, {circuit.num_gates})")
        value = self.faulty_circuit.get_faulty_wire_value(input_vector, wire_id)
        result = ProbeResult(wire_id=wire_id, input_vector=list(input_vector), value=value)
        self.probes.append(result)
        return result

    def submit(self, gate_id: int) -> bool:
        """Submit a diagnosis (which gate is faulty).

        Args:
            gate_id: The gate index the agent thinks is faulty.

        Returns:
            True if the diagnosis is correct.

        Raises:
            ValueError: If no submissions remaining.
        """

        if self.submissions_remaining <= 0:
            raise ValueError(f"no submissions remaining (max={self.max_submissions})")
        self.submissions.append(gate_id)
        correct = self.faulty_circuit.verify_diagnosis(gate_id)
        if correct:
            self.solved = True
        return correct


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_prompt(circuit: Circuit, *, max_turns: int = 10) -> str:
    """Format the user-facing prompt describing the circuit.

    The prompt tells the agent the circuit structure (gate types and
    connectivity) but hides which gate is faulty. The agent must use
    tools to probe and diagnose.

    Args:
        circuit: The reference circuit (its structure is shown to the agent).
        max_turns: Maximum number of turns the agent has (default 10).
            This should match the MAX_TURNS in run_agent.py.

    Returns:
        A prompt string.
    """

    lines: list[str] = []
    lines.append("You are diagnosing a faulty logic circuit.")
    lines.append(f"The circuit has {circuit.num_inputs} inputs and {circuit.num_gates} gates total.")
    lines.append(f"Gate {circuit.num_gates - 1} is the output gate.")
    lines.append("")
    lines.append("Circuit structure (gate_id: type(inputs)):")
    for gate in circuit.gates:
        if gate.gate_type is GateType.INPUT:
            lines.append(f"  Gate {gate.gate_id}: INPUT")
        elif gate.gate_type is GateType.NOT:
            lines.append(f"  Gate {gate.gate_id}: NOT({gate.inputs[0]})")
        else:
            lines.append(f"  Gate {gate.gate_id}: {gate.gate_type.value}({gate.inputs[0]}, {gate.inputs[1]})")
    lines.append("")
    lines.append("One non-INPUT gate is faulty (stuck-at-0 or stuck-at-1).")
    lines.append("Use the 'probe' tool to set inputs and inspect wire values.")
    lines.append("Use the 'submit' tool to identify the faulty gate.")
    lines.append(f"You have at most {max_turns} turns. Call exactly one tool per turn.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_diagnosis(session: DiagnosisSession) -> float:
    """Score a diagnosis session.

    Rewards correct diagnosis with a bonus for fewer probes.

    Args:
        session: A completed or in-progress :class:`DiagnosisSession`.

    Returns:
        A score in [0.0, 1.0]. 0.0 if not solved; otherwise
        (1.0 - num_probes / max_probes * 0.5), so fewer probes = higher score.
    """

    if not session.solved:
        return 0.0
    efficiency = 1.0 - (session.num_probes / session.max_probes) * 0.5
    return max(0.5, efficiency)


def brute_force_baseline(faulty: FaultyCircuit, *, max_probes: int = 20) -> tuple[int, int]:
    """Brute-force diagnosis: try all input vectors on each non-INPUT gate.

    Compares faulty circuit output against reference for each gate. The
    first gate that shows a discrepancy is the likely fault.

    Args:
        faulty: The faulty circuit to diagnose.
        max_probes: Maximum probes to simulate.

    Returns:
        A tuple of (guessed_gate_id, probes_used).
    """

    circuit = faulty.reference
    num_inputs = circuit.num_inputs
    # Generate all possible input vectors.
    from itertools import product as iter_product

    all_vectors = list(iter_product([False, True], repeat=num_inputs))
    candidate_ids = [g.gate_id for g in circuit.gates if g.gate_type is not GateType.INPUT]
    probes_used = 0
    for gate_id in candidate_ids:
        is_faulty_candidate = False
        for vec in all_vectors:
            if probes_used >= max_probes:
                break
            ref_val = circuit.get_wire_value(list(vec), gate_id)
            faulty_val = faulty.get_faulty_wire_value(list(vec), gate_id)
            probes_used += 1
            if ref_val != faulty_val:
                is_faulty_candidate = True
                break
        if is_faulty_candidate:
            return gate_id, probes_used
    return candidate_ids[0], probes_used
