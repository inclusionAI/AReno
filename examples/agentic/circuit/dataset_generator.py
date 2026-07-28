"""Generate circuit diagnosis datasets for the agentic example (issue #193)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit  # noqa: E402

DEFAULT_COUNT = 128
DEFAULT_SEED = 2026
DEFAULT_NUM_INPUTS = 3
DEFAULT_NUM_GATES = 6


def generate_records(
    count: int = DEFAULT_COUNT,
    *,
    seed: int = DEFAULT_SEED,
    num_inputs: int = DEFAULT_NUM_INPUTS,
    num_gates: int = DEFAULT_NUM_GATES,
) -> list[dict]:
    """Generate reproducible circuit diagnosis records.

    Each record contains a circuit description, the faulty gate ID, and
    the fault type.

    Args:
        count: Number of unique circuits to generate.
        seed: Base random seed (each circuit uses seed + i).
        num_inputs: Number of input gates per circuit.
        num_gates: Total gates per circuit.

    Returns:
        A list of record dicts.
    """

    records: list[dict] = []
    seen: set[tuple[tuple, int, str]] = set()
    for i in range(count):
        circ = circuit.generate_circuit(num_inputs, num_gates, seed=seed + i)
        faulty = circuit.inject_fault(circ, seed=seed + i)
        # Create a hashable key to detect duplicates.
        key = (
            tuple((g.gate_type.value, g.inputs) for g in circ.gates),
            faulty.faulty_gate_id,
            faulty.fault_type,
        )
        if key in seen:
            continue
        seen.add(key)
        records.append(_circuit_to_record(circ, faulty, f"generated-{len(records):05d}"))
    return records


def _circuit_to_record(circ: circuit.Circuit, faulty: circuit.FaultyCircuit, record_id: str) -> dict:
    """Convert a Circuit + FaultyCircuit to a serialisable record."""

    return {
        "id": record_id,
        "num_inputs": circ.num_inputs,
        "num_gates": circ.num_gates,
        "gates": [
            {
                "gate_id": g.gate_id,
                "gate_type": g.gate_type.value,
                "inputs": list(g.inputs),
            }
            for g in circ.gates
        ],
        "faulty_gate_id": faulty.faulty_gate_id,
        "fault_type": faulty.fault_type,
        "prompt": circuit.format_prompt(circ),
    }


def write_jsonl(records: list[dict], output: TextIO) -> None:
    """Write generated records as JSONL."""

    for record in records:
        output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL circuit diagnosis records for the Areno agentic example.",
    )
    parser.add_argument("--output", "-o", default="-", help="Output JSONL path, or '-' for stdout.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of circuits to generate.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random base seed.")
    parser.add_argument("--num-inputs", type=int, default=DEFAULT_NUM_INPUTS, help="Number of input gates.")
    parser.add_argument("--num-gates", type=int, default=DEFAULT_NUM_GATES, help="Total gates per circuit.")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.num_inputs < 2:
        raise ValueError("--num-inputs must be >= 2")
    if args.num_gates <= args.num_inputs:
        raise ValueError("--num-gates must be > --num-inputs")

    records = generate_records(args.count, seed=args.seed, num_inputs=args.num_inputs, num_gates=args.num_gates)
    if args.output == "-":
        write_jsonl(records, sys.stdout)
    else:
        output_path = Path(args.output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            write_jsonl(records, handle)


if __name__ == "__main__":
    main()
