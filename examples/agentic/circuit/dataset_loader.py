"""Dataset loader for the circuit-diagnosis agentic example (issue #193)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import circuit  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load JSONL circuit records and convert them to Areno prompt records.

    Args:
        dataset_path: Path to a JSONL file, or a directory containing one.

    Returns:
        A list of records with ``prompt`` and metadata fields.
    """

    del default_loader
    return _load_records(dataset_path)


def _load_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if path.is_dir():
        path = path / "circuits.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"circuit dataset not found: {path}")
    if not path.is_file():
        raise ValueError(f"circuit dataset path is not a file: {path}")
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    raw = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path} at line {line_number}: {exc.msg}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"record in {path} at line {line_number} must be a JSON object")
                try:
                    records.append(_format_record(raw))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid circuit record in {path} at line {line_number}: {exc}") from exc
    if not records:
        raise ValueError(f"circuit dataset is empty: {path}")
    return records


def _format_record(raw: dict) -> dict:
    """Validate a raw circuit record and format it for the trainer."""

    required = {"id", "num_inputs", "num_gates", "gates", "faulty_gate_id", "fault_type"}
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"missing required fields: {', '.join(missing)}")
    if not isinstance(raw["id"], str) or not raw["id"].strip():
        raise ValueError("id must be a non-empty string")
    if isinstance(raw["num_inputs"], bool) or not isinstance(raw["num_inputs"], int):
        raise ValueError("num_inputs must be an integer")
    if isinstance(raw["num_gates"], bool) or not isinstance(raw["num_gates"], int):
        raise ValueError("num_gates must be an integer")
    if not isinstance(raw["gates"], list):
        raise ValueError("gates must be a list")
    if len(raw["gates"]) != raw["num_gates"]:
        raise ValueError(f"num_gates={raw['num_gates']} does not match {len(raw['gates'])} gate records")

    gates: list[circuit.Gate] = []
    for index, gate_record in enumerate(raw["gates"]):
        if not isinstance(gate_record, dict):
            raise ValueError(f"gate {index} must be an object")
        try:
            gate_id = gate_record["gate_id"]
            gate_type = circuit.GateType(gate_record["gate_type"])
            inputs = gate_record["inputs"]
        except KeyError as exc:
            raise ValueError(f"gate {index} is missing field {exc.args[0]!r}") from exc
        if not isinstance(inputs, list):
            raise ValueError(f"gate {index} inputs must be a list")
        gates.append(circuit.Gate(gate_id=gate_id, gate_type=gate_type, inputs=tuple(inputs)))

    validated = circuit.Circuit(gates=gates, num_inputs=raw["num_inputs"], num_outputs=1)
    faulty = circuit.FaultyCircuit(
        reference=validated,
        faulty_gate_id=raw["faulty_gate_id"],
        fault_type=raw["fault_type"],
    )
    if not circuit.fault_is_output_observable(faulty):
        raise ValueError("fault is not observable at any circuit output")

    result: dict[str, Any] = {
        "id": raw["id"],
        "prompt": circuit.format_prompt(validated),
        "num_inputs": validated.num_inputs,
        "num_gates": validated.num_gates,
        "gates": raw["gates"],
        "faulty_gate_id": faulty.faulty_gate_id,
        "fault_type": faulty.fault_type,
    }
    if "seed" in raw:
        result["seed"] = raw["seed"]
    return result
