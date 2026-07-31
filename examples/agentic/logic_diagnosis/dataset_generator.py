"""Generate reproducible logic-circuit diagnosis tasks."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from game import (
    BRUTE_FORCE_GATE_LIMIT,
    MAX_GATES,
    MAX_INPUTS,
    MAX_PROBES,
    MIN_GATES,
    MIN_INPUTS,
    brute_force_verify,
    evaluate,
    generate_circuit,
    make_prompt,
)

DATASET_MIN_GATES = MIN_GATES
DATASET_MAX_GATES = min(MAX_GATES, BRUTE_FORCE_GATE_LIMIT)
MAX_ATTEMPTS_PER_RECORD = 1_000
FAULT_CLASSES = (
    ("and", 0),
    ("and", 1),
    ("or", 0),
    ("or", 1),
    ("not", 0),
    ("not", 1),
)
T = TypeVar("T")


def generate_records(
    count: int = 256,
    *,
    seed: int = 2026,
) -> list[dict]:
    """Return balanced, deterministic, fully verified diagnosis records.

    Every record has 3--8 *live* gates, so the existing brute-force verifier
    can check every record. Records are balanced across input count, live gate
    count, and fault type/value; exact duplicate, ambiguous, and output-inert
    faults are rejected.
    """
    if count < 0:
        raise ValueError("count must be non-negative")

    rng = random.Random(seed)
    records: list[dict] = []
    seen_records: set[str] = set()
    target_inputs = _balanced_targets(range(MIN_INPUTS, MAX_INPUTS + 1), count, rng)
    target_gates = _balanced_targets(range(DATASET_MIN_GATES, DATASET_MAX_GATES + 1), count, rng)
    target_faults = _balanced_targets(FAULT_CLASSES, count, rng)
    max_attempts = max(count * MAX_ATTEMPTS_PER_RECORD, MAX_ATTEMPTS_PER_RECORD)

    attempts = 0
    for record_index, (n_inputs, n_gates, fault_class) in enumerate(
        zip(target_inputs, target_gates, target_faults), start=1
    ):
        fault_type, stuck_value = fault_class
        while attempts < max_attempts:
            attempts += 1
            requested_gates = rng.randint(n_gates, MAX_GATES)
            nodes = generate_circuit(
                n_inputs,
                requested_gates,
                seed=rng.randint(0, 2**31 - 1),
            )
            gate_nodes = [node for node in nodes if node["type"] in ("and", "or", "not")]
            if len(gate_nodes) != n_gates:
                continue

            matching_gates = [node for node in gate_nodes if node["type"] == fault_type]
            if not matching_gates:
                continue
            fault = {"node": rng.choice(matching_gates)["id"], "stuck_value": stuck_value}

            if not brute_force_verify(nodes, fault) or not _fault_changes_output(nodes, fault):
                continue

            record_key = json.dumps({"nodes": nodes, "fault": fault}, sort_keys=True, separators=(",", ":"))
            if record_key in seen_records:
                continue
            seen_records.add(record_key)

            record = {
                "id": f"logic-diag-{record_index:05d}",
                "nodes": nodes,
                "fault": fault,
                "n_inputs": n_inputs,
                "n_gates": n_gates,
                "max_probes": MAX_PROBES,
            }
            record["prompt"] = make_prompt(record)
            records.append(record)
            break
        else:
            raise RuntimeError(
                f"only generated {len(records)}/{count} records after {attempts} attempts; "
                "try a different seed or reduce the requested count"
            )

    return records


def _balanced_targets(values: Sequence[T], count: int, rng: random.Random) -> list[T]:
    """Return a shuffled schedule in which bucket counts differ by at most one."""
    values = list(values)
    targets = [values[index % len(values)] for index in range(count)]
    rng.shuffle(targets)
    return targets


def _fault_changes_output(nodes: list[dict], fault: dict[str, int]) -> bool:
    """Return whether some primary-input vector exposes ``fault`` at the output."""
    n_inputs = sum(node["type"] == "input" for node in nodes)
    output_id = next(node["id"] for node in nodes if node["type"] == "output")
    for value in range(1 << n_inputs):
        inputs = [bool(value >> index & 1) for index in range(n_inputs)]
        if evaluate(nodes, inputs)[output_id] != evaluate(nodes, inputs, fault)[output_id]:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path")
    parser.add_argument("--count", type=int, default=256, help="Number of records")
    parser.add_argument("--seed", type=int, default=2026, help="Reproducibility seed")
    args = parser.parse_args()

    records = generate_records(args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
