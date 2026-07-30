"""Generate reproducible logic-circuit diagnosis tasks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from game import (
    MAX_GATES,
    MAX_INPUTS,
    MAX_PROBES,
    MIN_GATES,
    MIN_INPUTS,
    brute_force_verify,
    generate_circuit,
    inject_fault,
    make_prompt,
)


def generate_records(
    count: int = 256,
    *,
    seed: int = 2026,
) -> list[dict]:
    """Return deterministic circuit diagnosis records with unique faults.

    Each record contains the full circuit topology, fault information, and a
    pre-rendered prompt. Small circuits (≤8 gates) are brute-force verified to
    guarantee a unique distinguishing I/O signature.
    """
    rng = random.Random(seed)
    records: list[dict] = []
    max_attempts = count * 200

    attempts = 0
    while len(records) < count and attempts < max_attempts:
        attempts += 1

        n_inputs = rng.randint(MIN_INPUTS, MAX_INPUTS)
        n_gates = rng.randint(MIN_GATES, MAX_GATES)
        circuit_seed = rng.randint(0, 2**31 - 1)
        nodes = generate_circuit(n_inputs, n_gates, seed=circuit_seed)

        # Ensure there is at least one gate to fault
        gate_nodes = [n for n in nodes if n["type"] in ("and", "or", "not")]
        if not gate_nodes:
            continue

        fault_seed = rng.randint(0, 2**31 - 1)
        fault = inject_fault(nodes, seed=fault_seed)

        if not brute_force_verify(nodes, fault):
            continue  # ambiguous — regenerate

        record_id = f"logic-diag-{len(records) + 1:05d}"
        record = {
            "id": record_id,
            "nodes": nodes,
            "fault": fault,
            "n_inputs": n_inputs,
            "n_gates": n_gates,
            "max_probes": MAX_PROBES,
        }
        record["prompt"] = make_prompt(record)
        records.append(record)

    if len(records) < count:
        raise RuntimeError(
            f"only generated {len(records)}/{count} records after {max_attempts} attempts; "
            "try a different seed or adjust circuit size ranges"
        )

    return records


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