"""Dataset loader for logic-circuit diagnosis agentic training."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import MAX_PROBES, make_prompt  # noqa: E402


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize records while remaining tokenizer and processor independent."""

    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)

        # Coerce and validate fields
        nodes = record.get("nodes", [])
        record["nodes"] = nodes
        record["n_inputs"] = int(record.get("n_inputs", sum(1 for n in nodes if n.get("type") == "input")))
        record["n_gates"] = int(
            record.get("n_gates", sum(1 for n in nodes if n.get("type") in ("and", "or", "not")))
        )
        record["max_probes"] = min(max(int(record.get("max_probes", MAX_PROBES)), 1), MAX_PROBES)
        record["id"] = str(record.get("id", f"logic-diag-{index:05d}"))

        # Ensure fault field exists
        fault = record.get("fault")
        if not isinstance(fault, dict):
            record["fault"] = {"node": -1, "stuck_value": 0}

        # Generate prompt (does NOT include fault info)
        record["prompt"] = make_prompt(record)
        records.append(record)

    return records