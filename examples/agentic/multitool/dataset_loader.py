"""Dataset loader for the multi-tool agentic example.

Validates each record's tool names and expected fields before the expensive
model/worker initialization, so misconfigured datasets fail fast with a
clear error instead of crashing mid-training.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from game import make_prompt  # noqa: E402

# Known tool names that can appear in required_tools / tool_calls.
KNOWN_TOOLS = {"lookup_contact", "read_note", "calculate", "unit_convert", "lookup_parcel"}

# Required expected_* fields per task id prefix. Each task type must have
# its expected fields present so the scoring logic can evaluate arguments.
REQUIRED_EXPECTED_FIELDS: dict[str, set[str]] = {
    "contact": {"expected_contact", "expected_note_key"},
    "budget": {"expected_note_keys"},
    "parcel": {"expected_parcel", "expected_contact_city"},
    "calc": {"expected_expression", "expected_note_key"},
    "convert": {"expected_value", "expected_from_unit", "expected_to_unit", "expected_parcel"},
}


def validate_record(record: dict) -> None:
    """Validate a single task record before model initialization.

    Checks:
      - "id" and "description" fields are present.
      - "required_tools" is a list of known tool names with at least 2 entries.
      - The task id prefix has the expected_* fields needed for scoring.

    Args:
        record: A task record dict from the JSONL dataset.

    Raises:
        ValueError: If a required field is missing, a tool name is unknown,
                    or expected fields for the task type are absent.
    """

    if "id" not in record:
        raise ValueError(f"record missing required field 'id': {record}")
    if "description" not in record:
        raise ValueError(f"record {record.get('id')!r} missing 'description'")
    required = record.get("required_tools")
    if not isinstance(required, list) or len(required) < 2:
        raise ValueError(
            f"record {record.get('id')!r} must have 'required_tools' with at least 2 entries, got: {required}"
        )
    unknown = [t for t in required if t not in KNOWN_TOOLS]
    if unknown:
        raise ValueError(
            f"record {record.get('id')!r} has unknown tool(s): {unknown}; known: {sorted(KNOWN_TOOLS)}"
        )
    task_prefix = str(record["id"]).split("-")[0]
    needed = REQUIRED_EXPECTED_FIELDS.get(task_prefix)
    if needed is not None:
        missing = [f for f in needed if f not in record]
        if missing:
            raise ValueError(
                f"record {record.get('id')!r} (task type {task_prefix!r}) missing expected fields: {missing}"
            )


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize JSONL rows into prompt-bearing records.

    Validates each record before returning, so schema errors surface before
    the expensive model/worker initialization begins.

    Args:
        dataset_path: Path to the JSONL dataset file.
        default_loader: AReno's default JSONL loader callable.

    Returns:
        A list of validated, prompt-bearing task records.

    Raises:
        ValueError: If any record fails validation.
    """

    rows = default_loader(dataset_path)
    records = []
    for row in rows:
        record = dict(row)
        validate_record(record)
        record["prompt"] = make_prompt(record)
        records.append(record)
    return records