"""Dataset loader for the agentic coding example."""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load JSONL coding tasks and attach the prompt used by AReno."""

    records = []
    for row in default_loader(dataset_path):
        record = dict(row)
        record["files"] = {str(path): str(content) for path, content in dict(record["files"]).items()}
        record["test_commands"] = [str(command) for command in record.get("test_commands", [])]
        record["prompt"] = _make_prompt(record)
        records.append(record)
    return records


def _make_prompt(record: dict) -> str:
    commands = ", ".join(record.get("test_commands") or [])
    instance_id = record.get("instance_id", record.get("id", "unknown"))
    problem = record.get("problem_statement", record.get("instruction", ""))
    return (
        f"Fix SWE-bench-style task {instance_id}.\n"
        f"{problem}\n"
        f"Allowed tests: {commands or 'none'}\n"
        "Use coding tools to inspect files, patch the repository, run tests, and submit the result."
    )
