"""Dataset loader for Countdown arithmetic agentic training."""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load Countdown dataset and format prompts."""

    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        numbers = record.get("numbers", [])
        target = record.get("target", 0)
        item_id = str(record.get("id", f"countdown-{index:05d}"))

        # Format the prompt for the agent
        prompt = (
            f"Solve this Countdown puzzle:\n"
            f"Numbers: {', '.join(map(str, numbers))}\n"
            f"Target: {target}\n\n"
            f"Use the available tools (add, subtract, multiply, divide) to reach the target. "
            f"Each number can only be used once. When you have the final answer, call the 'finish' tool."
        )

        record["id"] = item_id
        record["prompt"] = prompt
        record["numbers"] = numbers
        record["target"] = target
        records.append(record)

    return records
