"""Dataset loader for the Gymnasium rollout examples.

Each record describes one episode: an integer ``seed`` fixes the episode
trajectory and ``prompt`` carries the task statement.
"""

from __future__ import annotations

DEFAULT_PROMPT = "Play the episode and maximize the cumulative reward."


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize episode records while remaining tokenizer and processor independent."""

    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        if record.get("seed") is None:
            raise ValueError("gymnasium episode records require an integer seed")
        record["seed"] = int(record["seed"])
        record["id"] = str(record.get("id", f"gymnasium-{index:05d}"))
        record["prompt"] = str(record.get("prompt") or DEFAULT_PROMPT)
        records.append(record)
    return records
