"""Dataset loader for multi-turn chat SFT rows."""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize multi-turn chat rows to SFT ``messages`` format.

    Expects raw rows with a ``messages`` field (OpenAI/HF chat format) or a
    ``conversations`` field (ShareGPT format). Each item should have ``role``
    and ``content`` keys.
    """

    records = []
    for row in default_loader(dataset_path):
        record = dict(row)
        messages = record.get("messages") or record.get("conversations")
        if not messages:
            continue
        normalized = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            normalized.append({"role": role, "content": content})
        if not normalized:
            continue
        records.append({"messages": normalized})
    return records
