"""Dataset loader for Tic-Tac-Toe SFT rows.

Each JSONL line has {"prompt": str, "response": str} and is passed through
directly — the gen_sft_data.py script already writes the exact format that
areno.api.trainers.sft._record_to_train_sequence expects.
"""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Pass through JSONL rows that already have prompt/response fields."""

    records = []
    for row in default_loader(dataset_path):
        record = dict(row)
        if "prompt" not in record or "response" not in record:
            continue
        if record["prompt"] is None or record["response"] is None:
            continue
        records.append({"prompt": str(record["prompt"]), "response": str(record["response"])})
    return records
