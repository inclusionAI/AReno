"""Dataset loader for Alpaca-style SFT rows."""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Normalize Alpaca instruction/input/output rows to SFT prompt/response."""

    records = []
    for row in default_loader(dataset_path):
        record = dict(row)
        instruction = str(record["instruction"]).strip()
        input_text = str(record.get("input") or "").strip()
        prompt = f"Instruction: {instruction}\n"
        if input_text:
            prompt += f"Input: {input_text}\n"
        prompt += "Response:"
        records.append({"prompt": prompt, "response": str(record["output"])})
    return records
