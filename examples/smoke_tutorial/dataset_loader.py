"""Minimal dataset loader for the smoke tutorial.

This loader demonstrates the simplest possible dataset contract for AReno.
It loads JSONL files with `prompt` and optionally `reference` fields,
or wraps plain text lines into prompt format.
"""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load a minimal dataset for smoke testing.

    Supports:
    - JSONL files with `prompt` field (pass-through)
    - JSONL files with `question`/`answer` fields (normalize to prompt)
    - Plain text files (one prompt per line)
    """

    dataset = default_loader(dataset_path)
    if len(dataset) == 0:
        return dataset

    first = dataset[0]

    # Already in trainer format
    if "prompt" in first:
        return dataset

    # Question/answer format
    if "question" in first:
        return [
            {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "reference": str(row.get("answer", "")),
            }
            for row in dataset
        ]

    # Plain text - wrap into prompt format
    return [
        {"prompt": str(row.get("text", row.get("content", "")))}
        for row in dataset
    ]
