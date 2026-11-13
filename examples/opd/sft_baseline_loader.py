"""SFT baseline loader for GSM8K.

``--algo sft`` expects rows with ``prompt`` and ``response`` strings, while
the OPD/RL loader in ``examples/math/dataset_loader.py`` produces ``prompt``
and ``solutions``. This loader keeps the prompt text byte-identical to the OPD
runs so that the two algorithms are trained on the same questions and can be
compared with the same held-out generation benchmark. The duplication is
deliberate: each example directory must stay importable on its own, so
``tests/test_example_loaders_cpu.py`` pins the two prompts to the same bytes.
"""

from __future__ import annotations

PROMPT_PREFIX = (
    "Solve the following grade-school math problem. Show your reasoning "
    "and put the final answer in \\boxed{}.\n\n"
)


def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
    """Normalize GSM8K rows into SFT ``prompt`` / ``response`` pairs."""

    dataset = default_loader(dataset_path)
    if len(dataset) == 0:
        return dataset
    first = dataset[0]
    if "prompt" in first and "response" in first:
        return dataset
    if "question" in first and "answer" in first:
        return dataset.map(_format_gsm8k_record)
    raise KeyError(
        "SFT GSM8K rows must contain `prompt`/`response` or "
        "GSM8K-style `question`/`answer` fields"
    )


def _format_gsm8k_record(record: dict) -> dict:
    answer = str(record["answer"]).strip()
    final = answer.rsplit("####", 1)[-1].strip() if "####" in answer else answer
    return {
        "prompt": PROMPT_PREFIX + f"Problem: {record['question']}\nSolution:",
        "response": " " + answer,
        "solutions": [final],
        "answer": [final],
    }
