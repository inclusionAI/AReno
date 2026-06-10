"""Math-example dataset normalisation helpers.

Pass this file through `areno train --dataset-loader-fn` when the raw dataset
is not already in the trainer schema. It loads pre-normalised JSONL with
`prompt`/`solutions`, GSM8K-style `question`/`answer`, or NuminaMath-TIR-style
`problem`/`solution` rows and rewrites them into the trainer's expected schema.
"""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object):
    """Load math datasets and normalize them to the trainer prompt schema.

    The trainer core expects each row to contain `prompt` and optionally
    `solutions`. This loader keeps that contract local to math examples and
    supports already-normalized math JSONL, GSM8K-style rows with
    `question`/`answer`, and NuminaMath-TIR-style rows with
    `problem`/`solution`.
    """

    dataset = default_loader(dataset_path)
    if len(dataset) == 0:
        return dataset
    # Sniff schema from the first row: rows that already have a `prompt`
    # column are assumed to be in the canonical format and pass through
    # untouched.
    first = dataset[0]
    if "prompt" in first:
        return dataset
    if "question" in first and "answer" in first:
        return dataset.map(_format_gsm8k_record)
    if "problem" in first and "solution" in first:
        return dataset.map(_format_math_record)
    raise KeyError(
        "math dataset rows must contain `prompt`, GSM8K-style `question`/`answer`, "
        "or NuminaMath-style `problem`/`solution` fields"
    )


def _format_gsm8k_record(record: dict) -> dict:
    # GSM8K answers usually include a rationale followed by `#### final`.
    answer = str(record["answer"])
    final = answer.rsplit("####", 1)[-1].strip() if "####" in answer else answer.strip()
    return {
        "prompt": (
            "Solve the following grade-school math problem. Show your reasoning "
            "and put the final answer in \\boxed{}.\n\n"
            f"Problem: {record['question']}\nSolution:"
        ),
        "solutions": [final],
        "solution": final,
    }


def _format_math_record(record: dict) -> dict:
    # Wraps the raw NuminaMath problem in a chat-of-thought style prompt and
    # exposes the solution as a single-element `solutions` list (the schema
    # the math reward function expects).
    solution = record["solution"]
    return {
        "prompt": (
            "Solve the following math problem. Show your reasoning and put the "
            "final answer in \\boxed{}.\n\n"
            f"Problem: {record['problem']}\nSolution:"
        ),
        "solutions": [solution],
        "solution": solution,
    }
