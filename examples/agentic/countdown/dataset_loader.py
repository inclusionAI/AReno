"""Dataset loader for Countdown arithmetic agentic training.

This module converts raw Countdown puzzle records (numbers + target) stored in
JSONL format into the prompt records that AReno's agentic trainer consumes.

Role in the training pipeline:
    JSONL file  --default_loader-->  raw dict rows
                                       |
                                   load_training_dataset()
                                       |
                                       v
                              list of records with a formatted
                              `prompt` field + `numbers` / `target`
                                       |
                                  AReno rollout (run_agent.py)
"""

from __future__ import annotations


def load_training_dataset(dataset_path: str, *, default_loader, **_: object) -> list[dict]:
    """Load Countdown dataset and format each row into an agent prompt.

    Args:
        dataset_path: Path to the JSONL file. Each line is a JSON object with
            ``numbers`` (list[int]) and ``target`` (int).
        default_loader: AReno's built-in JSONL reader; called once per line
            and yields a dict. We accept it as an injected dependency so this
            loader stays compatible with AReno's CLI plumbing.
        **_: Extra kwargs AReno may pass in future versions; ignored for
            forward compatibility.

    Returns:
        A list of records. Each record carries the original ``numbers`` /
        ``target`` / ``id`` plus a ``prompt`` field ready to be fed to the
        model as the initial user message.
    """
    records = []
    for index, row in enumerate(default_loader(dataset_path), start=1):
        record = dict(row)
        numbers = record.get("numbers", [])
        target = record.get("target", 0)
        # Fall back to a stable synthetic id if the dataset omits one.
        item_id = str(record.get("id", f"countdown-{index:05d}"))

        # Format the prompt for the agent: describe the puzzle and tell the
        # model which tools are available. The tool definitions themselves
        # are injected by run_agent.py via the OpenAI tools API; here we only
        # describe the rules of the game.
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