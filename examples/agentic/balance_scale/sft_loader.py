"""SFT dataset loader for the balance-scale example.

Flattens multi-turn weigh/submit_answer conversations into single
prompt/response pairs suitable for AReno's SFT trainer.

Each SFT record from sft_data_generator.py contains a list of messages
(system → user → assistant → tool → assistant → ... → assistant).
We flatten this into:
  prompt: system + user + all tool results so far
  response: the next assistant tool call (weigh or submit_answer)

This produces one training row per assistant turn, allowing the model
to learn the mapping from (puzzle state + weighing history) → next action.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)


def load_training_dataset(dataset_path: str, *, default_loader=None, **_: object) -> list[dict]:
    """Load SFT JSONL and flatten into prompt/response pairs."""

    del default_loader
    records = _load_sft_records(dataset_path)
    return _flatten_records(records)


def _load_sft_records(dataset_path: str) -> list[dict]:
    path = Path(dataset_path).expanduser()
    if not path.exists():
        logger.warning("SFT dataset not found: %s", path)
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def _flatten_records(records: list[dict]) -> list[dict]:
    """Convert multi-turn conversations into individual prompt/response rows."""

    rows: list[dict] = []
    for record in records:
        messages = record.get("messages", [])
        if not messages:
            continue

        # Build prompt incrementally: system + user + (assistant + tool)*
        prompt_parts: list[str] = []
        response_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"[System]\n{content}")
            elif role == "user":
                prompt_parts.append(f"[User]\n{content}")
            elif role == "tool":
                prompt_parts.append(f"[Scale Result]\n{content}")
            elif role == "assistant":
                # This is a response — create a training row
                prompt = "\n\n".join(prompt_parts)
                # Format response as the tool call JSON
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    tc = tool_calls[0]
                    args = tc["function"]["arguments"]
                    response = args  # The JSON arguments string
                else:
                    response = content or ""

                rows.append({
                    "prompt": prompt + "\n\n[Assistant]",
                    "response": response,
                })
                # Add this assistant response to prompt for next turn
                prompt_parts.append(f"[Assistant]\n{response}")

    return rows