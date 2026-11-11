"""Prepare the fixed Ling payment workload with the original MLX-LM packing."""

from __future__ import annotations

import os

from areno.api.data_utils import apply_chat_template, encode_prompt_value
from areno.api.tokenizer import configure_chat_template_enable_thinking, load_tokenizer


def load_training_dataset(
    dataset_path: str, *, default_loader, model_path: str | None = None, **_: object
) -> list[dict]:
    """Encode conversations in file order; SFTTrainer owns seeded shuffling."""

    model_dir = model_path or os.environ.get("LING_MODEL_DIR")
    if not model_dir:
        raise ValueError("Pass model_path from areno train or set LING_MODEL_DIR when calling the loader directly")
    tokenizer = load_tokenizer(model_dir)
    configure_chat_template_enable_thinking(tokenizer, True)
    rows = []
    for row in default_loader(dataset_path):
        messages = row["messages"]
        tokens = apply_chat_template(tokenizer, messages)
        prompt = encode_prompt_value(tokenizer, messages[:-1])
        if tokens[: len(prompt)] != prompt:
            raise ValueError("Chat template prompt does not match the training sequence")
        if not 0 < len(prompt) < len(tokens) <= 1024:
            raise ValueError("Ling payment conversations must contain a response and fit within 1024 tokens")
        # Match the reference iterate_batches: round width to 32 plus one,
        # pad with token 0, and supervise its inclusive sequence boundary.
        width = min(1 + 32 * ((len(tokens) + 31) // 32), 1024)
        rows.append(
            {
                "tokens": tokens + [0] * (width - len(tokens)),
                "prompt_mask": [not len(prompt) <= index <= len(tokens) for index in range(width)],
            }
        )
    return rows
