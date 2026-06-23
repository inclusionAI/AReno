"""Dataset tokenization helpers shared by offline trainers."""

from __future__ import annotations

from typing import Any

from areno.api.tokenizer import encode_generation_prompt, normalize_token_ids


def apply_chat_template(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    """Encode full chat messages, with a plain-text fallback for base tokenizers."""

    if getattr(tokenizer, "chat_template", None):
        return normalize_token_ids(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)
    return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))


def encode_prompt_value(tokenizer, prompt) -> list[int]:
    """Encode a DPO prompt that may be either plain text or chat messages."""

    if isinstance(prompt, list):
        if getattr(tokenizer, "chat_template", None):
            return normalize_token_ids(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
        text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in prompt)
        return encode_generation_prompt(tokenizer, text)
    return encode_generation_prompt(tokenizer, prompt)


def prompt_response_to_tokens_and_mask(
    prompt: str, response: str, tokenizer, eos_token_id: int
) -> tuple[list[int], list[bool]]:
    """Encode prompt text plus response text and mask the prompt prefix."""

    prompt_ids = encode_generation_prompt(tokenizer, prompt)
    return response_to_tokens_and_mask(prompt_ids, response, tokenizer, eos_token_id)


def response_to_tokens_and_mask(
    prompt_ids: list[int], response: str, tokenizer, eos_token_id: int
) -> tuple[list[int], list[bool]]:
    """Append a response to pre-tokenized prompt ids and mask prompt tokens."""

    response_ids = normalize_token_ids(tokenizer.encode(response, add_special_tokens=False))
    if eos_token_id is not None and (not response_ids or response_ids[-1] != eos_token_id):
        response_ids.append(eos_token_id)
    return prompt_ids + response_ids, [True] * len(prompt_ids) + [False] * len(response_ids)


def has_any(record: dict[str, Any], keys: tuple[str, ...]) -> bool:
    """Return whether a record has any string field in keys."""

    return any(isinstance(record.get(key), str) for key in keys)


def first_text(record: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return the first string field for required text schemas."""

    for key in keys:
        value = record.get(key)
        if isinstance(value, str):
            return value
    raise KeyError(keys[0])


def first_value(record: dict[str, Any], keys: tuple[str, ...]):
    """Return the first string/list field for optional preference schemas."""

    for key in keys:
        value = record.get(key)
        if isinstance(value, str | list):
            return value
    return None
