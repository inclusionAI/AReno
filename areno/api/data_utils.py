"""Dataset tokenization helpers shared by offline trainers."""

from __future__ import annotations

from typing import Any

from areno.api.tokenizer import apply_chat_template_with_options, encode_generation_prompt, normalize_token_ids


def apply_chat_template(tokenizer, messages: list[dict[str, Any]]) -> list[int]:
    """Encode full chat messages, with a plain-text fallback for base tokenizers."""

    if getattr(tokenizer, "chat_template", None):
        return normalize_token_ids(
            apply_chat_template_with_options(tokenizer, messages, tokenize=True, add_generation_prompt=False)
        )
    text = "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in messages)
    return normalize_token_ids(tokenizer.encode(text, add_special_tokens=False))


def encode_prompt_value(tokenizer, prompt) -> list[int]:
    """Encode a DPO prompt that may be either plain text or chat messages."""

    if isinstance(prompt, list):
        if getattr(tokenizer, "chat_template", None):
            return normalize_token_ids(
                apply_chat_template_with_options(tokenizer, prompt, tokenize=True, add_generation_prompt=True)
            )
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


def _try_chat_template_encoding(
    messages: list[dict[str, Any]], tokenizer, trainable_assistant_indices: set[int]
) -> tuple[list[int], list[bool]] | None:
    """Attempt incremental chat-template encoding; return None if not prefix-stable.

    Tries to encode the conversation turn by turn using the tokenizer's
    ``chat_template``.  If the re-encoded prefix ever differs from what was
    already accumulated, the tokenizer is not prefix-stable and we return
    ``None`` so the caller can fall back to plain-text concatenation.
    """

    if not getattr(tokenizer, "chat_template", None):
        return None

    tokens: list[int] = []
    mask: list[bool] = []
    for i in range(len(messages)):
        partial_ids = normalize_token_ids(
            apply_chat_template_with_options(
                tokenizer, messages[: i + 1], tokenize=True, add_generation_prompt=False
            )
        )
        # Guard against tokenizers whose chat_template is not prefix-stable.
        if tokens and partial_ids[: len(tokens)] != tokens:
            import warnings

            warnings.warn(
                "tokenizer chat_template is not prefix-stable; "
                "falling back to plain-text encoding for multi-turn SFT",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        # Only keep tokens added by the current turn.
        new_tokens = partial_ids[len(tokens):]
        role = messages[i].get("role", "user")
        is_trainable = role == "assistant" and i in trainable_assistant_indices
        tokens.extend(new_tokens)
        mask.extend([not is_trainable] * len(new_tokens))
    return tokens, mask


def messages_to_tokens_and_mask(
    messages: list[dict[str, Any]],
    tokenizer,
    eos_token_id: int,
    *,
    last_assistant_only: bool = False,
) -> tuple[list[int], list[bool]]:
    """Encode multi-turn chat messages into tokens with a training mask.

    The mask follows the same convention as
    :func:`prompt_response_to_tokens_and_mask`: ``True`` means "do not train"
    (prompt context), ``False`` means "train" (assistant response).

    * user / system / tool turns are always masked out (``True``).
    * assistant turns are trainable (``False``) unless *last_assistant_only*
      is set, in which case only the final assistant turn is trainable and
      earlier assistant turns are treated as context (``True``).

    The function uses the tokenizer chat template when available so turn
    markers and special tokens match the model's expected format. For base
    tokenizers without a chat template, a plain-text fallback concatenates
    ``role: content`` per turn.

    EOS is appended after the last message if not already present, so the
    model learns to stop.
    """

    # Determine which assistant turns are trainable.
    assistant_indices = [
        i for i, msg in enumerate(messages) if msg.get("role") == "assistant"
    ]
    if last_assistant_only and assistant_indices:
        trainable_assistant_indices = {assistant_indices[-1]}
    else:
        trainable_assistant_indices = set(assistant_indices)

    chat_template_tokens = _try_chat_template_encoding(
        messages, tokenizer, trainable_assistant_indices
    )
    if chat_template_tokens is not None:
        tokens, mask = chat_template_tokens
    else:
        # Plain-text fallback: concatenate "role: content" per turn.
        tokens = []
        mask = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            turn_text = f"{role}: {content}"
            turn_ids = normalize_token_ids(tokenizer.encode(turn_text, add_special_tokens=False))
            is_trainable = role == "assistant" and i in trainable_assistant_indices
            tokens.extend(turn_ids)
            mask.extend([not is_trainable] * len(turn_ids))

    # Append EOS if not already present so the model learns to stop.
    if eos_token_id is not None and (not tokens or tokens[-1] != eos_token_id):
        tokens.append(eos_token_id)
        mask.append(False)

    return tokens, mask


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
