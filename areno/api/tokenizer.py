"""Tokenizer loading and prompt-encoding helpers.

The HuggingFace tokenizer loader sometimes blows up on configs that store
`extra_special_tokens` as a list; the shim here works around that. The other
helpers normalise EOS handling (multi-EOS configs are common in chat models)
and apply chat templates only when the prompt is not already formatted.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from areno.engine.data.tokenizer import load_tokenizer as load_tokenizer  # noqa: F401

_CHAT_TEMPLATE_ENABLE_THINKING_ATTR = "_areno_chat_template_enable_thinking"


def configure_chat_template_enable_thinking(tokenizer, enable_thinking: bool | None) -> None:
    """Store the optional chat-template thinking switch on a tokenizer.

    ``None`` keeps the tokenizer default.  A boolean is passed to
    ``apply_chat_template`` calls when the tokenizer/template supports it.
    """

    if enable_thinking is None:
        if hasattr(tokenizer, _CHAT_TEMPLATE_ENABLE_THINKING_ATTR):
            delattr(tokenizer, _CHAT_TEMPLATE_ENABLE_THINKING_ATTR)
        return
    setattr(tokenizer, _CHAT_TEMPLATE_ENABLE_THINKING_ATTR, bool(enable_thinking))


def apply_chat_template_with_options(tokenizer, messages, **kwargs):
    """Apply a tokenizer chat template with AReno-level optional kwargs."""

    kwargs = dict(kwargs)
    enable_thinking = getattr(tokenizer, _CHAT_TEMPLATE_ENABLE_THINKING_ATTR, None)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = bool(enable_thinking)
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        if "enable_thinking" not in kwargs:
            raise
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def eos_token_ids(model_path: str | Path, tokenizer) -> tuple[int, ...]:
    """Collect EOS ids from tokenizer and HF config.

    Some multimodal/chat configs expose multiple EOS ids at the top level and
    inside `text_config`; rollout should stop on any of them. Duplicates are
    removed while preserving first-seen order.
    """

    ids: list[int] = []
    _extend_token_ids(ids, getattr(tokenizer, "eos_token_id", None))
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        _extend_token_ids(ids, config.get("eos_token_id"))
        # Multimodal models (e.g. minicpm-v) nest LM config under text_config
        # and may declare a different EOS id for the language tower.
        text_config = config.get("text_config")
        if isinstance(text_config, dict):
            _extend_token_ids(ids, text_config.get("eos_token_id"))
    return tuple(dict.fromkeys(ids))


def encode_generation_prompt(tokenizer, prompt: str) -> list[int]:
    """Encode a prompt for generation, applying chat template when available.

    If the prompt already contains chat-format markers we keep it verbatim so
    upstream pipelines that build their own messages are not double-wrapped.
    """

    if _looks_chat_formatted(prompt) or not getattr(tokenizer, "chat_template", None):
        return normalize_token_ids(tokenizer.encode(prompt))
    return normalize_token_ids(
        apply_chat_template_with_options(
            tokenizer,
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )


def normalize_token_ids(value: Any) -> list[int]:
    """Convert tokenizer outputs to a plain list of integer token ids."""

    if hasattr(value, "ids"):
        value = value.ids
    if hasattr(value, "input_ids"):
        value = value.input_ids
    if isinstance(value, list | tuple):
        if value and hasattr(value[0], "ids"):
            if len(value) != 1:
                raise TypeError("expected one tokenized prompt, got a batch of encodings")
            return normalize_token_ids(value[0])
        if value and isinstance(value[0], list | tuple):
            if len(value) != 1:
                raise TypeError("expected one tokenized prompt, got a batch of token id lists")
            return normalize_token_ids(value[0])
        try:
            return [int(token_id) for token_id in value]
        except TypeError as exc:
            raise TypeError(f"expected token ids, got {type(value).__name__}") from exc
    raise TypeError(f"expected token ids or tokenizer Encoding, got {type(value).__name__}")


def _looks_chat_formatted(prompt: str) -> bool:
    # Heuristic: presence of any known turn marker is enough to skip the chat
    # template and avoid re-wrapping an already-formatted prompt.
    markers = ("<|im_start|>", "<start_of_turn>", "<turn|>", "<|user|>", "<|assistant|>")
    return any(marker in prompt for marker in markers)


def _extend_token_ids(out: list[int], value) -> None:
    # EOS can be a single int or a list of ints in HF configs; accept both
    # without forcing the caller to branch.
    if value is None:
        return
    if isinstance(value, int):
        out.append(value)
        return
    if isinstance(value, list | tuple):
        for item in value:
            if isinstance(item, int):
                out.append(item)
