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

from areno.engine.data.tokenizer import load_processor as load_processor  # noqa: F401
from areno.engine.data.tokenizer import load_tokenizer as load_tokenizer  # noqa: F401

_CHAT_TEMPLATE_ENABLE_THINKING_ATTR = "_areno_chat_template_enable_thinking"


def configure_chat_template_enable_thinking(tokenizer, enable_thinking: bool | None) -> None:
    """Store the optional chat-template thinking switch on a tokenizer.

    ``None`` keeps the tokenizer default.  A boolean is passed to
    ``apply_chat_template`` calls when the tokenizer/template supports it.
    """

    if tokenizer is None:
        return
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
    """Collect EOS ids from generation config, tokenizer, and model config.

    Generation config is authoritative for chat checkpoints whose tokenizer
    exposes only a generic end-of-text token. Some multimodal configs also
    expose EOS ids at the top level and inside ``text_config``; rollout should
    stop on any of them. Duplicates are removed while preserving first-seen
    order.
    """

    ids: list[int] = []
    generation_config_path = Path(model_path) / "generation_config.json"
    if generation_config_path.exists():
        with generation_config_path.open("r", encoding="utf-8") as f:
            generation_config = json.load(f)
        _extend_token_ids(ids, generation_config.get("eos_token_id"))
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
    markers = ("<|start|>", "<|im_start|>", "<start_of_turn>", "<turn|>", "<|user|>", "<|assistant|>")
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


TOKENIZER_PROBE = "AReno tokenizer compatibility probe: 12345 <|endoftext|>"


def vocab_size(tokenizer: Any) -> int | None:
    """Best-effort vocabulary size for a tokenizer-like object."""

    value = getattr(tokenizer, "vocab_size", None)
    if isinstance(value, int) and value > 0:
        return int(value)
    try:
        return int(len(tokenizer))
    except (TypeError, AttributeError):
        return None


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return [int(item) for item in value]
    except TypeError:
        return [int(value)]


def assert_compatible_tokenizers(student: Any, teacher: Any, *, probe_text: str = TOKENIZER_PROBE) -> None:
    """Reject teacher/student pairs whose token ids are not interchangeable.

    A teacher-forcing scorer consumes *student* token ids. A vocab mismatch or
    a different encoding would shift every per-token score while the loss still
    looked finite, so this check runs at trainer start-up instead of at loss
    time. Generation-only tokens (eos/bos) are not compared because the scoring
    model never generates text.
    """

    student_size, teacher_size = vocab_size(student), vocab_size(teacher)
    if student_size is not None and teacher_size is not None and student_size != teacher_size:
        raise ValueError(
            f"OPD teacher and student vocab sizes differ ({teacher_size} vs {student_size}); "
            "teacher logprobs would be misaligned with student token ids"
        )
    # Only pad_token_id feeds the packed tensors shared with the teacher.
    # A Base student (eos <|endoftext|>) with a post-trained teacher
    # (eos <|im_end|>) is a valid and intended OPD setup.
    for attr in ("pad_token_id",):
        student_value = getattr(student, attr, None)
        teacher_value = getattr(teacher, attr, None)
        if student_value is not None and teacher_value is not None and student_value != teacher_value:
            raise ValueError(f"OPD teacher and student {attr} differ ({teacher_value} vs {student_value})")
    student_encode = getattr(student, "encode", None)
    teacher_encode = getattr(teacher, "encode", None)
    if callable(student_encode) and callable(teacher_encode):
        student_ids = _as_int_list(student_encode(probe_text))
        teacher_ids = _as_int_list(teacher_encode(probe_text))
        if student_ids != teacher_ids:
            raise ValueError(
                "OPD teacher and student tokenizers encode the same text differently; "
                "use a teacher from the same model family as the student"
            )
