"""Conversation-aware splitter that forms samples below the context limit.

Splits occur only at complete message boundaries and never separate a tool
call from its result. A required system context prefix is copied into every
chunk.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def split_conversation(
    messages: list[dict[str, Any]],
    max_tokens: int,
    tokenizer: Any,
    *,
    system_context: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Split an OpenAI-format message list at message boundaries.

    Every resulting chunk carries the optional *system_context* prefix so the
    model receives a complete standalone conversation. ``assistant`` messages
    that carry ``tool_calls`` are kept together with the immediately following
    ``tool``-role messages.

    Parameters
    ----------
    messages : list[dict]
        Dialog in OpenAI chat format. Each dict has at least ``"role"` and
        ``"content"``.
    max_tokens : int
        Maximum token budget for each chunk (excluding the system prefix and
        tools, which are measured separately).
    tokenizer :
        HuggingFace tokenizer (or compatible) that exposes a ``chat_template``
        method.
    system_context : list[dict] or None
        If given, prepended to every chunk. Typically ``[{"role": "system",
        "content": "..."}]``. The system context is *always* included in each
        chunk regardless of chunk budget.
    tools : list[dict] or None
        Optional tool definitions forwarded to the chat template.

    Returns
    -------
    list[list[dict]]
        One or more independent message lists, each fitting within
        *max_tokens* (plus system/tools prefix).

    Raises
    ------
    ValueError
        If a single atomic unit already exceeds *max_tokens*.
    """
    from areno.api.openai_chat import messages_to_prompt_tokens

    if not messages:
        return []

    units = _build_atomic_units(messages)
    prefix_tokens = _count_prefix(system_context, tokenizer, tools=tools)
    unit_token_counts = _token_counts(units, tokenizer, tools=tools)

    # Check unsplittable units before trying to form any chunk.
    _fail_if_any_unit_exceeds_limit(unit_token_counts, max_tokens)

    chunks: list[list[dict[str, Any]]] = []
    system_prefix = list(system_context or [])

    cursor = 0
    while cursor < len(units):
        used = 0
        chunk: list[dict[str, Any]] = []
        while cursor < len(units) and used + prefix_tokens + unit_token_counts[cursor] <= max_tokens:
            chunk.extend(units[cursor])
            used += unit_token_counts[cursor]
            cursor += 1
        if not chunk:
            raise ValueError(
                f"message unit [{cursor}] ({unit_token_counts[cursor]} tokens) "
                f"exceeds chunk limit ({max_tokens})"
            )
        chunks.append(system_prefix + _copy_messages(chunk))

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_atomic_units(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group *messages* into unsplittable units.

    Tool-call grouping: an ``assistant`` message that carries ``tool_calls``
    is bundled with every consecutive ``tool``-role message that follows it.
    This prevents training on an incomplete tool interaction.
    """
    if not messages:
        return []

    units: list[list[dict[str, Any]]] = []
    pending: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        if role == "tool":
            pending.append(dict(msg))
            continue
        if pending:
            units.append(pending)
            pending = []
        if role == "assistant" and _has_tool_calls(msg):
            pending.append(dict(msg))
            continue
        units.append([dict(msg)])

    if pending:
        units.append(pending)

    return units


def _has_tool_calls(message: dict[str, Any]) -> bool:
    tc = message.get("tool_calls")
    return isinstance(tc, list) and len(tc) > 0


def _copy_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(msg) for msg in messages]


def _token_counts(
    units: list[list[dict[str, Any]]],
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Return the token count for each atomic unit."""
    from areno.api.openai_chat import messages_to_prompt_tokens

    return [len(messages_to_prompt_tokens(tokenizer, unit, tools=tools)) for unit in units]


def _count_prefix(
    prefix: list[dict[str, Any]] | None,
    tokenizer: Any,
    *,
    tools: list[dict[str, Any]] | None = None,
) -> int:
    if not prefix:
        return 0
    from areno.api.openai_chat import messages_to_prompt_tokens

    return len(messages_to_prompt_tokens(tokenizer, prefix, tools=tools))


def _fail_if_any_unit_exceeds_limit(counts: list[int], limit: int) -> None:
    for idx, count in enumerate(counts):
        if count > limit:
            raise ValueError(
                f"message unit [{idx}] ({count} tokens) exceeds chunk limit "
                f"({limit}); this unit cannot be split — either a single message "
                f"is too long or a tool_call + tool_result pair must stay together"
            )