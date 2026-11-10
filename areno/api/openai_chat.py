"""Shared OpenAI-compatible chat helpers for serve and agentic rollout."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterator

from areno.api.tokenizer import apply_chat_template_with_options, normalize_token_ids
from areno.api.tool_call_parser import ToolCallParser


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize chat messages before tokenizer template rendering."""

    normalized = []
    for message in messages:
        item = dict(message)
        # OpenAI assistant tool-call messages commonly carry content=null.
        # Some local chat templates require a string while still preserving
        # the tool_calls payload.
        if item.get("content") is None:
            item["content"] = ""
        elif isinstance(item.get("content"), list):
            text_content = _text_only_content(item["content"])
            if text_content is not None:
                item["content"] = text_content
        if isinstance(item.get("tool_calls"), list):
            item["tool_calls"] = [_normalize_message_tool_call(call) for call in item["tool_calls"]]
        normalized.append(item)
    return normalized


def _text_only_content(content: list[Any]) -> str | None:
    """Flatten OpenAI text parts while preserving multimodal content lists."""

    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
            return None
        text = part.get("text")
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


def _normalize_message_tool_call(call: Any) -> Any:
    if not isinstance(call, dict):
        return call
    item = dict(call)
    function = item.get("function")
    if isinstance(function, dict):
        function = dict(function)
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                function["arguments"] = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                pass
        item["function"] = function
    return item


def messages_to_prompt_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    fallback_prompt: str = "",
) -> list[int]:
    """Tokenize an OpenAI-style message list with the model chat template."""

    messages = normalize_messages(messages)
    if getattr(tokenizer, "chat_template", None):
        messages = _embed_tools_in_system_message(tokenizer, messages, tools)
        kwargs: dict[str, Any] = {"tokenize": True, "add_generation_prompt": True}
        if tools:
            kwargs["tools"] = tools
        try:
            return normalize_token_ids(apply_chat_template_with_options(tokenizer, messages, **kwargs))
        except TypeError:
            if tools:
                kwargs["tools"] = _normalize_tools_for_chat_template(tools)
                try:
                    return normalize_token_ids(apply_chat_template_with_options(tokenizer, messages, **kwargs))
                except TypeError:
                    pass
            kwargs.pop("tools", None)
            return normalize_token_ids(apply_chat_template_with_options(tokenizer, messages, **kwargs))
    return normalize_token_ids(tokenizer.encode(messages_to_text(messages) or fallback_prompt))


def _embed_tools_in_system_message(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Adapt templates that read a JSON tool list from the system message.

    Phi-4-Multimodal's published template ignores the conventional global
    ``tools`` variable and instead concatenates ``message['tools']`` between
    ``<|tool|>`` markers. Keep the normal ``tools=`` call contract while also
    supplying that model-specific message field when the template declares it.
    """

    template = str(getattr(tokenizer, "chat_template", "") or "")
    embeds_message_tools = "<|tool|>" in template and (
        "'tools' in message" in template or '"tools" in message' in template
    )
    if not tools or not embeds_message_tools:
        return messages

    flat_tools = []
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        flat_tools.append(function if isinstance(function, dict) else tool)
    serialized_tools = json.dumps(flat_tools, ensure_ascii=False, separators=(",", ":"))

    rendered_messages = [dict(message) for message in messages]
    for message in rendered_messages:
        if message.get("role") == "system":
            message.setdefault("tools", serialized_tools)
            break
    else:
        rendered_messages.insert(0, {"role": "system", "content": "", "tools": serialized_tools})
    return rendered_messages


def _normalize_tools_for_chat_template(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        item = dict(tool)
        function = item.get("function")
        if isinstance(function, dict):
            function = dict(function)
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                function["parameters"] = {"type": "object", "properties": {}}
            elif not isinstance(parameters.get("properties", {}), dict):
                parameters = dict(parameters)
                parameters["properties"] = {}
                function["parameters"] = parameters
            item["function"] = function
        normalized.append(item)
    return normalized


def messages_to_text(messages: list[dict[str, Any]]) -> str:
    """Flatten text-like message content for tokenizers without chat templates."""

    parts = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def first_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the first user text, falling back to all text messages."""

    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return messages_to_text(messages)


def build_chat_completion_response(
    *,
    tokenizer: Any,
    model: str,
    prompt_tokens: int,
    response_ids: list[list[int]],
    finish_reasons: list[str],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: Any = None,
    tool_call_parser: ToolCallParser | None = None,
    parsed_tool_calls: list[list[dict[str, Any]]] | None = None,
    response_logprobs: list[list[float]] | None = None,
    routed_experts: list[list[list[list[int]]]] | None = None,
    include_areno_metadata: bool = False,
    input_tokens: list[int] | None = None,
    stop_strings: list[str] | None = None,
) -> dict[str, Any]:
    """Build an OpenAI chat-completion response and parse tool calls if asked."""

    tools = list(tools or [])
    stop_strings = list(stop_strings or [])
    response_logprobs = response_logprobs or [[] for _ in response_ids]
    choices: list[dict[str, Any]] = []
    completion_tokens = 0
    for index, token_ids in enumerate(response_ids):
        raw_text = _decode(tokenizer, token_ids, skip_special_tokens=False)
        display_text = _decode(tokenizer, token_ids, skip_special_tokens=True)
        display_text, stop_hit = _trim_stop_strings(display_text, stop_strings)
        completion_tokens += len(token_ids)
        tool_calls = (
            parsed_tool_calls[index] if parsed_tool_calls is not None and index < len(parsed_tool_calls) else []
        )
        if not tool_calls and tools and tool_call_parser is not None:
            tool_calls = tool_call_parser.parse(raw_text, tools, tool_choice).tool_calls
        finish_reason = "stop" if stop_hit or finish_reasons[index] == "stop" else "length"
        message: dict[str, Any] = {"role": "assistant", "content": display_text}
        if tool_calls:
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        choices.append({"index": index, "message": message, "finish_reason": finish_reason})

    response: dict[str, Any] = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    if include_areno_metadata:
        response["areno"] = {
            "input_tokens": list(input_tokens or []),
            "response_tokens": list(response_ids[0] if response_ids else []),
            "response_logprobs": list(response_logprobs[0] if response_logprobs else []),
            "routed_experts": list(routed_experts[0] if routed_experts else []),
        }
    return response


def format_sse_data(payload: Any) -> bytes:
    """Frame one SSE `data:` event; pass the string ``"[DONE]"`` for the sentinel."""

    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n".encode("utf-8")


def iter_chat_completion_chunks(
    response: dict[str, Any],
    *,
    tokenizer: Any = None,
    response_ids: list[list[int]] | None = None,
    include_usage: bool = False,
) -> Iterator[dict[str, Any]]:
    """Split a completed chat-completion envelope into OpenAI streaming chunks.

    Yields ``chat.completion.chunk`` objects: a role chunk and token-aligned
    content deltas per choice, then a terminal chunk carrying the finish reason
    (and the full ``tool_calls`` list when the message parsed any — callers that
    only accumulate content still see the raw tool-call text via non-stream
    responses, while streaming clients read them from this delta), and finally
    an empty-choices usage chunk when ``include_usage`` is set.

    ``response_ids``/``tokenizer`` make content deltas token-aligned via
    cumulative prefix decoding; without them each choice streams as one delta.
    """

    base = {
        "id": response["id"],
        "object": "chat.completion.chunk",
        "created": response["created"],
        "model": response["model"],
    }

    def chunk(index: int | None, delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
        payload = dict(base)
        if index is None:
            payload["choices"] = []
        else:
            payload["choices"] = [{"index": index, "delta": delta, "finish_reason": finish_reason}]
        return payload

    for choice in response.get("choices", []):
        index = int(choice["index"])
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason") or "stop"
        yield chunk(index, {"role": "assistant"})
        tool_calls = message.get("tool_calls")
        if tool_calls:
            yield chunk(index, {"tool_calls": tool_calls}, finish_reason)
            continue
        content = message.get("content") or ""
        ids = response_ids[index] if response_ids is not None and index < len(response_ids) else None
        for piece in _content_deltas(tokenizer, ids, content):
            yield chunk(index, {"content": piece})
        yield chunk(index, {}, finish_reason)
    if include_usage:
        usage_chunk = dict(base)
        usage_chunk["choices"] = []
        usage_chunk["usage"] = response.get("usage", {})
        yield usage_chunk


def _content_deltas(tokenizer: Any, token_ids: list[int] | None, content: str) -> list[str]:
    """Split `content` into token-aligned pieces via cumulative prefix decoding.

    The concatenation of the returned pieces is exactly `content`, even when
    stop-string trimming or special-token skipping makes the full decode differ
    from the per-token concatenation.
    """

    if not content:
        return []
    if tokenizer is None or not token_ids:
        return [content]
    deltas: list[str] = []
    accumulated = ""
    for cut in range(1, len(token_ids) + 1):
        prefix = _decode(tokenizer, token_ids[:cut], skip_special_tokens=True)
        if not prefix.startswith(accumulated):
            # Prefix decode churned (tokenizer normalization); fall through to
            # the remainder so the concatenation stays exact.
            break
        piece = prefix[len(accumulated) :]
        if not piece:
            continue
        if not content.startswith(prefix):
            # Decode went past the final text (e.g. a stop marker was trimmed).
            break
        deltas.append(piece)
        accumulated = prefix
        if accumulated == content:
            return deltas
    remainder = content[len(accumulated) :]
    if remainder:
        deltas.append(remainder)
    return deltas


def _decode(tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        return tokenizer.decode(token_ids)


def _trim_stop_strings(text: str, stop: list[str]) -> tuple[str, bool]:
    if not stop:
        return text, False
    first = None
    for marker in stop:
        idx = text.find(marker)
        if idx >= 0 and (first is None or idx < first):
            first = idx
    if first is None:
        return text, False
    return text[:first], True
