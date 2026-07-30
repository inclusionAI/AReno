"""Shared OpenAI-compatible chat helpers for serve and agentic rollout."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

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
        if isinstance(item.get("tool_calls"), list):
            item["tool_calls"] = [_normalize_message_tool_call(call) for call in item["tool_calls"]]
        normalized.append(item)
    return normalized


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


def single_message_token_count(tokenizer, message: dict[str, Any]) -> int:
    """Template-aware token count of one message rendered alone.

    Renders the single message through the model's chat template with
    ``add_generation_prompt=False`` so the count is on the same scale the full
    prompt is measured (template/special-token aware), which is what the agentic
    overlength classifier needs to decide whether a single tool result already
    exceeds the context cap. Falls back to raw ``tokenizer.encode(text)`` when
    the tokenizer has no chat template or the template cannot render the
    message alone; returns ``0`` if even the fallback errors so the caller
    treats the message as not-decidably-oversized.
    """

    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    if getattr(tokenizer, "chat_template", None):
        try:
            return len(
                normalize_token_ids(
                    apply_chat_template_with_options(
                        tokenizer,
                        normalize_messages([message]),
                        tokenize=True,
                        add_generation_prompt=False,
                    )
                )
            )
        except Exception:
            pass
    try:
        return len(tokenizer.encode(text))
    except Exception:
        return 0


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
    include_areno_metadata: bool = False,
    input_tokens: list[int] | None = None,
    stop_strings: list[str] | None = None,
    termination_reason: str | None = None,
    force_finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build an OpenAI chat-completion response and parse tool calls if asked.

    ``termination_reason`` is an additive Areno-only signal (carried in the
    ``areno`` metadata block, never in the OpenAI ``finish_reason`` surface) so
    the agentic proxy can distinguish overlength causes without changing the
    wire shape seen by agent code. ``force_finish_reason`` overrides the derived
    ``finish_reason`` *only* when the choice has no tool calls (e.g. the
    safe-stop policy drops a half-finished tool call and wants to surface
    ``"length"``); it never converts a ``tool_calls`` finish reason.
    """

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
        # Only fall back to parsing from the decoded text when the caller did not
        # pre-parse. The agentic proxy always passes ``parsed_tool_calls`` (even
        # an empty list, e.g. after the safe-stop policy drops a half tool call),
        # so re-parsing here would silently re-introduce the dropped call.
        if not tool_calls and tools and tool_call_parser is not None and parsed_tool_calls is None:
            tool_calls = tool_call_parser.parse(raw_text, tools, tool_choice).tool_calls
        finish_reason = "stop" if stop_hit or finish_reasons[index] == "stop" else "length"
        message: dict[str, Any] = {"role": "assistant", "content": display_text}
        if tool_calls:
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            finish_reason = "tool_calls"
        elif force_finish_reason is not None:
            finish_reason = force_finish_reason
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
        areno_meta: dict[str, Any] = {
            "input_tokens": list(input_tokens or []),
            "response_tokens": list(response_ids[0] if response_ids else []),
            "response_logprobs": list(response_logprobs[0] if response_logprobs else []),
        }
        if termination_reason is not None:
            areno_meta["termination_reason"] = termination_reason
        response["areno"] = areno_meta
    return response


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
