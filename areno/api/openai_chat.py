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
        reasoning_content, content = _split_reasoning_content(
            tokenizer, token_ids, display_text, stop_strings=stop_strings
        )
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content
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


def _decode(tokenizer: Any, token_ids: list[int], *, skip_special_tokens: bool) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        return tokenizer.decode(token_ids)


# Literal think markers the chat template uses to separate reasoning from the
# answer. Bailing V3 registers them as added tokens with ``special=False``, so
# ``skip_special_tokens=True`` does not remove them from decoded text. Checkpoints
# disagree on the spelling: ``get_added_vocab`` exposes the word form (leading
# space plus ``thinking`` / ``response``) while ``added_tokens_decoder`` carries
# the angle-bracket tag form. Both spellings are listed and compared after
# stripping the leading space, so either representation resolves to the marker.
_THINK_OPEN_TEXTS = {" " + "thinking", " " + "<" + "think" + ">"}
_THINK_CLOSE_TEXTS = {" " + "response", " " + "<" + "/" + "think" + ">"}


def _norm_marker(text: str) -> str:
    return text.lstrip(" ")


def _added_token_ids(tokenizer: Any, texts: set[str]) -> set[int]:
    """Resolve which token ids in ``tokenizer`` correspond to the given markers.

    Reads the tokenizer's added vocabulary rather than relying on a hardcoded id,
    because ids differ per checkpoint. Unrecognised markers simply contribute no
    ids, which disables splitting for tokenizers that never define them.
    """

    wanted = {_norm_marker(text) for text in texts}
    ids: set[int] = set()
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if isinstance(decoder, dict):
        for token_id, entry in decoder.items():
            content = getattr(entry, "content", None)
            if content is None and isinstance(entry, dict):
                content = entry.get("content")
            if isinstance(content, str) and _norm_marker(content) in wanted:
                try:
                    ids.add(int(token_id))
                except (TypeError, ValueError):
                    continue
    if ids:
        return ids
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        try:
            vocab = get_added_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, dict):
            for text, token_id in vocab.items():
                if isinstance(text, str) and _norm_marker(text) in wanted:
                    try:
                        ids.add(int(token_id))
                    except (TypeError, ValueError):
                        continue
    return ids


def _split_reasoning_content(
    tokenizer: Any, token_ids: list[int], display_text: str, *, stop_strings: list[str] | None = None
) -> tuple[str, str]:
    """Separate reasoning span from the answer using the chat template semantics.

    The chat template treats everything before ``' response'`` as reasoning and the
    rest as the answer. This helper applies the same semantics, but locates the
    markers by **token id** instead of matching substrings on decoded text: the
    words "thinking"/"response" also occur in ordinary prose, so a substring split
    would truncate answers at the first literal match. Splitting is enabled only
    when the tokenizer actually defines the markers; other tokenizers keep the
    decoded text untouched in ``content``.

    With a marker present, tokens after ``' thinking'`` (if any) up to ``' response'``
    become ``reasoning_content`` and the remainder becomes ``content``. When the
    tokenizer defines the markers but the generation has not emitted a closing
    ``' response'`` yet (e.g. truncated mid-thought at ``max_tokens``), the entire
    span is reasoning with an empty answer.
    """

    stop = list(stop_strings or [])
    open_ids = _added_token_ids(tokenizer, _THINK_OPEN_TEXTS)
    close_ids = _added_token_ids(tokenizer, _THINK_CLOSE_TEXTS)
    marker_ids = open_ids | close_ids
    if not marker_ids:
        # Tokenizer does not define think markers; behave as before.
        return "", str(display_text).strip()

    open_pos = next((pos for pos, token_id in enumerate(token_ids) if token_id in open_ids), None)
    close_pos = next((pos for pos, token_id in enumerate(token_ids) if token_id in close_ids), None)
    if open_pos is None and close_pos is None:
        # Markers are defined but were never emitted: the whole turn is reasoning.
        reasoning, _ = _trim_stop_strings(_decode(tokenizer, list(token_ids), skip_special_tokens=True).strip(), stop)
        return reasoning, ""

    reasoning_start = 0 if open_pos is None else open_pos + 1
    if close_pos is None:
        reasoning_ids, content_ids = list(token_ids[reasoning_start:]), []
    else:
        reasoning_ids, content_ids = list(token_ids[reasoning_start:close_pos]), list(token_ids[close_pos + 1 :])
    reasoning, _ = _trim_stop_strings(_decode(tokenizer, reasoning_ids, skip_special_tokens=True).strip(), stop)
    content, _ = _trim_stop_strings(_decode(tokenizer, content_ids, skip_special_tokens=True).strip(), stop)
    return reasoning, content


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
