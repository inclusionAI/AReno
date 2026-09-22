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
# answer. They are registered as added tokens with ``special=False``, so
# ``skip_special_tokens=True`` does not remove them from decoded text. Both the
# word form (``thinking`` / ``response``) and the angle-bracket form
# (`` thinking`` / `` response``) are listed because checkpoints disagree on the
# spelling; the lookup strips surrounding whitespace so either resolves.
_THINK_OPEN_TEXTS = ("thinking", "<" + "think" + ">")
_THINK_CLOSE_TEXTS = ("response", "<" + "/" + "think" + ">")

# What a tokenizer emits for a multi-byte character whose trailing bytes have not
# arrived yet. A decode ending in this character is not stable and is held back
# by the streaming splitter until the next token completes it.
_REPLACEMENT_CHAR = "\ufffd"


def _norm_marker(text: str) -> str:
    return text.strip()


def _added_token_ids(tokenizer: Any, texts: tuple[str, ...]) -> set[int]:
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


class ReasoningSplitter:
    """Token-level state machine that separates reasoning from the answer.

    The chat template treats everything before the closing think marker as
    reasoning and the rest as the answer. Engines implement this as a token
    state machine rather than a post-hoc substring split, so the same instance
    serves both the non-streaming path (accumulate the whole turn) and the
    streaming path (forward per-token deltas as they arrive).

    Markers are located by **token id**, never by decoded text: the words
    "thinking" / "response" also occur in ordinary prose and inside other added
    tokens (``</tool_response>``), so a substring split truncates answers at the
    first literal match.

    Channel assignment follows the chat template:

    * tokens before the opening marker, if any, start in reasoning;
    * an opening marker moves the machine into reasoning;
    * a closing marker moves it into the answer;
    * with the markers defined but never emitted, the whole turn is reasoning
      (the answer stays empty). This lets a truncated thought stream verbatim.

    When the tokenizer defines no markers at all the splitter is *disabled* and
    every token is plain content, so non-reasoning models keep their exact
    previous behaviour.
    """

    __slots__ = (
        "_tokenizer",
        "_open_ids",
        "_close_ids",
        "_in_reasoning",
        "_reasoning_ids",
        "_content_ids",
        "_reasoning_emitted",
        "_content_emitted",
    )

    def __init__(self, tokenizer: Any, open_ids: set[int], close_ids: set[int]) -> None:
        self._tokenizer = tokenizer
        self._open_ids = open_ids
        self._close_ids = close_ids
        # Disabled splitter (no markers): everything is content.
        self._in_reasoning = bool(open_ids or close_ids)
        self._reasoning_ids: list[int] = []
        self._content_ids: list[int] = []
        # Characters already handed to the caller, per channel. Used to emit each
        # stable character exactly once across incremental decodes.
        self._reasoning_emitted = 0
        self._content_emitted = 0

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        *,
        open_ids: set[int] | None = None,
        close_ids: set[int] | None = None,
    ) -> "ReasoningSplitter":
        """Build a splitter for *tokenizer*, resolving markers by token id.

        Marker ids are taken from *open_ids* / *close_ids* when given (a model or
        generation config that declares reasoning tokens), falling back to the
        tokenizer's added vocabulary. The reference checkpoint's config declares
        no reasoning tokens, so the added vocabulary is the actual source there;
        the lookup tolerates both the word (``thinking`` / ``response``) and
        angle-bracket (`` thinking`` / `` response``) spellings.
        """

        if open_ids is None:
            open_ids = _added_token_ids(tokenizer, _THINK_OPEN_TEXTS)
        if close_ids is None:
            close_ids = _added_token_ids(tokenizer, _THINK_CLOSE_TEXTS)
        return cls(tokenizer, set(open_ids), set(close_ids))

    @property
    def enabled(self) -> bool:
        """True when the tokenizer defines think markers and splitting applies."""

        return bool(self._open_ids or self._close_ids)

    def push(self, token_id: int) -> tuple[str, str]:
        """Feed one token, returning the ``(reasoning_delta, content_delta)`` it adds.

        At most one of the two strings is non-empty; both are empty when the
        token was a consumed marker or contributed nothing visible yet. Deltas
        are produced by diffing the decoded prefix of the token's own channel,
        which keeps multi-token BPE pieces and partial UTF-8 sequences intact.
        """

        if token_id in self._open_ids:
            self._in_reasoning = True
            return "", ""
        if token_id in self._close_ids:
            self._in_reasoning = False
            return "", ""
        if self._in_reasoning:
            _, delta = self._append(self._reasoning_ids, token_id, self._reasoning_emitted)
            self._reasoning_emitted += len(delta)
            return delta, ""
        _, delta = self._append(self._content_ids, token_id, self._content_emitted)
        self._content_emitted += len(delta)
        return "", delta

    def _append(self, channel: list[int], token_id: int, emitted: int) -> tuple[str, str]:
        """Append *token_id* to *channel*, returning ``(stable_text, delta)``.

        A multi-byte character split across tokens decodes as U+FFFD until its
        trailing bytes arrive, so the tail of the decode is not stable yet. Only
        the part before any trailing replacement character is handed out; the
        caller picks up the remainder on the next push once it is complete.
        """

        channel.append(token_id)
        text = self._decode(channel)
        # Only a *trailing* replacement character marks an incomplete multi-byte
        # character; a U+FFFD in the middle is genuine decoded content and must
        # not be held back.
        stable = text.rstrip(_REPLACEMENT_CHAR) if text.endswith(_REPLACEMENT_CHAR) else text
        return stable, stable[emitted:]

    def _decode(self, token_ids: list[int]) -> str:
        return _decode(self._tokenizer, token_ids, skip_special_tokens=True)

    def flush(self) -> tuple[str, str]:
        """Return the ``(reasoning, content)`` tail still held back.

        A trailing incomplete multi-byte character is withheld during streaming
        until its bytes arrive. Call this once the stream ends so the final
        pending text is not dropped.
        """

        reasoning = self._decode(self._reasoning_ids)
        content = self._decode(self._content_ids)
        r_tail = reasoning[self._reasoning_emitted :]
        c_tail = content[self._content_emitted :]
        self._reasoning_emitted = len(reasoning)
        self._content_emitted = len(content)
        return r_tail, c_tail

    def finish(self) -> tuple[str, str]:
        """Return the accumulated ``(reasoning, content)`` after stripping.

        Stop-string trimming is applied by the caller, which owns the request's
        ``stop`` list.
        """

        return self._decode(self._reasoning_ids).strip(), self._decode(self._content_ids).strip()

    def split(
        self, token_ids: list[int], display_text: str, *, stop_strings: list[str] | None = None
    ) -> tuple[str, str]:
        """Replay *token_ids* through the machine and return stripped spans.

        Non-streaming helper. When the splitter is disabled it preserves the
        pre-existing behaviour: the decoded text is returned verbatim as content
        with no reasoning span.
        """

        if not self.enabled:
            return "", str(display_text).strip()
        for token_id in token_ids:
            self.push(token_id)
        reasoning, content = self.finish()
        stop = list(stop_strings or [])
        reasoning, _ = _trim_stop_strings(reasoning, stop)
        content, _ = _trim_stop_strings(content, stop)
        return reasoning, content


def _split_reasoning_content(
    tokenizer: Any, token_ids: list[int], display_text: str, *, stop_strings: list[str] | None = None
) -> tuple[str, str]:
    """Separate the reasoning span from the answer for one completed turn.

    Thin wrapper over :class:`ReasoningSplitter` kept for the existing
    non-streaming call path.
    """

    return ReasoningSplitter.from_tokenizer(tokenizer).split(
        token_ids, display_text, stop_strings=stop_strings
    )


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
