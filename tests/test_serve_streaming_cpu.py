"""CPU integration tests for SSE streaming via the /v1/chat/completions endpoint.

These tests exercise the full HTTP → StreamingResponse → SSE pipeline with a mocked
engine, verifying that the serve module produces OpenAI-compatible SSE output.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from areno.cli import serve as serve_mod
from areno.cli.serve import (
    ChatCompletionChoice,
    ChatCompletionResponse,
    ChatCompletionUsage,
    _ServeRollout,
)


# -- Helpers ------------------------------------------------------------------


def _text_response(content: str, finish_reason: str = "stop") -> ChatCompletionResponse:
    """Build a single-choice text ChatCompletionResponse."""
    return ChatCompletionResponse(
        id="chatcmpl-test",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        choices=[
            ChatCompletionChoice(
                index=0,
                message={"role": "assistant", "content": content},
                finish_reason=finish_reason,
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _tool_call_response() -> ChatCompletionResponse:
    """Build a single-choice tool-call ChatCompletionResponse."""
    return ChatCompletionResponse(
        id="chatcmpl-tool",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        choices=[
            ChatCompletionChoice(
                index=0,
                message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'},
                        }
                    ],
                },
                finish_reason="tool_calls",
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=8, completion_tokens=12, total_tokens=20),
    )


# -- Mock infrastructure ------------------------------------------------------


def _make_stream_generator(response_ids: list[list[int]]):
    """Return an async callable that yields ``_ServeStreamStep`` objects.

    Each invocation creates a fresh async generator that replays the given token
    sequences for prompt index 0, appending ``"stop"`` as the finish reason for
    the last token of each sequence.
    """

    from areno.cli.serve import _ServeStreamStep

    async def _generate(*args, **kwargs):
        for ids in response_ids:
            for j, token_id in enumerate(ids):
                is_last = j == len(ids) - 1
                yield _ServeStreamStep(0, token_id, "stop" if is_last else None)

    return _generate


@contextmanager
def _mock_serve(response: ChatCompletionResponse, *, response_ids: list[list[int]] | None = None):
    """Mock serve-module internals so streaming requests resolve with *response*.

    When *response_ids* is given, the mock engine returns a ``_ServeRollout`` so
    the SSE path exercises token-level incremental decode.  The mock tokenizer
    decodes ``[id_1, id_2, …]`` as ``"T1T2…"``, allowing per-token delta diffs.
    """
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode = lambda ids, **kw: "".join(f"T{i}" for i in ids)

    mock_engine = MagicMock()
    mock_engine.max_model_len = 4096
    mock_engine.tokenizer = mock_tokenizer
    mock_engine.processor = None
    if response_ids is not None:
        mock_engine.generate_rollout_async = AsyncMock(
            return_value=_ServeRollout(
                response_ids=response_ids,
                finish_reason=[response.choices[0].finish_reason for _ in response_ids],
            )
        )
        # Also wire up generate_rollout_stream_async so the true-streaming
        # path (n=1, no tools) can be exercised with the same token data.
        mock_engine.generate_rollout_stream_async = _make_stream_generator(response_ids)
    else:
        mock_engine.generate_rollout_async = AsyncMock(
            return_value=_ServeRollout(response_ids=[[1]], finish_reason=["stop"])
        )
        mock_engine.generate_rollout_stream_async = _make_stream_generator([[1]])
    # Store the raw engine so non-streaming _run_request_rollout can reach it.
    mock_engine._engine = mock_engine

    mock_tokenizer.chat_template = None

    _create_app_patches = [
        patch.object(serve_mod, "default_backend_type", return_value=serve_mod.BackendType.MLX),
        patch.object(serve_mod, "load_tokenizer", return_value=mock_tokenizer),
        patch.object(serve_mod, "load_processor", return_value=None),
        patch.object(serve_mod, "configure_chat_template_enable_thinking"),
        patch.object(serve_mod, "_resolve_serve_attn_backend", return_value=("native", None)),
        patch.object(serve_mod, "_create_serve_runtime", return_value=mock_engine),
        patch.object(serve_mod, "get_tool_call_parser", return_value=MagicMock()),
        patch.object(serve_mod, "infer_tool_call_parser_name", return_value=""),
    ]
    for p in _create_app_patches:
        p.start()

    _orig_encode = serve_mod._encode_messages_with_features
    _orig_stop_ids = serve_mod._stop_token_ids
    _orig_eos_id = serve_mod._first_eos_token_id

    serve_mod._encode_messages_with_features = MagicMock(return_value=([1, 2, 3], None))
    serve_mod._stop_token_ids = MagicMock(return_value=())
    serve_mod._first_eos_token_id = MagicMock(return_value=None)

    try:
        yield
    finally:
        serve_mod._encode_messages_with_features = _orig_encode
        serve_mod._stop_token_ids = _orig_stop_ids
        serve_mod._first_eos_token_id = _orig_eos_id
        for p in _create_app_patches:
            p.stop()


def _build_streaming_app():
    """Build a real FastAPI app via ``create_app`` with all heavy deps mocked.

    Must be called inside a ``_mock_serve`` context.
    """
    return serve_mod.create_app(
        model_path="/mock/model",
        tp_size=1,
        world_size=1,
        max_running_prompts=4,
        default_max_tokens=256,
        decode_progress_interval_s=0.0,
    )


def _collect_sse_events(app, request_body: dict) -> list:
    """Make a streaming HTTP request; return parsed SSE events.

    Each element is a ``dict`` (JSON data chunk) or ``"[DONE]"`` sentinel.
    """

    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            async with client.stream("POST", "/v1/chat/completions", json=request_body) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]

                events: list = []
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        events.append("[DONE]")
                    else:
                        events.append(json.loads(payload))
                return events

    return asyncio.run(_run())


def _content_deltas(events: list) -> list[str]:
    """Extract non-empty content delta strings from parsed SSE events."""
    return [
        c["choices"][0]["delta"]["content"]
        for c in events
        if c != "[DONE]"
        and c.get("choices")
        and c["choices"][0]["delta"].get("content", "") != ""
    ]


def _without_sentinel(events: list) -> list[dict]:
    """Return only the JSON data chunks, dropping ``[DONE]``."""
    return [e for e in events if e != "[DONE]"]


# -- Tests --------------------------------------------------------------------


class TestStreamingEndpoint:
    """Integration tests for the ``/v1/chat/completions`` streaming endpoint.

    Every test goes through the full HTTP stack with mocked engine and
    tokenizer — no GPU or model weights needed.
    """

    @staticmethod
    def _stream(n_tokens: int = 3, *, finish_reason: str = "stop") -> list:
        """Stream a text response with *n_tokens* and return parsed SSE events."""
        response_ids = [[i + 1 for i in range(n_tokens)]]
        with _mock_serve(
            _text_response("ignored — SSE deltas come from tokenizer decode", finish_reason=finish_reason),
            response_ids=response_ids,
        ):
            app = _build_streaming_app()
            return _collect_sse_events(
                app,
                {"messages": [{"role": "user", "content": "Hi"}], "stream": True},
            )

    # -- tests -----------------------------------------------------------------

    def test_first_chunk_contains_role(self):
        """The very first SSE data chunk must set ``delta.role = 'assistant'``."""
        events = self._stream()
        assert events[0]["choices"][0]["delta"]["role"] == "assistant"

    def test_starts_with_chat_completion_chunk_object(self):
        """Every data chunk must have ``object = 'chat.completion.chunk'``."""
        events = self._stream()
        for chunk in _without_sentinel(events):
            assert chunk["object"] == "chat.completion.chunk"

    def test_final_chunk_is_done(self):
        """The stream must end with a ``[DONE]`` sentinel."""
        events = self._stream()
        assert events[-1] == "[DONE]"

    def test_token_level_streaming(self):
        """Each content delta corresponds to exactly one token.

        The mock tokenizer produces ``"T{i}"`` for token id *i*, so deltas
        are ``["T1", "T2", ...]`` and concatenate back to ``"T1T2..."``.
        """
        n = 7
        events = self._stream(n_tokens=n)

        deltas = _content_deltas(events)
        assert deltas == [f"T{i}" for i in range(1, n + 1)]
        assert "".join(deltas) == "".join(f"T{i}" for i in range(1, n + 1))

    def test_empty_content_still_streams_role_and_done(self):
        """Empty token list still emits role chunk and ``[DONE]``, no content deltas."""
        events = self._stream(n_tokens=0)
        data_events = _without_sentinel(events)

        assert len(data_events) >= 2  # role + finish
        assert data_events[0]["choices"][0]["delta"]["role"] == "assistant"
        assert events[-1] == "[DONE]"
        assert _content_deltas(events) == []

    def test_finish_reason_in_final_data_chunk(self):
        """Only the last data payload (before ``[DONE]``) carries ``finish_reason``."""
        events = self._stream()
        data_events = _without_sentinel(events)

        for chunk in data_events[:-1]:
            for choice in chunk["choices"]:
                assert choice["finish_reason"] is None

        last = data_events[-1]
        for choice in last["choices"]:
            assert choice["finish_reason"] == "stop"
            assert choice["delta"] == {}

    def test_usage_in_final_chunk(self):
        """The final data chunk must include usage token counts."""
        events = self._stream(n_tokens=5)
        last = _without_sentinel(events)[-1]

        # Prompt is mocked as [1, 2, 3], so prompt_tokens == 3.
        assert last["usage"]["prompt_tokens"] == 3
        assert last["usage"]["completion_tokens"] == 5
        assert last["usage"]["total_tokens"] == 8

    def test_tool_calls_in_delta(self):
        """Tool-call response emits ``tool_calls`` in delta with correct finish_reason."""
        tool_response = _tool_call_response()
        with _mock_serve(tool_response, response_ids=[[1, 2, 3]]):
            # The MLX fallback rebuilds the response from token ids, but the mock
            # tokenizer can't produce tool-call JSON.  Patch _build_response so the
            # pre-built tool-call response is used directly.
            with patch.object(serve_mod, "_build_response", return_value=tool_response):
                app = _build_streaming_app()
                events = _collect_sse_events(
                    app,
                    {
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                    },
                )

        data_events = _without_sentinel(events)

        assert data_events[0]["choices"][0]["delta"]["role"] == "assistant"

        tool_call_chunks = [
            c for c in data_events if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
        ]
        assert len(tool_call_chunks) >= 1
        tc = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"]
        assert tc[0]["function"]["name"] == "get_weather"
        assert '"city"' in tc[0]["function"]["arguments"]

        assert data_events[-1]["choices"][0]["finish_reason"] == "tool_calls"
        assert events[-1] == "[DONE]"

    def test_multi_choice_response(self):
        """Multi-choice (n > 1) emits per-choice deltas with correct indices."""
        response = ChatCompletionResponse(
            id="chatcmpl-multi",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                ChatCompletionChoice(index=0, message={"role": "assistant", "content": "Hi"}, finish_reason="stop"),
                ChatCompletionChoice(index=1, message={"role": "assistant", "content": "Hello"}, finish_reason="stop"),
            ],
            usage=ChatCompletionUsage(prompt_tokens=5, completion_tokens=6, total_tokens=11),
        )

        with _mock_serve(response, response_ids=[[1, 2], [3, 4, 5, 6, 7]]):
            app = _build_streaming_app()
            events = _collect_sse_events(
                app,
                {"messages": [{"role": "user", "content": "Hi"}], "stream": True, "n": 2},
            )

        data_events = _without_sentinel(events)
        assert events[-1] == "[DONE]"

        # Role chunk covers both choices.
        role_chunk = data_events[0]
        assert len(role_chunk["choices"]) == 2
        for c in role_chunk["choices"]:
            assert c["delta"]["role"] == "assistant"

        # Content appears for both choices.
        content_seen: set[int] = set()
        for chunk in data_events[1:-1]:
            for c in chunk["choices"]:
                if c["delta"].get("content"):
                    content_seen.add(c["index"])
        assert content_seen == {0, 1}

        # Finish chunk covers both choices.
        finish_chunk = data_events[-1]
        assert len(finish_chunk["choices"]) == 2
        for c in finish_chunk["choices"]:
            assert c["finish_reason"] == "stop"
            assert c["delta"] == {}

    def test_non_streaming_request(self):
        """A ``stream=False`` request returns a complete JSON response, not SSE."""
        with _mock_serve(_text_response("Hi"), response_ids=[[1]]):
            app = _build_streaming_app()

            async def _run():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    return await client.post(
                        "/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "Hi"}], "stream": False},
                    )

            resp = asyncio.run(_run())
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/json"
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "T1"