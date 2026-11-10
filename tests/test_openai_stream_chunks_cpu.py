from __future__ import annotations

import io
import json
import unittest

from areno.api.agentic import _write_sse_chat
from areno.api.openai_chat import format_sse_data, iter_chat_completion_chunks


class FakeTokenizer:
    """Decodes each token id to its piece so prefix diffs are token-aligned."""

    def __init__(self, pieces: list[str]):
        self.pieces = pieces

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.pieces[i] for i in token_ids)


class FakeHandler:
    """Minimal BaseHTTPRequestHandler stand-in capturing the SSE write."""

    def __init__(self):
        self.status = None
        self.headers: list[tuple[str, str]] = []
        self.wfile = io.BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers.append((key, value))

    def end_headers(self):
        pass


def make_response(choice_messages, finish_reasons=None, *, usage=True):
    finish_reasons = finish_reasons or ["stop"] * len(choice_messages)
    choices = [
        {"index": index, "message": message, "finish_reason": finish_reasons[index]}
        for index, message in enumerate(choice_messages)
    ]
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "model-x",
        "choices": choices,
    }
    if usage:
        response["usage"] = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    return response


def chunks_of(response, **kwargs):
    return list(iter_chat_completion_chunks(response, **kwargs))


class ContentChunkTest(unittest.TestCase):
    """Content streams as token-aligned deltas that concatenate exactly."""

    def test_token_aligned_deltas_concatenate_to_content(self):
        response = make_response([{"role": "assistant", "content": "Hello world"}])
        chunks = chunks_of(
            response,
            tokenizer=FakeTokenizer(["Hel", "lo", " world"]),
            response_ids=[[0, 1, 2]],
        )
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        content_deltas = [c["choices"][0]["delta"]["content"] for c in chunks[1:-1]]
        self.assertEqual(content_deltas, ["Hel", "lo", " world"])
        self.assertEqual("".join(content_deltas), "Hello world")
        final = chunks[-1]["choices"][0]
        self.assertEqual(final["delta"], {})
        self.assertEqual(final["finish_reason"], "stop")

    def test_stop_trimmed_content_streams_only_kept_text(self):
        pieces = ["Answer", ": 42", " STOP", " ignored"]
        response = make_response([{"role": "assistant", "content": "Answer: 42"}])
        chunks = chunks_of(response, tokenizer=FakeTokenizer(pieces), response_ids=[[0, 1, 2, 3]])
        content_deltas = [c["choices"][0]["delta"]["content"] for c in chunks[1:-1]]
        self.assertEqual("".join(content_deltas), "Answer: 42")

    def test_without_tokenizer_content_is_one_delta(self):
        response = make_response([{"role": "assistant", "content": "whole text"}])
        chunks = chunks_of(response)
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "whole text"})

    def test_empty_content_streams_role_and_finish_only(self):
        response = make_response([{"role": "assistant", "content": ""}])
        chunks = chunks_of(response, tokenizer=FakeTokenizer([]), response_ids=[[]])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[1]["choices"][0]["finish_reason"], "stop")

    def test_multiple_choices_keep_their_indices(self):
        response = make_response(
            [
                {"role": "assistant", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        )
        chunks = chunks_of(response)
        # Choices stream in index groups; clients dispatch deltas by index.
        indices = [c["choices"][0]["index"] for c in chunks]
        self.assertEqual(indices, [0, 0, 0, 1, 1, 1])
        deltas = [c["choices"][0]["delta"] for c in chunks]
        self.assertEqual(
            deltas,
            [
                {"role": "assistant"},
                {"content": "a"},
                {},
                {"role": "assistant"},
                {"content": "b"},
                {},
            ],
        )


class ToolCallChunkTest(unittest.TestCase):
    """Parsed tool calls ride the terminal delta; content is not duplicated."""

    def test_tool_calls_arrive_on_terminal_delta(self):
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
        response = make_response(
            [{"role": "assistant", "content": None, "tool_calls": tool_calls}],
            finish_reasons=["tool_calls"],
        )
        chunks = chunks_of(response, tokenizer=FakeTokenizer(["<tool>"]), response_ids=[[0]])
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        self.assertEqual(len(chunks), 2)
        terminal = chunks[1]["choices"][0]
        self.assertEqual(terminal["delta"], {"tool_calls": tool_calls})
        self.assertEqual(terminal["finish_reason"], "tool_calls")


class UsageAndFramingTest(unittest.TestCase):
    """Usage appears only when requested; SSE framing is OpenAI-compatible."""

    def test_usage_chunk_only_when_requested(self):
        response = make_response([{"role": "assistant", "content": "x"}])
        without = chunks_of(response, include_usage=False)
        self.assertNotIn("usage", without[-1])
        with_usage = chunks_of(response, include_usage=True)
        self.assertEqual(with_usage[-1]["choices"], [])
        self.assertEqual(with_usage[-1]["usage"]["total_tokens"], 8)

    def test_chunk_envelope_fields(self):
        response = make_response([{"role": "assistant", "content": "x"}])
        for chunk in chunks_of(response):
            self.assertEqual(chunk["id"], "chatcmpl-test")
            self.assertEqual(chunk["object"], "chat.completion.chunk")
            self.assertEqual(chunk["model"], "model-x")

    def test_format_sse_data_frames_json_and_done(self):
        self.assertEqual(format_sse_data({"a": 1}), b'data: {"a": 1}\n\n')
        self.assertEqual(format_sse_data("[DONE]"), b"data: [DONE]\n\n")


class AgenticSseTest(unittest.TestCase):
    """The agentic proxy replays its completed response as an SSE stream."""

    def test_write_sse_chat_emits_chunks_and_done(self):
        response = make_response([{"role": "assistant", "content": "Hi there"}])
        response["areno"] = {"response_tokens": [0, 1]}
        handler = FakeHandler()
        _write_sse_chat(handler, response, tokenizer=FakeTokenizer(["Hi", " there"]), include_usage=True)
        self.assertEqual(handler.status, 200)
        self.assertIn(("content-type", "text/event-stream"), handler.headers)
        events = [e for e in handler.wfile.getvalue().decode().split("\n\n") if e]
        self.assertTrue(events[-1].endswith("data: [DONE]"))
        parsed = [json.loads(e.split("data: ", 1)[1]) for e in events[:-1]]
        contents = [c["choices"][0]["delta"].get("content") for c in parsed if c["choices"] and "content" in c["choices"][0]["delta"]]
        self.assertEqual("".join(contents), "Hi there")
        self.assertEqual(parsed[-1]["choices"], [])
        self.assertEqual(parsed[-1]["usage"]["total_tokens"], 8)


if __name__ == "__main__":
    unittest.main()
