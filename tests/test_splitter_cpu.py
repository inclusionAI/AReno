"""CPU tests for conversation-aware splitter (areno.api.splitter)."""

from __future__ import annotations

import pytest

from areno.api.splitter import (
    _build_atomic_units,
    _copy_messages,
    _count_prefix,
    _fail_if_any_unit_exceeds_limit,
    _has_tool_calls,
    _token_counts,
    split_conversation,
)


# ---------------------------------------------------------------------------
# Deterministic mock tokenizer
# ---------------------------------------------------------------------------


def _mock_tokenizer(words_per_message: int | None = None):
    """Return a SimpleNamespace that behaves like a chat-template tokenizer.

    By default each message is encoded as ``words_per_message`` placeholder
    tokens. Pass ``words_per_message`` in the call to override the default
    (which is the message content length after splitting on whitespace).
    """

    class _MockTokenizer:
        def __init__(self) -> None:
            self.chat_template = True
            self._words_per_message = words_per_message

        def apply_chat_template(self, messages, **kwargs):
            """Return a list of placeholder token ids."""
            return self.encode(
                "\n".join(
                    m.get("content", "") or ""
                    for m in messages
                    if isinstance(m.get("content"), str)
                )
            )

        def encode(self, text: str, **kwargs):
            words = text.split()
            count = len(words)
            if self._words_per_message is not None:
                count = self._words_per_message * sum(
                    1
                    for m in self._raw_messages
                    if isinstance(m.get("content"), str) and m.get("content")
                )
                assert count == len(words) * self._words_per_message or count == 0
            return list(range(count))

        def decode(self, token_ids, **kwargs):
            return " ".join(f"tok_{i}" for i in token_ids)

    return _MockTokenizer()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizer():
    return _mock_tokenizer()


@pytest.fixture
def system_context():
    return [{"role": "system", "content": "You are a helpful assistant."}]


@pytest.fixture
def short_messages():
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


@pytest.fixture
def long_messages():
    return [
        {"role": "user", "content": "a " * 50},
        {"role": "assistant", "content": "b " * 50},
        {"role": "user", "content": "c " * 50},
        {"role": "assistant", "content": "d " * 50},
    ]


@pytest.fixture
def messages_with_tool_calls():
    return [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "get_weather"}, "id": "1"}]},
        {"role": "tool", "tool_call_id": "1", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny today."},
    ]


# ---------------------------------------------------------------------------
# Tests: _build_atomic_units
# ---------------------------------------------------------------------------


class TestAtomicUnits:
    def test_single_user_message(self):
        messages = [{"role": "user", "content": "hello"}]
        units = _build_atomic_units(messages)
        assert units == [[{"role": "user", "content": "hello"}]]

    def test_multiple_simple_messages(self):
        messages = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        units = _build_atomic_units(messages)
        assert len(units) == 4

    def test_tool_call_binds_with_following_tool(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
        ]
        units = _build_atomic_units(messages)
        assert len(units) == 1
        assert len(units[0]) == 2

    def test_tool_call_binds_multiple_tool_results(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "tool_call_id": "1", "content": "r1"},
            {"role": "tool", "tool_call_id": "2", "content": "r2"},
        ]
        units = _build_atomic_units(messages)
        assert len(units) == 1
        assert len(units[0]) == 3

    def test_tool_call_separated_from_user_then_tool(self):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result"},
            {"role": "user", "content": "followup"},
        ]
        units = _build_atomic_units(messages)
        # assistant(tool_calls)+tool is 1 unit, user is 2nd unit
        assert len(units) == 2
        assert units[0][0]["role"] == "assistant"
        assert units[0][1]["role"] == "tool"
        assert units[1][0]["role"] == "user"

    def test_empty_messages(self):
        assert _build_atomic_units([]) == []

    def test_system_preserved_as_independent_unit(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        units = _build_atomic_units(messages)
        assert len(units) == 2


# ---------------------------------------------------------------------------
# Tests: _has_tool_calls
# ---------------------------------------------------------------------------


class TestHasToolCalls:
    def test_empty(self):
        assert _has_tool_calls({}) is False

    def test_empty_list(self):
        assert _has_tool_calls({"tool_calls": []}) is False

    def test_non_list(self):
        assert _has_tool_calls({"tool_calls": "not_a_list"}) is False

    def test_populated(self):
        assert _has_tool_calls({"tool_calls": [{"id": "1"}]}) is True


# ---------------------------------------------------------------------------
# Tests: _copy_messages
# ---------------------------------------------------------------------------


class TestCopyMessages:
    def test_deep_copies(self):
        original = [{"role": "user", "content": "hello"}]
        copied = _copy_messages(original)
        copied[0]["content"] = "changed"
        assert original[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# Tests: _token_counts / _count_prefix
# ---------------------------------------------------------------------------


class TestTokenCounts:
    def test_counts_single_unit(self, tokenizer):
        counts = _token_counts([[{"role": "user", "content": "hello world"}]], tokenizer)
        assert counts == [2]  # "hello", "world"

    def test_counts_multiple_units(self, tokenizer):
        units = [
            [{"role": "user", "content": "a b c"}],
            [{"role": "assistant", "content": "d e"}],
        ]
        counts = _token_counts(units, tokenizer)
        assert counts == [3, 2]

    def test_count_prefix_returns_zero_for_none(self, tokenizer):
        assert _count_prefix(None, tokenizer) == 0

    def test_count_prefix_returns_zero_for_empty(self, tokenizer):
        assert _count_prefix([], tokenizer) == 0

    def test_count_prefix_measures_tokens(self, tokenizer):
        assert _count_prefix([{"role": "system", "content": "x y"}], tokenizer) == 2


# ---------------------------------------------------------------------------
# Tests: _fail_if_any_unit_exceeds_limit
# ---------------------------------------------------------------------------


class TestFailIfAnyUnitExceedsLimit:
    def test_all_within_limit(self):
        _fail_if_any_unit_exceeds_limit([10, 20, 30], 100)

    def test_one_exceeds_raises(self):
        with pytest.raises(ValueError) as exc_info:
            _fail_if_any_unit_exceeds_limit([10, 200, 30], 100)
        assert "200 tokens" in str(exc_info.value)
        assert "[1]" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Tests: split_conversation (integration)
# ---------------------------------------------------------------------------


class TestSplitConversation:
    def test_under_limit_no_split(self, tokenizer, system_context):
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        result = split_conversation(messages, max_tokens=100, tokenizer=tokenizer, system_context=system_context)
        assert len(result) == 1
        # system prefix prepended
        assert result[0][0]["role"] == "system"
        assert result[0][1]["role"] == "user"
        assert result[0][2]["role"] == "assistant"

    def test_splits_at_message_boundary(self, tokenizer):
        messages = [
            {"role": "user", "content": "a " * 10},         # 10 tokens
            {"role": "assistant", "content": "b " * 10},     # 10 tokens
            {"role": "user", "content": "c " * 10},         # 10 tokens
            {"role": "assistant", "content": "d " * 10},     # 10 tokens
        ]
        # system prefix: 0 tokens (no system_context), 2 messages per chunk
        result = split_conversation(messages, max_tokens=22, tokenizer=tokenizer)

        assert len(result) >= 2
        total = sum(len(chunk) for chunk in result)
        assert total == 4  # all messages present once

    def test_system_context_copied_to_every_chunk(self, tokenizer, system_context):
        messages = [
            {"role": "user", "content": "a " * 20},
            {"role": "assistant", "content": "b " * 20},
            {"role": "user", "content": "c " * 20},
            {"role": "assistant", "content": "d " * 20},
        ]
        result = split_conversation(
            messages, max_tokens=42, tokenizer=tokenizer, system_context=system_context
        )
        assert len(result) >= 2
        for chunk in result:
            assert chunk[0] == {"role": "system", "content": "You are a helpful assistant."}

    def test_empty_messages(self, tokenizer):
        assert split_conversation([], max_tokens=100, tokenizer=tokenizer) == []

    def test_tool_call_and_result_kept_together(self, tokenizer):
        messages = [
            {"role": "user", "content": "a " * 5},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "result " * 5},
            {"role": "assistant", "content": "b " * 5},
        ]
        # Make limit tight so second assistant might need new chunk
        result = split_conversation(messages, max_tokens=16, tokenizer=tokenizer)

        # Verify the tool_call + tool_result stay together
        for chunk in result:
            for i, msg in enumerate(chunk):
                if msg.get("role") == "assistant" and _has_tool_calls(msg):
                    # Next message must be tool in same chunk
                    assert i + 1 < len(chunk)
                    assert chunk[i + 1]["role"] == "tool", "tool_call and tool_result must stay together"

    def test_single_unit_exceeds_limit_raises(self, tokenizer):
        messages = [
            {"role": "user", "content": "x " * 100},
        ]
        with pytest.raises(ValueError) as exc_info:
            split_conversation(messages, max_tokens=50, tokenizer=tokenizer)
        error_text = str(exc_info.value)
        assert "exceeds chunk limit" in error_text
        assert "100 tokens" in error_text

    def test_tool_call_unit_exceeds_limit(self, tokenizer):
        messages = [
            {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
            {"role": "tool", "tool_call_id": "1", "content": "y " * 100},
        ]
        with pytest.raises(ValueError) as exc_info:
            split_conversation(messages, max_tokens=50, tokenizer=tokenizer)
        error = str(exc_info.value)
        assert "exceeds chunk limit" in error

    def test_with_tools_argument(self, tokenizer):
        """split_conversation forwards tools to tokenizer."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = split_conversation(
            messages,
            max_tokens=100,
            tokenizer=tokenizer,
            tools=[{"type": "function", "function": {"name": "test"}}],
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: 默认关闭（backward compatibility）
# ---------------------------------------------------------------------------


class TestDefaultBehaviorUnchanged:
    """When split_messages is False, load_prompt_batches behaves identically."""

    def test_split_messages_false_uses_legacy_path(self):
        # Verify the method signature accepts split_messages with default False.
        from areno.api.trainer import Trainer

        # By inspecting the function signature we confirm backward compat.
        import inspect

        sig = inspect.signature(Trainer.load_prompt_batches)
        params = sig.parameters
        assert "split_messages" in params
        assert params["split_messages"].default is False


# ---------------------------------------------------------------------------
# Tests: 确定性输出
# ---------------------------------------------------------------------------


class TestDeterministicOutput:
    def test_same_input_same_chunks(self, tokenizer):
        messages = [
            {"role": "user", "content": "a " * 10},
            {"role": "assistant", "content": "b " * 10},
            {"role": "user", "content": "c " * 10},
            {"role": "assistant", "content": "d " * 10},
        ]
        a = split_conversation(messages, max_tokens=22, tokenizer=tokenizer)
        b = split_conversation(messages, max_tokens=22, tokenizer=tokenizer)
        assert len(a) == len(b)
        for ca, cb in zip(a, b):
            assert ca == cb

    def test_no_side_effects_on_input(self, tokenizer):
        messages = [
            {"role": "user", "content": "x y z"},
            {"role": "assistant", "content": "a b c"},
        ]
        snapshot = [dict(m) for m in messages]
        split_conversation(messages, max_tokens=100, tokenizer=tokenizer)
        assert messages == snapshot