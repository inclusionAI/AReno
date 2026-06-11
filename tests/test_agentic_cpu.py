import importlib.util
import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from areno.api.tool_call_parser import Gemma4ToolCallParser, JsonToolCallParser, MiniCPMToolCallParser, QwenToolCallParser


def _load_agentic_module():
    path = Path(__file__).resolve().parents[1] / "areno" / "api" / "agentic.py"
    spec = importlib.util.spec_from_file_location("agentic_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agentic = _load_agentic_module()
AgentBatch = agentic.AgentBatch
LossMaskPolicy = agentic.LossMaskPolicy
RolloutSession = agentic.RolloutSession


def test_agent_batch_expands_records_by_n_samples():
    batch = AgentBatch(records=[{"prompt": "p0"}, {"prompt": "p1"}], prompts=["p0", "p1"], input_tokens=[[1], [2]], n_samples=2)

    items = list(batch.iter_samples())

    assert [(item.prompt_index, item.sample_index, item.prompt) for item in items] == [
        (0, 0, "p0"),
        (0, 1, "p0"),
        (1, 0, "p1"),
        (1, 1, "p1"),
    ]


def test_tool_call_json_is_trainable_by_default():
    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    sample = _sample(item, '{"name": "search", "arguments": {"q": "x"}}', [10, 11])

    sample.response_kind = "assistant_tool_call"

    assert session._response_loss_mask(sample) == [True, True]


def test_tool_call_json_can_be_masked_when_disabled():
    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy(assistant_tool_calls=False))
    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    sample = _sample(item, '{"name": "search", "arguments": {"q": "x"}}', [10, 11])

    sample.response_kind = "assistant_tool_call"

    assert session._response_loss_mask(sample) == [False, False]


def test_get_train_batch_trains_tool_call_tokens_by_default():
    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{"board": "b"}], prompts=["p"], input_tokens=[[1, 2]], n_samples=1).iter_samples())
    sample = _sample(item, '{"name": "move", "arguments": {"direction": "left"}}', [10, 11, 12])
    sample.response_kind = "assistant_tool_call"
    session._agent_items = [item]
    session._samples = [sample]

    batch = asyncio.run(session.get_train_batch(reward_fn=lambda record: 0.5))

    assert batch.token_rows == [[1, 2, 10, 11, 12]]
    assert batch.response_masks == [[False, False, True, True, True]]
    assert batch.loss_masks == [[False, False, True, True, True]]
    assert batch.rollout_logprobs == [[0.0, 0.0, 0.0, 0.0, 0.0]]
    assert batch.rewards == [0.5]
    assert batch.reward_records[0].loss_mask == [True, True, True]


def test_tool_call_loss_mask_stops_before_tool_response_sentinel():
    tokenizer = _PieceTokenizer(["<|tool_call>", "call:choose_square", "{square:5}", "<tool_call|>", "<|tool_response>", "<eos>"])

    mask = agentic._tool_call_loss_mask(tokenizer, [0, 1, 2, 3, 4, 5])

    assert mask == [True, True, True, True, False, False]


def test_tool_call_loss_mask_decodes_response_once():
    pieces = ["x"] * 128 + ["<|tool_response>", "ignored"]
    tokenizer = _PieceTokenizer(pieces)

    mask = agentic._tool_call_loss_mask(tokenizer, list(range(len(pieces))))

    assert mask == [True] * 128 + [False, False]
    assert tokenizer.decode_calls == 1


def test_get_train_batch_keeps_assistant_text_tokens_trainable():
    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    sample = _sample(item, "final answer", [20, 21])
    session._agent_items = [item]
    session._samples = [sample]

    batch = asyncio.run(session.get_train_batch())

    assert batch.response_masks == [[False, True, True]]
    assert batch.loss_masks == [[False, True, True]]
    assert batch.reward_records[0].loss_mask == [True, True]


def test_claim_item_reuses_source_record_for_extra_agent_requests():
    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy())
    batch = AgentBatch(records=[{"board": "b"}], prompts=["same prompt"], input_tokens=[[1]], n_samples=1)
    session.attach_batch(batch)
    messages = [{"role": "user", "content": "same prompt"}]

    first = session._claim_item(messages)
    second = session._claim_item(messages)

    assert first.prompt_index == 0
    assert second.prompt_index == 0
    assert second.sample_index == 0
    assert second.record == {"board": "b"}


def test_claim_item_matches_prompt_before_sequential_order():
    """Out-of-order concurrent HTTP arrivals should not attach to the wrong source record."""

    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy())
    batch = AgentBatch(
        records=[{"task": "a"}, {"task": "b"}],
        prompts=["prompt a", "prompt b"],
        input_tokens=[[1], [2]],
        n_samples=1,
    )
    session.attach_batch(batch)

    item = session._claim_item([{"role": "user", "content": "prompt b"}])

    assert item.prompt == "prompt b"
    assert item.record == {"task": "b"}
    assert item.prompt_index == 1


def test_messages_to_prompt_tokens_passes_tools_to_chat_template():
    tokenizer = _ToolAwareTokenizer()
    messages = [{"role": "user", "content": "choose"}]
    tools = [{"type": "function", "function": {"name": "choose_square"}}]

    tokens = agentic._messages_to_prompt_tokens(tokenizer, messages, tools=tools, fallback_prompt="fallback")

    assert tokens == [1, 1]
    assert tokenizer.calls == [(messages, tools)]


def test_normalize_messages_rewrites_null_tool_call_content_for_templates():
    tokenizer = _StrictContentTokenizer()
    messages = agentic._normalize_messages(
        [
            {"role": "user", "content": "choose"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "choose_square", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
        ]
    )

    tokens = agentic._messages_to_prompt_tokens(tokenizer, messages, tools=[], fallback_prompt="fallback")

    assert tokens == [3]
    assert messages[1]["content"] == ""


def test_messages_to_prompt_tokens_falls_back_when_template_rejects_tools():
    tokenizer = _ToolRejectingTokenizer()
    messages = [{"role": "user", "content": "choose"}]
    tools = [{"type": "function", "function": {"name": "choose_square"}}]

    tokens = agentic._messages_to_prompt_tokens(tokenizer, messages, tools=tools, fallback_prompt="fallback")

    assert tokens == [1]
    assert tokenizer.calls == [messages]


def test_proxy_keeps_max_running_prompts_global_across_dp():
    trainer = _FakeTrainer(world_size=8, tp_size=1)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=64,
    )

    assert session.max_running_prompts == 64
    assert session._local_max_running_prompts == 8


def test_proxy_rollout_uses_async_trainer_entry():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=4,
    )
    params = _FakeSamplingParams()
    pending = _pending_chat(1, params)

    asyncio.run(session._run_chat_rollout_async(pending))

    assert trainer.rollout_batches == [([[1]], 1)]
    assert pending.event.is_set()
    assert pending.response is not None
    assert len(session._samples) == 1


def test_rollout_session_context_owns_backend_lifecycle():
    trainer = _FakeTrainer(world_size=1, tp_size=1)

    async def run_session():
        session = RolloutSession(
            trainer,
            sampling_params=_FakeSamplingParams(),
            loss_mask_policy=LossMaskPolicy(),
            max_running_prompts=4,
        )
        session._http_server_cls = lambda: _FakeHTTPServer
        async with session:
            pending = _pending_chat(1, _FakeSamplingParams())
            await session._run_chat_rollout_async(pending)

    asyncio.run(run_session())

    assert trainer.rollout_session_events == ["begin", "end"]
    assert trainer.rollout_batches == [([[1]], 1)]


def test_proxy_filters_prompt_exceeding_max_sequence_len_without_rollout():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _FixedTokenizer(list(range(10)))
    params = _FakeSamplingParams()
    params.max_prompt_len = 5
    params.max_new_tokens = 4
    session = RolloutSession(
        trainer,
        sampling_params=params,
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=4,
    )
    batch = AgentBatch(records=[{"task": "long"}], prompts=["prompt"], input_tokens=[[1]], n_samples=1)
    session.attach_batch(batch)

    response = session._complete_chat({"model": "policy", "messages": [{"role": "user", "content": "long prompt"}]})
    train_batch = asyncio.run(session.get_train_batch())

    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["max_sequence_len"] == 9
    assert trainer.rollout_batches == []
    assert train_batch.token_rows == []


def test_proxy_server_backlog_tracks_max_running_prompts():
    trainer = _FakeTrainer()
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=256,
    )

    assert session._http_server_cls().request_queue_size >= 256


def test_agentic_partial_with_logprobs_completes_http_request():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=4,
    )
    params = _FakeSamplingParams()
    pending = _pending_chat(0, params)

    session._record_chat_sample(pending, [1, 2], [-0.3, -0.4])

    assert pending.event.is_set()
    assert pending.response is not None
    assert len(session._samples) == 1
    assert session._samples[0].response_logprobs == [-0.3, -0.4]


def test_agentic_multi_turn_calls_merge_into_one_training_sample():
    """Multiple model calls for one agent item should form one trajectory sample."""

    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=4,
    )
    item = agentic.AgentItem(record={"task": "multi"}, prompt="p", input_tokens=[1, 2], prompt_index=0, sample_index=0)
    first = _pending_chat(0, _FakeSamplingParams())
    first.item = item
    first.input_tokens = [1, 2]
    first.messages = [{"role": "user", "content": "step 1"}]
    second = _pending_chat(0, _FakeSamplingParams())
    second.item = item
    second.input_tokens = [1, 2, 10, 11, 30, 31]
    second.messages = [
        {"role": "user", "content": "step 1"},
        {"role": "assistant", "content": "10 11"},
        {"role": "tool", "content": "tool result"},
        {"role": "user", "content": "step 2"},
    ]

    session._record_chat_sample(first, [10, 11], [-0.1, -0.2])
    session._record_chat_sample(second, [20], [-0.3])
    batch = asyncio.run(session.get_train_batch(require_finished=False))
    record = batch.reward_records[0]

    assert len(session._samples) == 1
    assert batch.token_rows == [[1, 2, 10, 11, 30, 31, 20]]
    assert batch.response_masks == [[False, False, True, True, False, False, True]]
    assert batch.loss_masks == [[False, False, True, True, False, False, True]]
    assert batch.rollout_logprobs == [[0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3]]
    assert record.tokens == [10, 11, 20]
    assert record.tool_results == [{"name": None, "tool_call_id": None, "content": "tool result"}]
    assert record.completion == "10 11\n20"
    assert [event.type for event in record.trace].count("request") == 2
    assert record.source_record == {"task": "multi"}


def test_agentic_reward_record_renders_messages_with_chat_template():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _DisplayTokenizer()
    session = RolloutSession(trainer, sampling_params=_FakeSamplingParams(), loss_mask_policy=LossMaskPolicy())
    item = agentic.AgentItem(record={}, prompt="p", input_tokens=[1], prompt_index=0, sample_index=0)
    sample = _sample(item, "answer", [2])
    sample.messages = [
        {"role": "user", "content": "question"},
        {"role": "tool", "content": "tool result"},
    ]

    record = session._reward_record(sample)

    assert record.completion == "answer"
    assert record.rendered_completion == "user:question|tool:tool result|assistant:answer"


def test_agentic_tool_request_returns_tool_call_and_reward_record():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _LiteralTokenizer('{"name":"choose_move","arguments":{"direction":"left"}}')
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=4,
    )
    params = _FakeSamplingParams()
    pending = _pending_chat(0, params)
    pending.tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_move",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}}},
            },
        }
    ]
    pending.tool_choice = {"type": "function", "function": {"name": "choose_move"}}

    session._record_chat_sample(pending, [1, 2], [-0.3, -0.4])
    response = pending.response
    record = session._reward_record(session._samples[0])

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "choose_move"
    assert '"direction":"left"' in tool_call["function"]["arguments"]
    assert session._samples[0].response_kind == "assistant_tool_call"
    assert record.tool_calls == [{"name": "choose_move", "arguments": '{"direction":"left"}'}]
    assert record.loss_mask == [True, True]


def test_json_tool_call_parser_prefers_explicit_final_direction():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_move",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}}},
            },
        }
    ]

    parsed = JsonToolCallParser().parse(
        "Valid moves are up, down, left, right. I choose left.",
        tools,
        {"type": "function", "function": {"name": "choose_move"}},
    )

    assert len(parsed.tool_calls) == 1
    assert '"direction":"left"' in parsed.tool_calls[0]["function"]["arguments"]


def test_json_tool_call_parser_rejects_plain_reasoning_without_action():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_move",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}}},
            },
        }
    ]

    parsed = JsonToolCallParser().parse(
        "Let's analyze the possible moves before choosing.",
        tools,
        {"type": "function", "function": {"name": "choose_move"}},
    )

    assert parsed.tool_calls == []
    assert parsed.normal_text


def test_qwen_tool_call_parser_supports_chat_completions_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_move",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}}},
            },
        }
    ]

    parsed = QwenToolCallParser().parse(
        '<tool_call>\n{"name":"choose_move","arguments":{"direction":"right"}}\n</tool_call>',
        tools,
        "required",
    )

    assert parsed.normal_text == ""
    assert len(parsed.tool_calls) == 1
    assert '"direction":"right"' in parsed.tool_calls[0]["function"]["arguments"]


def test_gemma4_tool_call_parser_supports_chat_completions_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_move",
                "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}}},
            },
        }
    ]

    parsed = Gemma4ToolCallParser().parse(
        '<|tool_call>call:choose_move{direction:<|"|>left<|"|>}<tool_call|>',
        tools,
        "required",
    )

    assert parsed.normal_text == ""
    assert len(parsed.tool_calls) == 1
    assert '"direction":"left"' in parsed.tool_calls[0]["function"]["arguments"]


def test_minicpm_tool_call_parser_supports_xml_function_calls():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_square",
                "parameters": {"type": "object", "properties": {"square": {"type": "integer", "minimum": 1, "maximum": 9}}},
            },
        }
    ]

    parsed = MiniCPMToolCallParser().parse(
        '<think>Try the function.</think>\n\n<function name="choose_square"><param name="square">2</param></function><|im_end|>',
        tools,
        {"type": "function", "function": {"name": "choose_square"}},
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "choose_square"
    assert '"square":2' in parsed.tool_calls[0]["function"]["arguments"]


def test_minicpm_tool_call_parser_supports_v46_tool_call_blocks():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_square",
                "parameters": {"type": "object", "properties": {"square": {"type": "integer", "minimum": 1, "maximum": 9}}},
            },
        }
    ]

    parsed = MiniCPMToolCallParser().parse(
        "<tool_call>\n"
        "<function=choose_square>\n"
        "<parameter=square>\n"
        "4\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call><|im_end|>",
        tools,
        {"type": "function", "function": {"name": "choose_square"}},
    )

    assert parsed.normal_text == ""
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "choose_square"
    assert '"square":4' in parsed.tool_calls[0]["function"]["arguments"]


def test_tool_call_parser_supports_flat_tool_schema():
    tools = [
        {
            "type": "function",
            "name": "choose_move",
            "parameters": {"type": "object", "properties": {"direction": {"type": "string", "enum": ["left", "right"]}}},
        }
    ]

    parsed = JsonToolCallParser().parse(
        '{"direction":"left"}',
        tools,
        {"type": "function", "function": {"name": "choose_move"}},
    )

    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "choose_move"


def test_json_tool_call_parser_rejects_tool_choice_mismatch():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "submit_bundle",
                "parameters": {"type": "object", "properties": {"item_ids": {"type": "array", "items": {"type": "string"}}}},
            },
        }
    ]

    parsed = JsonToolCallParser().parse(
        '{"name":"check_kit","arguments":{"item_ids":["packable-rain-shell","insulated-bottle-750"]}}',
        tools,
        {"type": "function", "function": {"name": "submit_bundle"}},
    )

    assert parsed.tool_calls == []
    assert parsed.normal_text


def _sample(item, text, tokens):
    return agentic._AgentSample(
        item=item,
        messages=[],
        response_text=text,
        response_tokens=tokens,
        response_logprobs=[0.0] * len(tokens),
        trace=[],
    )


class _FakeSamplingParams:
    greedy = False
    max_new_tokens = 4
    temperature = 0.0
    top_p = 1.0
    top_k = -1
    stop_token_ids = None
    ignore_eos = False
    skip_special_tokens = True
    max_prompt_len = None

    def model_copy(self):
        copied = _FakeSamplingParams()
        copied.greedy = self.greedy
        copied.max_new_tokens = self.max_new_tokens
        copied.temperature = self.temperature
        copied.top_p = self.top_p
        copied.top_k = self.top_k
        copied.stop_token_ids = self.stop_token_ids
        copied.ignore_eos = self.ignore_eos
        copied.skip_special_tokens = self.skip_special_tokens
        copied.max_prompt_len = self.max_prompt_len
        return copied


class _FakeTokenizer:
    def encode(self, text):
        return [len(text)]

    def decode(self, tokens):
        return " ".join(str(token) for token in tokens)


class _LiteralTokenizer(_FakeTokenizer):
    def __init__(self, text):
        self.text = text

    def decode(self, tokens):
        del tokens
        return self.text


class _FixedTokenizer(_FakeTokenizer):
    def __init__(self, tokens):
        self.tokens = list(tokens)

    def encode(self, text):
        del text
        return list(self.tokens)


class _PieceTokenizer(_FakeTokenizer):
    def __init__(self, pieces):
        self.pieces = list(pieces)
        self.decode_calls = 0

    def decode(self, tokens):
        self.decode_calls += 1
        return "".join(self.pieces[token] for token in tokens)

    def encode(self, text):
        tokens = []
        offset = 0
        for idx, piece in enumerate(self.pieces):
            if text.startswith(piece, offset):
                tokens.append(idx)
                offset += len(piece)
            if offset >= len(text):
                break
        return tokens


class _ToolAwareTokenizer(_FakeTokenizer):
    chat_template = "template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tools=None, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        self.calls.append((messages, tools))
        return [len(messages), len(tools or [])]


class _DisplayTokenizer(_FakeTokenizer):
    chat_template = "template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is False
        return "|".join(f"{message['role']}:{message.get('content')}" for message in messages)


class _StrictContentTokenizer(_FakeTokenizer):
    chat_template = "template"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
        del tools
        assert tokenize is True
        assert add_generation_prompt is True
        for message in messages:
            assert isinstance(message.get("content"), str)
        return [len(messages)]


class _ToolRejectingTokenizer(_FakeTokenizer):
    chat_template = "template"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        self.calls.append(messages)
        return [len(messages)]


class _FakeTrainer:
    def __init__(self, *, world_size=2, tp_size=1):
        self.config = SimpleNamespace(world_size=world_size, tp_size=tp_size)
        self.rollout_batches = []
        self.tokenizer = _FakeTokenizer()
        self.rollout_session_events = []

    def get_tokenizer(self):
        return self.tokenizer

    def rollout_token_batch(self, prompt_tokens, n_samples, sampling_params):
        del sampling_params
        self.rollout_batches.append((prompt_tokens, n_samples))
        return [
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        resp_tokens=[100 + idx],
                        resp_logprobs=[-0.1],
                    )
                ]
            )
            for idx, _prompt in enumerate(prompt_tokens)
        ]

    async def rollout_token_batch_async(self, prompt_tokens, n_samples, sampling_params):
        return self.rollout_token_batch(prompt_tokens, n_samples, sampling_params)

    async def begin_rollout_session_async(self):
        self.rollout_session_events.append("begin")

    async def end_rollout_session_async(self):
        self.rollout_session_events.append("end")


class _FakeHTTPServer:
    def __init__(self, server_address, handler_cls):
        del handler_cls
        self.server_address = (server_address[0], 12345)
        self.timeout = None
        self.shutdown_called = False
        self.server_close_called = False

    def serve_forever(self):
        return None

    def shutdown(self):
        self.shutdown_called = True

    def server_close(self):
        self.server_close_called = True


def _pending_chat(idx, params):
    return agentic._PendingChat(
        item=agentic.AgentItem(record={}, prompt=f"p{idx}", input_tokens=[idx], prompt_index=idx, sample_index=0),
        messages=[{"role": "user", "content": f"prompt-{idx}"}],
        input_tokens=[idx],
        params=params,
        key=agentic._chat_batch_key(params),
        model="policy",
        created_at=time.monotonic(),
    )
