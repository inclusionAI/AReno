import asyncio
import importlib.util
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from areno.api.tool_call_parser import (
    Gemma4ToolCallParser,
    JsonToolCallParser,
    MiniCPMToolCallParser,
    QwenToolCallParser,
)
from areno.api.trainers.policy_only import PolicyOnlyTrainer


def _load_agentic_module():
    path = Path(__file__).resolve().parents[1] / "areno" / "api" / "agentic.py"
    spec = importlib.util.spec_from_file_location("agentic_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agentic = _load_agentic_module()
AgentBatch = agentic.AgentBatch
AgentTrajectoryTurn = agentic.AgentTrajectoryTurn
LossMaskPolicy = agentic.LossMaskPolicy
RolloutSession = agentic.RolloutSession


def test_agent_batch_expands_records_by_n_samples():
    batch = AgentBatch(
        records=[{"prompt": "p0"}, {"prompt": "p1"}], prompts=["p0", "p1"], input_tokens=[[1], [2]], n_samples=2
    )

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


def test_agentic_train_rows_train_tool_call_tokens_by_default():
    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{"board": "b"}], prompts=["p"], input_tokens=[[1, 2]], n_samples=1).iter_samples())
    sample = _sample(item, '{"name": "move", "arguments": {"direction": "left"}}', [10, 11, 12])
    sample.response_kind = "assistant_tool_call"

    rows = session._train_rows_from_samples([sample])
    record = session.reward_record(sample)

    assert rows.token_rows == [[1, 2, 10, 11, 12]]
    assert rows.response_masks == [[False, False, True, True, True]]
    assert rows.loss_masks == [[False, False, True, True, True]]
    assert rows.rollout_logprobs == [[0.0, 0.0, 0.0, 0.0, 0.0]]
    assert record.loss_mask == [True, True, True]


def test_tool_call_loss_mask_stops_before_tool_response_sentinel():
    tokenizer = _PieceTokenizer(
        ["<|tool_call>", "call:choose_square", "{square:5}", "<tool_call|>", "<|tool_response>", "<eos>"]
    )

    mask = agentic._tool_call_loss_mask(tokenizer, [0, 1, 2, 3, 4, 5])

    assert mask == [True, True, True, True, False, False]


def test_tool_call_loss_mask_decodes_response_once():
    pieces = ["x"] * 128 + ["<|tool_response>", "ignored"]
    tokenizer = _PieceTokenizer(pieces)

    mask = agentic._tool_call_loss_mask(tokenizer, list(range(len(pieces))))

    assert mask == [True] * 128 + [False, False]
    assert tokenizer.decode_calls == 1


def test_agentic_train_rows_keep_assistant_text_tokens_trainable():
    session = RolloutSession(None, sampling_params=None, loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    sample = _sample(item, "final answer", [20, 21])

    rows = session._train_rows_from_samples([sample])
    record = session.reward_record(sample)

    assert rows.response_masks == [[False, True, True]]
    assert rows.loss_masks == [[False, True, True]]
    assert record.loss_mask == [True, True]


def test_agent_trajectory_turn_extracts_response_metadata():
    item = next(
        AgentBatch(records=[{"task": "same"}], prompts=["same prompt"], input_tokens=[[1]], n_samples=1).iter_samples()
    )
    turn = AgentTrajectoryTurn(
        item=item,
        messages=[{"role": "user", "content": "same prompt"}],
        response={"areno": {"response_tokens": [10, 11], "response_logprobs": [-0.1, -0.2]}},
    )

    assert turn.item.record == {"task": "same"}
    assert turn.response_tokens == [10, 11]
    assert turn.response_logprobs == [-0.1, -0.2]


def test_agent_trajectory_uses_proxy_parsed_tool_calls_only():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _LiteralTokenizer('<tool_call>{"name":"submit","arguments":{"status":"solved"}}</tool_call>')
    session = RolloutSession(trainer, sampling_params=_FakeSamplingParams(), loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '<tool_call>{"name":"submit","arguments":{"status":"solved"}}</tool_call>',
                }
            }
        ],
        "areno": {"response_tokens": [10, 11], "response_logprobs": [-0.1, -0.2]},
    }

    turn = AgentTrajectoryTurn(item=item, messages=[{"role": "user", "content": "p"}], response=response)
    sample = session._sample_from_trajectory_turn(turn)
    record = session.reward_record(sample)

    assert turn.parsed_tool_calls == []
    assert sample.response_kind == "assistant_text"
    assert record.tool_calls == []


def test_agent_trajectory_accepts_proxy_parsed_tool_calls():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _LiteralTokenizer('<tool_call>{"name":"submit","arguments":{"status":"solved"}}</tool_call>')
    session = RolloutSession(trainer, sampling_params=_FakeSamplingParams(), loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-submit",
                            "type": "function",
                            "function": {"name": "submit", "arguments": '{"status":"solved"}'},
                        }
                    ],
                }
            }
        ],
        "areno": {"response_tokens": [10, 11], "response_logprobs": [-0.1, -0.2]},
    }

    turn = AgentTrajectoryTurn(item=item, messages=[{"role": "user", "content": "p"}], response=response)
    sample = session._sample_from_trajectory_turn(turn)
    record = session.reward_record(sample)

    assert sample.response_kind == "assistant_tool_call"
    assert record.tool_calls == [{"name": "submit", "arguments": '{"status":"solved"}'}]


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
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "choose_square", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
        ]
    )

    tokens = agentic._messages_to_prompt_tokens(tokenizer, messages, tools=[], fallback_prompt="fallback")

    assert tokens == [3]
    assert messages[1]["content"] == ""


def test_messages_to_prompt_tokens_normalizes_tool_call_arguments_for_templates():
    tokenizer = _ToolCallArgumentsMappingTokenizer()
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "inspect_tree", "arguments": '{"path":".","max_depth":3}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
    ]

    tokens = agentic._messages_to_prompt_tokens(tokenizer, messages, tools=[], fallback_prompt="fallback")

    assert tokens == [3]


def test_explicit_trajectory_tokenization_normalizes_null_tool_call_content():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _StrictContentTokenizer()
    session = RolloutSession(trainer, sampling_params=_FakeSamplingParams(), loss_mask_policy=LossMaskPolicy())
    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    turn = AgentTrajectoryTurn(
        item=item,
        messages=[
            {"role": "user", "content": "choose"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "choose_square", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
        ],
        response={"areno": {"response_tokens": [1], "response_logprobs": [-0.1]}},
    )

    sample = session._sample_from_trajectory_turn(turn)

    assert sample.token_row == [3, 1]


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


def test_proxy_http_server_allows_large_thread_pool():
    session = RolloutSession(
        _FakeTrainer(),
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=256,
    )
    server_cls = session._http_server_cls()

    assert server_cls.max_threads == 2048
    assert server_cls.request_queue_size >= 2048


def test_proxy_rollout_batches_queued_requests():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=2,
    )

    async def run():
        session._loop = asyncio.get_running_loop()
        return await asyncio.gather(
            session._complete_chat({"model": "policy", "messages": [{"role": "user", "content": "p0"}]}),
            session._complete_chat({"model": "policy", "messages": [{"role": "user", "content": "p1"}]}),
        )

    responses = asyncio.run(run())

    assert trainer.rollout_batches == [([[2]], 1), ([[2]], 1)]
    assert trainer.rollout_sync_count == 0
    assert [response["usage"]["completion_tokens"] for response in responses] == [1, 1]
    assert all(response["areno"]["response_logprobs"] == [-0.1] for response in responses)


def test_explicit_trajectory_helper_builds_train_rows_without_prompt_claiming():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=2,
    )
    batch = AgentBatch(records=[{"answer": "ok"}], prompts=["same prompt"], input_tokens=[[7]], n_samples=1)
    item = next(batch.iter_samples())
    response = {
        "choices": [{"message": {"role": "assistant", "content": "100"}}],
        "areno": {"response_tokens": [100], "response_logprobs": [-0.25]},
    }

    turn = AgentTrajectoryTurn(
        item,
        messages=[{"role": "user", "content": "same prompt"}],
        response=response,
    )
    sample = session._sample_from_trajectory_turn(turn)
    rows = session._train_rows_from_samples([sample])
    record = session.reward_record(sample)

    assert rows.token_rows == [[len("same prompt"), 100]]
    assert rows.response_masks == [[False, True]]
    assert rows.rollout_logprobs == [[0.0, -0.25]]
    assert record.metadata == {"prompt_index": 0, "sample_index": 0}


def test_openai_chat_completion_preserves_proxy_trajectory_metadata():
    from openai.types.chat import ChatCompletion

    item = next(AgentBatch(records=[{}], prompts=["p"], input_tokens=[[1]], n_samples=1).iter_samples())
    response = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "policy",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "100"}, "finish_reason": "stop"}],
            "areno": {"response_tokens": [100], "response_logprobs": [-0.1]},
        }
    )

    turn = AgentTrajectoryTurn(item=item, messages=[{"role": "user", "content": "p"}], response=response)

    assert turn.response_tokens == [100]
    assert turn.response_logprobs == [-0.1]


def test_agentic_overlong_trajectory_is_filtered_with_warning(caplog):
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(trainer, sampling_params=_FakeSamplingParams(), loss_mask_policy=LossMaskPolicy())
    item0 = agentic.AgentItem(record={}, prompt="p0", input_tokens=[1], prompt_index=0, sample_index=0)
    item1 = agentic.AgentItem(record={}, prompt="p1", input_tokens=[2], prompt_index=1, sample_index=0)
    short = _sample(item0, "ok", [10])
    long = _sample(item1, "too long", [20, 21, 22])
    session._set_sample_training_row(short, [1])
    session._set_sample_training_row(long, [2, 3, 4, 5])
    params = _FakeSamplingParams()
    params.max_prompt_len = 2
    params.max_new_tokens = 2
    policy = object.__new__(PolicyOnlyTrainer)
    policy.logger = logging.getLogger("test.agentic.filter")
    policy.areno = SimpleNamespace(model_context_len=lambda: 4)

    with caplog.at_level(logging.WARNING):
        kept, filtered, diagnostics = policy._filter_overlong_agent_samples(session, [short, long], params)

    assert kept == [short]
    assert filtered == 1
    assert diagnostics["max_context_len"] == 4
    assert diagnostics["filtered"] == 1
    assert diagnostics["top"][0]["tokens"] == 7
    assert "agentic trajectory filtered" in caplog.text
    assert "max_context_len=4" in caplog.text


def test_agentic_trajectory_filter_uses_total_context_not_prompt_limit():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(trainer, sampling_params=_FakeSamplingParams(), loss_mask_policy=LossMaskPolicy())
    item = agentic.AgentItem(record={}, prompt="p0", input_tokens=[1], prompt_index=0, sample_index=0)
    sample = _sample(item, "fits total context", [10, 11, 12])
    session._set_sample_training_row(sample, [1, 2, 3, 4, 5, 6, 7])
    params = _FakeSamplingParams()
    params.max_prompt_len = 2
    params.max_new_tokens = 5
    policy = object.__new__(PolicyOnlyTrainer)
    policy.logger = logging.getLogger("test.agentic.filter")
    policy.areno = SimpleNamespace(model_context_len=lambda: 10)

    kept, filtered, diagnostics = policy._filter_overlong_agent_samples(session, [sample], params)

    assert kept == [sample]
    assert filtered == 0
    assert diagnostics["max_context_len"] == 10


def test_agentic_filter_diagnostics_formats_top_trajectories():
    policy = object.__new__(PolicyOnlyTrainer)
    diagnostics = {
        "max_context_len": 8,
        "total": 2,
        "kept": 0,
        "filtered": 2,
        "min_tokens": 12,
        "p50_tokens": 12,
        "p90_tokens": 20,
        "max_tokens": 20,
        "top": [
            {
                "prompt_idx": 1,
                "sample_idx": 0,
                "tokens": 20,
                "messages": 6,
                "assistant_messages": 3,
                "tool_results": 2,
                "response_tokens": 5,
                "trace_events": 4,
                "prompt": "build kit",
            }
        ],
    }

    text = policy._format_agent_filter_diagnostics(diagnostics)

    assert "max_context_len=8" in text
    assert "tokens[min/p50/p90/max]=12/12/20/20" in text
    assert "prompt_idx=1" in text
    assert "tool_results=2" in text


def test_proxy_client_cancellation_does_not_cancel_queued_rollout():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.rollout_delay_s = 0.05
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
    )

    async def run():
        session._loop = asyncio.get_running_loop()
        task = asyncio.create_task(
            session._complete_chat({"model": "policy", "messages": [{"role": "user", "content": "p0"}]})
        )
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.1)

    asyncio.run(run())

    assert trainer.rollout_batches == [([[2]], 1)]


def test_rollout_session_context_owns_backend_lifecycle():
    trainer = _FakeTrainer(world_size=1, tp_size=1)

    async def run_session():
        session = RolloutSession(
            trainer,
            sampling_params=_FakeSamplingParams(),
            loss_mask_policy=LossMaskPolicy(),
            max_running_prompts=4,
            proxy=False,
        )
        async with session:
            pass

    asyncio.run(run_session())

    assert trainer.rollout_session_events == ["begin", "end"]


def test_rollout_session_uses_trainer_effective_dp_size():
    trainer = _FakeTrainer(world_size=8, tp_size=4)
    trainer.effective_dp_size = 4

    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=10,
        proxy=False,
    )

    assert session._dp_size == 4
    assert session._local_max_running_prompts == 3


def test_rollout_session_sync_is_explicit_batch_level_hook():
    trainer = _FakeTrainer(world_size=1, tp_size=1)

    async def run_session():
        session = RolloutSession(
            trainer,
            sampling_params=_FakeSamplingParams(),
            loss_mask_policy=LossMaskPolicy(),
            max_running_prompts=4,
            proxy=False,
        )
        async with session:
            await session.sync_rollout_session_async()

    asyncio.run(run_session())

    assert trainer.rollout_session_events == ["begin", "end"]
    assert trainer.rollout_sync_count == 1


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
        max_running_prompts=1,
    )

    async def run():
        session._loop = asyncio.get_running_loop()
        return await session._complete_chat(
            {"model": "policy", "messages": [{"role": "user", "content": "long prompt"}]}
        )

    response = asyncio.run(run())

    assert response["choices"][0]["finish_reason"] == "length"
    assert response["usage"]["max_sequence_len"] == 5
    assert response["areno"]["response_tokens"] == []
    assert response["areno"]["response_logprobs"] == []
    assert trainer.rollout_batches == []
    assert trainer.rollout_sync_count == 0


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

    response = session._build_chat_response(pending, agentic._ResponseData([1, 2], [-0.3, -0.4]))

    assert response["areno"]["response_tokens"] == [1, 2]
    assert response["areno"]["response_logprobs"] == [-0.3, -0.4]


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

    first_sample = session._sample_from_pending_chat(first, agentic._ResponseData([10, 11], [-0.1, -0.2]))
    second_sample = session._sample_from_pending_chat(second, agentic._ResponseData([20], [-0.3]))
    session._append_sample_response(first_sample, second_sample)
    rows = session._train_rows_from_samples([first_sample])
    record = session.reward_record(first_sample)

    assert rows.token_rows == [[1, 2, 10, 11, 30, 31, 20]]
    assert rows.response_masks == [[False, False, True, True, False, False, True]]
    assert rows.loss_masks == [[False, False, True, True, False, False, True]]
    assert rows.rollout_logprobs == [[0.0, 0.0, -0.1, -0.2, 0.0, 0.0, -0.3]]
    assert record.tokens == [10, 11, 20]
    assert record.tool_results == [{"name": None, "tool_call_id": None, "content": "tool result"}]
    assert record.completion == "10 11\n20"
    assert record.final_answer == "20"
    assert [event.type for event in record.trace].count("request") == 2
    assert [event.type for event in record.trace].count("tool_result") == 1
    assert record.source_record == {"task": "multi"}


def test_agentic_interleaved_trajectories_do_not_cross_items():
    """Interleaved agent turns must only merge with the matching AgentItem."""

    trainer = _FakeTrainer(world_size=1, tp_size=1)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=4,
    )
    items = list(
        AgentBatch(
            records=[{"task": "a"}, {"task": "b"}],
            prompts=["prompt-a", "prompt-b"],
            input_tokens=[[1], [2]],
            n_samples=1,
        ).iter_samples()
    )
    turns = [
        AgentTrajectoryTurn(
            items[0],
            messages=[{"role": "user", "content": "a-1"}],
            response={"areno": {"response_tokens": [10], "response_logprobs": [-0.1]}},
        ),
        AgentTrajectoryTurn(
            items[1],
            messages=[{"role": "user", "content": "b-1"}],
            response={"areno": {"response_tokens": [20], "response_logprobs": [-0.2]}},
        ),
        AgentTrajectoryTurn(
            items[0],
            messages=[
                {"role": "user", "content": "a-1"},
                {"role": "assistant", "content": "10"},
                {"role": "user", "content": "a-2"},
            ],
            response={"areno": {"response_tokens": [11], "response_logprobs": [-0.11]}},
        ),
        AgentTrajectoryTurn(
            items[1],
            messages=[
                {"role": "user", "content": "b-1"},
                {"role": "assistant", "content": "20"},
                {"role": "user", "content": "b-2"},
            ],
            response={"areno": {"response_tokens": [21], "response_logprobs": [-0.21]}},
        ),
    ]
    policy = object.__new__(PolicyOnlyTrainer)
    samples = []

    for turn in turns:
        sample = session._sample_from_trajectory_turn(turn)
        existing = policy._find_agent_sample(samples, sample.item)
        if existing is None:
            samples.append(sample)
        else:
            session._append_sample_response(existing, sample)

    records = [session.reward_record(sample) for sample in samples]

    assert [(sample.item.prompt_index, sample.item.sample_index) for sample in samples] == [(0, 0), (1, 0)]
    assert [sample.response_tokens for sample in samples] == [[10, 11], [20, 21]]
    assert [record.source_record for record in records] == [{"task": "a"}, {"task": "b"}]
    assert [record.final_answer for record in records] == ["11", "21"]
    assert "b-" not in str(records[0].messages)
    assert "a-" not in str(records[1].messages)


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

    record = session.reward_record(sample)

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
                "parameters": {
                    "type": "object",
                    "properties": {"direction": {"type": "string", "enum": ["left", "right"]}},
                },
            },
        }
    ]
    pending.tool_choice = {"type": "function", "function": {"name": "choose_move"}}

    response = session._build_chat_response(pending, agentic._ResponseData([1, 2], [-0.3, -0.4]))
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    tool_call = response["choices"][0]["message"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "choose_move"
    assert '"direction":"left"' in tool_call["function"]["arguments"]

    sample = session._sample_from_pending_chat(
        pending,
        agentic._ResponseData([1, 2], [-0.3, -0.4]),
        tool_calls=response["choices"][0]["message"]["tool_calls"],
    )
    record = session.reward_record(sample)

    assert sample.response_kind == "assistant_tool_call"
    assert record.tool_calls == [{"name": "choose_move", "arguments": '{"direction":"left"}'}]
    assert record.loss_mask == [True, True]


def test_json_tool_call_parser_prefers_explicit_final_direction():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_move",
                "parameters": {
                    "type": "object",
                    "properties": {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
                },
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
                "parameters": {
                    "type": "object",
                    "properties": {"direction": {"type": "string", "enum": ["up", "down", "left", "right"]}},
                },
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
                "parameters": {
                    "type": "object",
                    "properties": {"direction": {"type": "string", "enum": ["left", "right"]}},
                },
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


def test_qwen_tool_call_parser_supports_angle_function_blocks():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "inspect_tree",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "max_depth": {"type": "integer"},
                    },
                },
            },
        }
    ]

    parsed = QwenToolCallParser().parse(
        "Let me start by inspecting the repository structure.\n"
        "</think>\n\n"
        "<tool_call>\n"
        "<function=inspect_tree>\n"
        "<parameter=path>\n"
        ".\n"
        "</parameter>\n"
        "<parameter=max_depth>\n"
        "3\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>",
        tools,
        "required",
    )

    assert parsed.normal_text == "Let me start by inspecting the repository structure.\n</think>"
    assert len(parsed.tool_calls) == 1
    assert parsed.tool_calls[0]["function"]["name"] == "inspect_tree"
    assert '"path":"."' in parsed.tool_calls[0]["function"]["arguments"]
    assert '"max_depth":3' in parsed.tool_calls[0]["function"]["arguments"]


def test_gemma4_tool_call_parser_supports_chat_completions_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_move",
                "parameters": {
                    "type": "object",
                    "properties": {"direction": {"type": "string", "enum": ["left", "right"]}},
                },
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


def test_gemma4_tool_call_parser_supports_nested_action_arrays():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "actions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string"},
                                    "direction": {"type": "string"},
                                },
                            },
                        }
                    },
                },
            },
        }
    ]

    parsed = Gemma4ToolCallParser().parse(
        '<|tool_call>call:choose_action{actions:[{action:<|"|>MOVE<|"|>,direction:<|"|>UP<|"|>},'
        '{action:<|"|>ATTACK<|"|>,direction:<|"|>UP<|"|>}]}<tool_call|>',
        tools,
        "required",
    )

    assert len(parsed.tool_calls) == 1
    assert (
        parsed.tool_calls[0]["function"]["arguments"]
        == '{"actions":[{"action":"MOVE","direction":"UP"},{"action":"ATTACK","direction":"UP"}]}'
    )


def test_minicpm_tool_call_parser_supports_xml_function_calls():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "choose_square",
                "parameters": {
                    "type": "object",
                    "properties": {"square": {"type": "integer", "minimum": 1, "maximum": 9}},
                },
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
                "parameters": {
                    "type": "object",
                    "properties": {"square": {"type": "integer", "minimum": 1, "maximum": 9}},
                },
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
            "parameters": {
                "type": "object",
                "properties": {"direction": {"type": "string", "enum": ["left", "right"]}},
            },
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
                "parameters": {
                    "type": "object",
                    "properties": {"item_ids": {"type": "array", "items": {"type": "string"}}},
                },
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
        last_response_text=text,
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


class _ToolCallArgumentsMappingTokenizer(_StrictContentTokenizer):
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
        for message in messages:
            for call in message.get("tool_calls") or []:
                assert isinstance(call["function"]["arguments"], dict)
                list(call["function"]["arguments"].items())
        return super().apply_chat_template(
            messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt, tools=tools
        )


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
        self.effective_dp_size = max(world_size // tp_size, 1)
        self.rollout_batches = []
        self.tokenizer = _FakeTokenizer()
        self.rollout_session_events = []
        self.rollout_sync_count = 0
        self.rollout_delay_s = 0.0

    def get_tokenizer(self):
        return self.tokenizer

    def dp_size(self):
        return self.effective_dp_size

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
        if self.rollout_delay_s:
            await asyncio.sleep(self.rollout_delay_s)
        return self.rollout_token_batch(prompt_tokens, n_samples, sampling_params)

    async def begin_rollout_session_async(self):
        self.rollout_session_events.append("begin")

    async def sync_rollout_session_async(self):
        self.rollout_sync_count += 1

    async def end_rollout_session_async(self):
        self.rollout_session_events.append("end")


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
