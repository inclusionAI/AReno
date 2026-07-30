"""CPU tests for agentic overlength (safe-stop) handling.

These tests reuse the fakes from ``test_agentic_cpu`` (no GPU, no network) and
exercise the three termination-reason classifications, the ``off`` vs
``safe-stop`` policy split, per-reason metric emission, multi-turn
``termination_reason`` propagation, and the exact-limit boundary. GPU /
distributed rollout behavior is isolated behind the ``_FakeTrainer`` fakes; the
remaining GPU validation is documented in the troubleshooting page.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure the sibling test module (which owns the shared fakes) is importable
# when this file is collected standalone.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the shared fakes/fixtures from the existing agentic CPU suite.
from test_agentic_cpu import (  # noqa: F401  (re-exports for test ergonomics), E402
    _FakeSamplingParams,
    _FakeTokenizer,
    _FakeTrainer,
    _FixedTokenizer,
    _LiteralTokenizer,
    _pending_chat,
    agentic,  # noqa: E402  (the module loaded by the existing suite)
)

from areno.api.agentic import AgentItem, AgentTrajectoryTurn, LossMaskPolicy, RolloutSession  # noqa: E402
from areno.api.metrics import init_rollout_stats, merge_overlength_counts, record_training_stats  # noqa: E402
from areno.api.trainer_config import TrainerConfig  # noqa: E402
from areno.api.trainers.policy_only import _aggregate_overlength_counts  # noqa: E402


class _LengthFakeTrainer(_FakeTrainer):
    """Fake trainer whose rollout sequences carry an engine ``finish_reason``."""

    def __init__(self, *, finish_reason="length", resp_tokens=None, **kwargs):
        super().__init__(**kwargs)
        self._finish_reason = finish_reason
        self._resp_tokens = list(resp_tokens or [101, 102])

    def rollout_token_batch(self, prompt_tokens, n_samples, sampling_params):
        del sampling_params
        self.rollout_batches.append((prompt_tokens, n_samples))
        return [
            SimpleNamespace(
                sequences=[
                    SimpleNamespace(
                        resp_tokens=list(self._resp_tokens),
                        resp_logprobs=[-0.1] * len(self._resp_tokens),
                        finish_reason=self._finish_reason,
                    )
                ]
            )
            for _ in prompt_tokens
        ]


class _CharTokenizer(_FakeTokenizer):
    """Character-level tokenizer: ``len(encode(text)) == len(text)``."""

    def encode(self, text):
        return list(range(len(text)))

    def decode(self, tokens):
        return "x" * len(tokens)


def _session(trainer, *, params=None):
    return RolloutSession(
        trainer,
        sampling_params=params or _FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
    )


def _complete(session, body):
    async def run():
        session._loop = asyncio.get_running_loop()
        return await session._complete_chat(body)

    return asyncio.run(run())


def _item(idx=0):
    return AgentItem(record={}, prompt=f"p{idx}", input_tokens=[idx], prompt_index=idx, sample_index=0)


def _sample_from_response(session, response, *, messages=None, item=None):
    turn = AgentTrajectoryTurn(
        item=item or _item(),
        messages=messages or [{"role": "user", "content": "go"}],
        response=response,
        tools=[],
    )
    return session._sample_from_trajectory_turn(turn)


# ---------------------------------------------------------------------------
# 1. generation_limit + safe-stop drops the half tool call
# ---------------------------------------------------------------------------


def test_overlength_generation_limit_safe_stop_drops_half_tool_call():
    trainer = _LengthFakeTrainer(finish_reason="length", resp_tokens=[101, 102, 103])
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    trainer.tokenizer = _LiteralTokenizer('{"name":"foo","arguments":')
    session = _session(trainer)

    response = _complete(session, {"model": "policy", "messages": [{"role": "user", "content": "go"}]})

    assert response["choices"][0]["finish_reason"] == "length"
    assert response["areno"]["termination_reason"] == "generation_limit"
    assert response["choices"][0]["message"].get("tool_calls") in (None, [])
    assert trainer.rollout_batches == [([[2]], 1)]  # one rollout call only

    sample = _sample_from_response(session, response)
    assert sample.termination_reason == "generation_limit"
    finish_events = [event for event in sample.trace if event.type == "finish"]
    assert finish_events and finish_events[-1].metadata["finish_reason"] == "length"
    assert finish_events[-1].metadata["termination_reason"] == "generation_limit"


# ---------------------------------------------------------------------------
# 2. context_limit (pre-generation short-circuit)
# ---------------------------------------------------------------------------


def test_overlength_context_limit_pre_generation_marks_terminal():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    trainer.tokenizer = _FixedTokenizer(list(range(10)))  # 10 tokens regardless of text
    params = _FakeSamplingParams()
    params.max_context_len = 5
    session = _session(trainer, params=params)

    response = _complete(session, {"model": "policy", "messages": [{"role": "user", "content": "long prompt"}]})

    assert trainer.rollout_batches == []  # pre-generation short-circuit, no rollout
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["areno"]["termination_reason"] == "context_limit"
    assert response["areno"]["response_tokens"] == []


# ---------------------------------------------------------------------------
# 3. oversized_tool_result classified separately from context_limit
# ---------------------------------------------------------------------------


def _tool_messages(tool_content):
    return [
        {"role": "user", "content": "a" * 15},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "f", "content": tool_content},
    ]


def test_overlength_oversized_tool_result_classified_separately():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    trainer.tokenizer = _CharTokenizer()
    params = _FakeSamplingParams()
    params.max_context_len = 10
    session = _session(trainer, params=params)

    response = _complete(session, {"model": "policy", "messages": _tool_messages("b" * 15)})

    assert response["areno"]["termination_reason"] == "oversized_tool_result"
    assert trainer.rollout_batches == []


def test_overlength_context_limit_when_last_tool_result_is_small():
    """Total overlong but the last tool result fits -> context_limit, not oversized."""

    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    trainer.tokenizer = _CharTokenizer()
    params = _FakeSamplingParams()
    params.max_context_len = 10
    session = _session(trainer, params=params)

    # Full trajectory text > 10 (user content alone is 15), but the tool result
    # is only 3 tokens -> context_limit.
    response = _complete(session, {"model": "policy", "messages": _tool_messages("b" * 3)})

    assert response["areno"]["termination_reason"] == "context_limit"


# ---------------------------------------------------------------------------
# 4. generation_limit uses engine finish_reason, not the length heuristic
# ---------------------------------------------------------------------------


def test_overlength_generation_limit_reason_from_engine_finish_reason():
    # finish_reason="length" but len(resp_tokens) < max_new_tokens: the engine
    # signal must win over the length heuristic.
    trainer = _LengthFakeTrainer(finish_reason="length", resp_tokens=[1])
    trainer.config = SimpleNamespace(agent_overlength_policy="off", world_size=1, tp_size=1)
    trainer.tokenizer = _LiteralTokenizer("partial")
    params = _FakeSamplingParams()
    params.max_new_tokens = 4  # len(resp_tokens)=1 < 4 -> heuristic alone would say "not length"
    session = _session(trainer, params=params)

    response = _complete(session, {"model": "policy", "messages": [{"role": "user", "content": "go"}]})

    assert response["areno"]["termination_reason"] == "generation_limit"
    assert trainer.rollout_batches == [([[2]], 1)]


# ---------------------------------------------------------------------------
# 5. policy=off preserves current behavior (tool_calls kept, observability set)
# ---------------------------------------------------------------------------


_FOO_TOOL = {"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}
_FOO_CALL_JSON = '{"name":"foo","arguments":{"x":1}}'


def _generation_limit_response(policy):
    trainer = _LengthFakeTrainer(finish_reason="length", resp_tokens=[1, 2])
    trainer.config = SimpleNamespace(agent_overlength_policy=policy, world_size=1, tp_size=1)
    trainer.tokenizer = _LiteralTokenizer(_FOO_CALL_JSON)
    session = _session(trainer)
    response = _complete(
        session,
        {"model": "policy", "messages": [{"role": "user", "content": "go"}], "tools": [_FOO_TOOL]},
    )
    return response, trainer


def test_overlength_policy_off_preserves_current_behavior():
    response, trainer = _generation_limit_response("off")

    message = response["choices"][0]["message"]
    assert message.get("tool_calls")  # parsed tool call is preserved
    assert len(message["tool_calls"]) == 1
    # OpenAI finish_reason follows the normal tool_calls/stop derivation.
    assert response["choices"][0]["finish_reason"] == "tool_calls"
    # termination_reason is still observable in metadata.
    assert response["areno"]["termination_reason"] == "generation_limit"
    assert trainer.rollout_batches == [([[2]], 1)]


def test_overlength_policy_safe_stop_drops_tool_call_that_off_keeps():
    response, _ = _generation_limit_response("safe-stop")

    assert response["choices"][0]["message"].get("tool_calls") in (None, [])
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["areno"]["termination_reason"] == "generation_limit"


# ---------------------------------------------------------------------------
# 6. invalid policy values rejected
# ---------------------------------------------------------------------------


def test_overlength_policy_invalid_value_rejected():
    assert TrainerConfig(algo="gspo", ckpt="x", dataset_path="x").agent_overlength_policy == "off"
    with pytest.raises(ValueError, match="agent_overlength_policy must be one of"):
        TrainerConfig(algo="gspo", ckpt="x", dataset_path="x", agent_overlength_policy="bogus")


def test_overlength_policy_invalid_cli_value_rejected():
    from click.testing import CliRunner

    from areno.cli.train import train_command

    result = CliRunner().invoke(train_command, ["--agent-overlength-policy", "bogus"])
    assert result.exit_code != 0
    assert "agent-overlength-policy" in result.output


# ---------------------------------------------------------------------------
# 7. per-reason metrics emitted (proxy -> sample -> aggregate -> writer)
# ---------------------------------------------------------------------------


class _MockWriter:
    def __init__(self):
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, float(value), int(step)))

    def flush(self):
        pass


def _reason_sample(session, response):
    return _sample_from_response(session, response)


def test_overlength_per_reason_metrics_emitted():
    # generation_limit sample (safe-stop path)
    gen_trainer = _LengthFakeTrainer(finish_reason="length", resp_tokens=[1, 2])
    gen_trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    gen_trainer.tokenizer = _LiteralTokenizer("partial")
    gen_response = _complete(_session(gen_trainer), {"model": "policy", "messages": [{"role": "user", "content": "go"}]})
    gen_sample = _reason_sample(_session(gen_trainer), gen_response)

    # context_limit sample
    ctx_trainer = _FakeTrainer(world_size=1, tp_size=1)
    ctx_trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    ctx_trainer.tokenizer = _FixedTokenizer(list(range(10)))
    ctx_params = _FakeSamplingParams()
    ctx_params.max_context_len = 5
    ctx_response = _complete(_session(ctx_trainer, params=ctx_params), {"model": "policy", "messages": [{"role": "user", "content": "long"}]})
    # Rebuild the turn against a session bound to the ctx trainer so the sample
    # is materialized with its tokenizer/context.
    ctx_sample = _sample_from_response(_session(ctx_trainer, params=ctx_params), ctx_response)

    # oversized_tool_result sample
    over_trainer = _FakeTrainer(world_size=1, tp_size=1)
    over_trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    over_trainer.tokenizer = _CharTokenizer()
    over_params = _FakeSamplingParams()
    over_params.max_context_len = 10
    over_response = _complete(_session(over_trainer, params=over_params), {"model": "policy", "messages": _tool_messages("b" * 15)})
    over_sample = _sample_from_response(_session(over_trainer, params=over_params), over_response)

    samples = [gen_sample, ctx_sample, over_sample]
    counts = _aggregate_overlength_counts(samples)
    assert counts == {"generation_limit": 1, "context_limit": 1, "oversized_tool_result": 1}

    writer = _MockWriter()
    stats = merge_overlength_counts(init_rollout_stats(), counts)
    record_training_stats(writer, stats, step=0, train_res={}, train_batch=[])

    overlength_scalars = [name for name, _value, _step in writer.scalars if name.startswith("rollout/overlength_")]
    assert "rollout/overlength_generation_limit" in overlength_scalars
    assert "rollout/overlength_context_limit" in overlength_scalars
    assert "rollout/overlength_oversized_tool_result" in overlength_scalars
    assert "rollout/overlength_total" in overlength_scalars
    assert len(overlength_scalars) == 4


# ---------------------------------------------------------------------------
# 8. multi-turn propagates the last termination reason
# ---------------------------------------------------------------------------


def test_overlength_multi_turn_propagates_last_reason():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    session = _session(trainer)
    params = _FakeSamplingParams()
    item = _item()

    first = _pending_chat(0, params)
    first.item = item
    first.input_tokens = [1, 2]
    first.messages = [{"role": "user", "content": "step 1"}]
    sample_one = session._sample_from_pending_chat(
        first, agentic._ResponseData(response_tokens=[10], response_logprobs=[-0.1]),
    )

    second = _pending_chat(0, params)
    second.item = item
    second.input_tokens = [1, 2, 10]
    second.messages = [{"role": "user", "content": "step 2"}]
    sample_two = session._sample_from_pending_chat(
        second, agentic._ResponseData(response_tokens=[20], response_logprobs=[-0.2]),
        termination_reason="generation_limit",
    )

    session._append_sample_response(sample_one, sample_two)

    assert sample_one.termination_reason == "generation_limit"
    finish_events = [event for event in sample_one.trace if event.type == "finish"]
    assert finish_events and finish_events[-1].metadata["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# 9. exact max_context_len boundary does not trigger
# ---------------------------------------------------------------------------


def test_overlength_exact_max_context_len_not_triggered():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    trainer.tokenizer = _FixedTokenizer(list(range(10)))  # 10 tokens
    params = _FakeSamplingParams()
    params.max_context_len = 10  # equal, not greater -> rollout proceeds
    session = _session(trainer, params=params)

    response = _complete(session, {"model": "policy", "messages": [{"role": "user", "content": "edge"}]})

    assert trainer.rollout_batches == [([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 1)]  # rollout called
    assert "termination_reason" not in response["areno"]  # no overlength classification
    assert response["choices"][0]["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 10. regression: Trainer without a .config attribute must not raise
#     (the real Trainer does not expose TrainerConfig; agent_overlength_policy
#      is threaded in via rollout_session, not read off trainer.config)
# ---------------------------------------------------------------------------


def test_overlength_policy_trainer_without_config_attr_defaults_off():
    """A trainer with no ``config`` attribute (like the real ``Trainer``) must
    not raise ``AttributeError`` when the overlength policy is read -- the prior
    implementation reached for ``self._trainer.config`` unguarded."""

    trainer = _LengthFakeTrainer(finish_reason="length", resp_tokens=[101, 102, 103], world_size=1, tp_size=1)
    del trainer.config  # mimic the real Trainer, which has no .config
    trainer.tokenizer = _LiteralTokenizer('{"name":"foo","arguments":')
    session = _session(trainer)  # no agent_overlength_policy threaded -> off

    # _agent_overlength_policy() must not raise and must report "off".
    assert session._agent_overlength_policy() == "off"

    # A generation-limit rollout under "off" keeps the parsed tool call and does
    # not raise mid-rollout (this is the 500 the bug produced in production).
    response = _complete(session, {"model": "policy", "messages": [{"role": "user", "content": "go"}]})
    assert response["areno"]["termination_reason"] == "generation_limit"
    assert response["choices"][0]["finish_reason"] == "stop"  # off -> normal derivation, not "length"


def test_overlength_policy_threaded_safe_stop_overrides_missing_config():
    """When the policy is threaded in explicitly (the real agentic trainer
    path), ``safe-stop`` takes effect even though the trainer has no ``.config``.
    """

    trainer = _LengthFakeTrainer(finish_reason="length", resp_tokens=[101, 102, 103], world_size=1, tp_size=1)
    del trainer.config
    trainer.tokenizer = _LiteralTokenizer('{"name":"foo","arguments":')
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
        agent_overlength_policy="safe-stop",
    )

    assert session._agent_overlength_policy() == "safe-stop"

    response = _complete(session, {"model": "policy", "messages": [{"role": "user", "content": "go"}]})
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["areno"]["termination_reason"] == "generation_limit"
    assert response["choices"][0]["message"].get("tool_calls") in (None, [])  # half tool call dropped


# ---------------------------------------------------------------------------
# 11. an oversized tool result that is NOT the last tool message is not missed
# ---------------------------------------------------------------------------


def test_overlength_earlier_oversized_tool_not_missed():
    """An oversized tool result that is *not* the last tool message must still
    classify as ``oversized_tool_result``. The prior short-circuiting classifier
    only inspected the last tool message and would have mislabelled this
    ``context_limit``."""
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    trainer.tokenizer = _CharTokenizer()
    params = _FakeSamplingParams()
    params.max_context_len = 10
    session = _session(trainer, params=params)

    messages = [
        {"role": "user", "content": "a" * 15},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "f", "content": "b" * 20},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "g", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_2", "name": "g", "content": "b" * 3},
    ]

    response = _complete(session, {"model": "policy", "messages": messages})

    assert response["areno"]["termination_reason"] == "oversized_tool_result"
    assert trainer.rollout_batches == []  # pre-generation short-circuit, no rollout


# ---------------------------------------------------------------------------
# 12. single_message_token_count is template-aware (口径 aligned with the total)
# ---------------------------------------------------------------------------


class _TemplateTokenizer(_FakeTokenizer):
    """A tokenizer with a chat template: wraps each message with role-tag
    markers so the rendered count exceeds the raw content length."""

    chat_template = "fake"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        ids = [100]  # opening role-tag marker
        for message in messages:
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            ids.extend(range(len(text)))
        ids.append(101)  # closing role-tag marker
        return ids

    def encode(self, text):
        return list(range(len(text)))


def test_single_message_token_count_template_aware():
    from areno.api.openai_chat import single_message_token_count

    message = {"role": "tool", "tool_call_id": "c1", "name": "f", "content": "hello"}
    # Template-bearing tokenizer wraps the content with role-tag markers, so the
    # count exceeds the raw content length -- same scale the full prompt render
    # uses, not raw ``encode``.
    assert single_message_token_count(_TemplateTokenizer(), message) == len("hello") + 2
    # A tokenizer without a chat template falls back to raw content encode.
    assert single_message_token_count(_CharTokenizer(), message) == len("hello")


# ---------------------------------------------------------------------------
# 13. filter diagnostics identify the oversized tool result (name + tokens)
# ---------------------------------------------------------------------------


def _policy_only_for_diagnostics():
    from areno.api.trainers.policy_only import PolicyOnlyTrainer

    return PolicyOnlyTrainer(
        config=SimpleNamespace(), instance=object(), dataset=None, reward_fn=None, loss_fn=None
    )


def test_agent_filter_detail_includes_oversized_tool():
    policy = _policy_only_for_diagnostics()
    tokenizer = _CharTokenizer()

    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.config = SimpleNamespace(agent_overlength_policy="safe-stop", world_size=1, tp_size=1)
    trainer.tokenizer = tokenizer
    params = _FakeSamplingParams()
    params.max_context_len = 10
    session = _session(trainer, params=params)

    over_messages = _tool_messages("b" * 25)
    over_response = _complete(session, {"model": "policy", "messages": over_messages})
    over_sample = _sample_from_response(session, over_response, messages=over_messages)
    assert over_sample.termination_reason == "oversized_tool_result"

    over_detail = policy._agent_sample_filter_detail(over_sample, token_len=999, tokenizer=tokenizer)
    assert over_detail["oversized_tool"] == {"name": "f", "tokens": 25}

    ctx_messages = [{"role": "user", "content": "a" * 15}]
    ctx_response = _complete(session, {"model": "policy", "messages": ctx_messages})
    ctx_sample = _sample_from_response(session, ctx_response, messages=ctx_messages)
    assert ctx_sample.termination_reason == "context_limit"
    ctx_detail = policy._agent_sample_filter_detail(ctx_sample, token_len=999, tokenizer=tokenizer)
    assert ctx_detail["oversized_tool"] is None

    diag = {
        "max_context_len": 10,
        "total": 2,
        "kept": 0,
        "filtered": 2,
        "min_tokens": 999,
        "p50_tokens": 999,
        "p90_tokens": 999,
        "max_tokens": 999,
        "top": [over_detail, ctx_detail],
    }
    formatted = policy._format_agent_filter_diagnostics(diag)
    assert "oversized_tool=name:f tokens:25" in formatted
    assert "oversized_tool=name:none" not in formatted  # context_limit entry carries no suffix
