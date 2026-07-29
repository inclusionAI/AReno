import asyncio
import importlib.util
import logging
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Load agentic module under test (same pattern as test_agentic_cpu.py)
# ---------------------------------------------------------------------------
def _load_agentic_module():
    path = Path(__file__).resolve().parents[1] / "areno" / "api" / "agentic.py"
    spec = importlib.util.spec_from_file_location("agentic_ctx_overflow_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agentic = _load_agentic_module()
AgentTrajectoryTurn = agentic.AgentTrajectoryTurn
LossMaskPolicy = agentic.LossMaskPolicy
RolloutSession = agentic.RolloutSession
_trim_messages_to_fit = agentic._trim_messages_to_fit
_filtered_chat_response = agentic._filtered_chat_response
_unfittable_chat_response = agentic._unfittable_chat_response


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------
class _ChatTemplateTokenizer:
    """Tokenizer that uses apply_chat_template to compute token counts,
    simulating a real chat-template tokenizer without torch dependency."""

    def __init__(self, tokens_per_message: int = 50, tools_overhead: int = 20, gen_prompt_overhead: int = 2):
        self._tpm = tokens_per_message
        self._tools_overhead = tools_overhead
        self._gen_overhead = gen_prompt_overhead
        self.chat_template = True  # Exists as attribute (truthy)
        self.last_messages = None
        self.last_tools = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, tools=None):
        assert tokenize is True
        self.last_messages = messages
        self.last_tools = tools
        total = len(messages) * self._tpm
        if tools:
            total += self._tools_overhead
        if add_generation_prompt:
            total += self._gen_overhead
        return list(range(total))

    def encode(self, text):
        return list(range(10))

    def decode(self, tokens, skip_special_tokens=True):
        return "decoded text"


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
    max_context_len = None

    def model_copy(self):
        return self


# ---------------------------------------------------------------------------
# Fake trainer
# ---------------------------------------------------------------------------
class _FakeTrainer:
    def __init__(self, *, world_size=1, tp_size=1):
        from types import SimpleNamespace

        self.config = SimpleNamespace(world_size=world_size, tp_size=tp_size)
        self.effective_dp_size = max(world_size // tp_size, 1)
        self.rollout_batches = []
        self.tokenizer = _ChatTemplateTokenizer()
        self.rollout_session_events = []
        self.rollout_sync_count = 0
        self.rollout_delay_s = 0.0
        self.rollout_token_count = 0

    def get_tokenizer(self):
        return self.tokenizer

    def dp_size(self):
        return self.effective_dp_size

    def rollout_token_batch(self, prompt_tokens, n_samples, sampling_params):
        self.rollout_batches.append((prompt_tokens, n_samples))
        tokens = prompt_tokens[0] if prompt_tokens else []
        resp_count = min(4, len(tokens))
        sequence = type("Seq", (), {"resp_tokens": list(tokens[:resp_count]),
                                     "resp_logprobs": [-0.1] * resp_count})()
        return [type("Rollout", (), {"sequences": [sequence]})()]

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


def _infer_tool_call_parser_name(trainer):
    return "json"


# Patch the register function that the module uses
agentic.infer_tool_call_parser_name = _infer_tool_call_parser_name


# Override get_tool_call_parser to return a simple parser
class _FakeToolCallParser:
    def parse(self, content, tools, tool_choice):
        result = type("ParseResult", (), {"tool_calls": []})()
        return result


agentic.get_tool_call_parser = lambda name: _FakeToolCallParser()


# ---------------------------------------------------------------------------
# Message constructor helpers
# ---------------------------------------------------------------------------
def _sys(content="You are a coder."):
    return {"role": "system", "content": content}


def _dev(content="You are a developer."):
    return {"role": "developer", "content": content}


def _usr(content="Write a sort function."):
    return {"role": "user", "content": content}


def _ast_no_tool(content="Here is the code..."):
    return {"role": "assistant", "content": content}


def _ast_with_tool_calls():
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}}],
    }


def _ast_two_tool_calls():
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "write_file", "arguments": "{}"}},
            {"id": "call-2", "type": "function", "function": {"name": "run_test", "arguments": "{}"}},
        ],
    }


def _tool(call_id="call-1", name="write_file", content="ok"):
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


# ---------------------------------------------------------------------------
# group_messages_into_units tests
# ---------------------------------------------------------------------------
def test_group_messages_into_units_basic():
    from areno.api.openai_chat import group_messages_into_units

    messages = [_sys(), _usr()]
    units = group_messages_into_units(messages)
    assert len(units) == 2
    assert units[0] == [_sys()]
    assert units[1] == [_usr()]


def test_group_messages_into_units_system_preserved():
    from areno.api.openai_chat import group_messages_into_units

    messages = [_sys(), _usr(), _ast_with_tool_calls(), _tool(), _tool(), _usr()]
    units = group_messages_into_units(messages)
    assert len(units) == 4
    assert units[0] == [_sys()]
    assert units[1] == [_usr()]
    assert units[2] == [_ast_with_tool_calls(), _tool(), _tool()]  # atomic
    assert units[3] == [_usr()]


def test_group_messages_into_units_developer_preserved():
    from areno.api.openai_chat import group_messages_into_units

    messages = [_dev(), _usr(), _ast_with_tool_calls(), _tool(), _usr()]
    units = group_messages_into_units(messages)
    assert len(units) == 4


def test_group_messages_into_units_assistant_no_tool_is_own_unit():
    from areno.api.openai_chat import group_messages_into_units

    messages = [_usr(), _ast_no_tool(), _usr()]
    units = group_messages_into_units(messages)
    assert len(units) == 3
    assert units[1] == [_ast_no_tool()]


def test_group_messages_into_units_tool_without_assistant():
    from areno.api.openai_chat import group_messages_into_units

    messages = [_tool()]
    units = group_messages_into_units(messages)
    assert len(units) == 1
    assert units[0] == [_tool()]


def test_group_messages_into_units_empty():
    from areno.api.openai_chat import group_messages_into_units

    units = group_messages_into_units([])
    assert units == []


# ---------------------------------------------------------------------------
# _trim_messages_to_fit tests
# ---------------------------------------------------------------------------
def test_trim_messages_fits_without_trim():
    """No removable units: system + last user = 2 units, no gap."""
    tok = _ChatTemplateTokenizer(tokens_per_message=10, tools_overhead=0, gen_prompt_overhead=0)
    messages = [_sys(), _usr()]
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=100, input_tokens=list(range(20))
    )
    assert result is None


def test_trim_messages_removes_oldest_unit():
    tok = _ChatTemplateTokenizer(tokens_per_message=50, tools_overhead=0, gen_prompt_overhead=0)
    messages = [_sys(), _usr("task 1"), _ast_no_tool("done"), _usr("task 2")]
    # 4 msgs * 50 = 200 tokens; max=120 forces removal
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=120, input_tokens=list(range(300))
    )
    assert result is not None
    # system preserved, latest user preserved → 2 removables: user1 + assistant
    assert result["diagnostics"]["units_removed"] == 2
    assert result["diagnostics"]["messages_removed"] == 2
    assert result["diagnostics"]["preserved_instructions"] is True
    # Only system + latest user remain
    assert len(result["messages"]) == 2


def test_trim_messages_preserves_system():
    tok = _ChatTemplateTokenizer(tokens_per_message=10, tools_overhead=0, gen_prompt_overhead=0)
    messages = [_sys(), _usr("t1"), _ast_no_tool("d1"), _usr("t2"), _ast_no_tool("d2"), _usr("t3")]
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=45, input_tokens=list(range(100))
    )
    assert result is not None
    roles = [msg["role"] for msg in result["messages"]]
    assert roles[0] == "system"


def test_trim_messages_preserves_latest_user_turn():
    tok = _ChatTemplateTokenizer(tokens_per_message=10, tools_overhead=0, gen_prompt_overhead=0)
    messages = [_sys(), _usr("t1"), _ast_no_tool("d1"), _usr("latest")]
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=35, input_tokens=list(range(100))
    )
    assert result is not None
    assert result["messages"][-1]["role"] == "user"
    assert result["messages"][-1]["content"] == "latest"


def test_trim_messages_keeps_tool_call_atomic():
    tok = _ChatTemplateTokenizer(tokens_per_message=5, tools_overhead=5, gen_prompt_overhead=2)
    messages = [
        _sys(),
        _usr("old task"),
        _ast_with_tool_calls(),
        _tool("call-1", "write", "result1"),
        _tool("call-2", "lint", "result2"),
        _usr("new task"),
    ]
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=28, input_tokens=list(range(37))
    )
    assert result is not None
    tool_count = sum(1 for m in result["messages"] if m["role"] == "tool")
    assert tool_count == 2


def test_trim_messages_impossible_to_fit():
    tok = _ChatTemplateTokenizer(tokens_per_message=50, tools_overhead=0, gen_prompt_overhead=50)
    messages = [_sys(), _usr("only one task")]
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=100, input_tokens=list(range(150))
    )
    assert result is None


def test_trim_messages_multiple_tool_calls():
    """If the assistant+tool unit is removed, its tool results go with it."""
    tok = _ChatTemplateTokenizer(tokens_per_message=3, tools_overhead=3, gen_prompt_overhead=1)
    messages = [
        _sys(),
        _usr("old"),
        _ast_two_tool_calls(),
        _tool("call-1", "write", "ok1"),
        _tool("call-2", "test", "ok2"),
        _usr("new"),
    ]
    # 6 * 3 + 3 + 1 = 22; max=12 forces removing both removable units
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=12, input_tokens=list(range(22))
    )
    assert result is not None
    # Only system + latest user remain; tool unit was removed as one atom
    n_tool = sum(1 for m in result["messages"] if m["role"] == "tool")
    assert n_tool == 0


def test_trim_messages_exact_boundary():
    tok = _ChatTemplateTokenizer(tokens_per_message=10, tools_overhead=0, gen_prompt_overhead=0)
    messages = [_sys(), _usr("t1"), _ast_no_tool("d1"), _usr("t2")]
    # 4 * 10 = 40 tokens; removing both t1 and d1 leaves 2 * 10 = 20 (exact boundary)
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=20, input_tokens=list(range(40))
    )
    assert result is not None
    assert result["diagnostics"]["units_removed"] == 2
    assert result["diagnostics"]["messages_removed"] == 2


def test_trim_does_not_mutate_input():
    tok = _ChatTemplateTokenizer(tokens_per_message=10, tools_overhead=0, gen_prompt_overhead=0)
    messages = [_sys(), _usr("t1"), _ast_no_tool("ok"), _usr("t2")]
    original = [dict(m) for m in messages]
    _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=25, input_tokens=list(range(40))
    )
    assert messages == original


def test_trim_messages_with_tools_includes_overhead():
    tok = _ChatTemplateTokenizer(tokens_per_message=10, tools_overhead=30, gen_prompt_overhead=5)
    messages = [_sys(), _usr("t1"), _ast_no_tool("ok"), _usr("t2")]
    # 4 * 10 + 30 + 5 = 75 tokens; removing both removes yields 2 * 10 + 30 + 5 = 55
    # max=60 > 55 so trimming should succeed
    result = _trim_messages_to_fit(
        tok, messages, tools=[{"type": "function"}], max_context_len=60, input_tokens=list(range(75))
    )
    assert result is not None
    assert result["diagnostics"]["units_removed"] == 2


# ---------------------------------------------------------------------------
# RolloutSession integration tests
# ---------------------------------------------------------------------------
def test_reject_policy_is_default():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _ChatTemplateTokenizer(tokens_per_message=100)
    params = _FakeSamplingParams()
    params.max_prompt_len = 10
    session = RolloutSession(
        trainer,
        sampling_params=params,
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
    )

    async def run():
        session._loop = asyncio.get_running_loop()
        return await session._complete_chat(
            {"model": "policy", "messages": [{"role": "user", "content": "long"}]}
        )

    response = asyncio.run(run())
    assert response["choices"][0]["finish_reason"] == "length"
    assert trainer.rollout_batches == []


def test_trim_messages_policy_trims_and_rolls_out():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _ChatTemplateTokenizer(tokens_per_message=50, tools_overhead=0, gen_prompt_overhead=0)
    params = _FakeSamplingParams()
    params.max_prompt_len = 120
    session = RolloutSession(
        trainer,
        sampling_params=params,
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
        agentic_context_overflow_policy="trim_messages",
    )

    async def run():
        session._loop = asyncio.get_running_loop()
        return await session._complete_chat(
            {"model": "policy", "messages": [_sys(), _usr("old"), _ast_no_tool("old response"), _usr("new")]}
        )

    response = asyncio.run(run())
    assert trainer.rollout_batches != []
    assert "trim_info" in response.get("areno", {})


def test_trim_messages_policy_impossible_returns_error():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _ChatTemplateTokenizer(tokens_per_message=100, tools_overhead=0, gen_prompt_overhead=50)
    params = _FakeSamplingParams()
    params.max_prompt_len = 50
    session = RolloutSession(
        trainer,
        sampling_params=params,
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
        agentic_context_overflow_policy="trim_messages",
    )

    async def run():
        session._loop = asyncio.get_running_loop()
        return await session._complete_chat(
            {"model": "policy", "messages": [_sys(), _usr("only task")]}
        )

    response = asyncio.run(run())
    assert response["choices"][0]["finish_reason"] == "length"
    assert response["areno"].get("error") == "context_overflow"
    assert trainer.rollout_batches == []


def test_trim_messages_response_carries_effective_messages():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _ChatTemplateTokenizer(tokens_per_message=50, tools_overhead=0, gen_prompt_overhead=0)
    params = _FakeSamplingParams()
    params.max_prompt_len = 120
    session = RolloutSession(
        trainer,
        sampling_params=params,
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
        agentic_context_overflow_policy="trim_messages",
    )

    async def run():
        session._loop = asyncio.get_running_loop()
        return await session._complete_chat(
            {"model": "policy", "messages": [_sys(), _usr("old"), _ast_no_tool("old"), _usr("new")]}
        )

    response = asyncio.run(run())
    assert "effective_messages" in response.get("areno", {})


def test_sample_from_trajectory_turn_uses_effective_messages():
    trainer = _FakeTrainer(world_size=1, tp_size=1)
    trainer.tokenizer = _ChatTemplateTokenizer(tokens_per_message=10, tools_overhead=0, gen_prompt_overhead=0)
    session = RolloutSession(
        trainer,
        sampling_params=_FakeSamplingParams(),
        loss_mask_policy=LossMaskPolicy(),
        max_running_prompts=1,
    )

    full_messages = [_sys(), _usr("old"), _ast_no_tool("old"), _usr("new")]
    effective_messages = [_sys(), _usr("new")]
    item = type("Item", (), {"prompt": "test", "input_tokens": [1],
                              "prompt_index": 0, "sample_index": 0, "record": {}})()
    # AgentTrajectoryTurn.__post_init__ reads response["areno"] metadata
    fake_response = {
        "areno": {
            "response_tokens": [10, 11],
            "response_logprobs": [-0.5, -0.5],
        }
    }
    turn = AgentTrajectoryTurn(
        item=item,
        messages=full_messages,
        effective_messages=effective_messages,
        response=fake_response,
    )

    sample = session._sample_from_trajectory_turn(turn)
    # 2 msgs * 10 = 20 prompt tokens + 2 response = 22 total
    assert len(sample.token_row) == 22


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------
def test_cli_accepts_valid_policy():
    from click.testing import CliRunner
    from areno.cli.train import train_command

    result = CliRunner().invoke(
        train_command,
        ["--help"],
        standalone_mode=False,
    )
    assert "--agentic-context-overflow-policy" in result.output
    assert "reject" in result.output
    assert "trim_messages" in result.output


def test_trainer_config_defaults_to_reject():
    from areno.api.trainer_config import TrainerConfig

    config = TrainerConfig(
        algo="sft",
        ckpt="dummy",
        dataset_path="dummy",
        dataset_loader_fn="dummy",
    )
    assert config.agentic_context_overflow_policy == "reject"


def test_trainer_config_rejects_invalid_policy():
    from areno.api.trainer_config import TrainerConfig

    with pytest.raises(ValueError, match="agentic_context_overflow_policy"):
        TrainerConfig(
            algo="sft",
            ckpt="dummy",
            dataset_path="dummy",
            dataset_loader_fn="dummy",
            agentic_context_overflow_policy="invalid",
        )


# ---------------------------------------------------------------------------
# unfittable response format test
# ---------------------------------------------------------------------------
def test_unfittable_chat_response_format():
    resp = _unfittable_chat_response(model="test", prompt_tokens=500, max_sequence_len=100)
    assert resp["choices"][0]["finish_reason"] == "length"
    assert resp["areno"]["error"] == "context_overflow"
    assert "detail" in resp["areno"]


# ---------------------------------------------------------------------------
# trim diagnostics format test
# ---------------------------------------------------------------------------
def test_trim_diagnostics_format():
    tok = _ChatTemplateTokenizer(tokens_per_message=20, tools_overhead=0, gen_prompt_overhead=0)
    messages = [_sys(), _usr("t1"), _ast_no_tool("ok"), _usr("t2")]
    result = _trim_messages_to_fit(
        tok, messages, tools=None, max_context_len=45, input_tokens=list(range(80))
    )
    assert result is not None
    diag = result["diagnostics"]
    assert diag["policy"] == "trim_messages"
    assert "original_prompt_tokens" in diag
    assert "effective_prompt_tokens" in diag
    assert "units_removed" in diag
    assert "messages_removed" in diag
    assert "preserved_instructions" in diag