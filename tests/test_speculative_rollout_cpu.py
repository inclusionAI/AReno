"""Speculative decode loop bookkeeping on CPU with a deterministic fake model.

The fake target always predicts ``token + 1``; the fake MTP draft predicts
``token + draft_shift``. With greedy sampling every draft is accepted when
``draft_shift == 1`` and every draft is rejected otherwise, so the exact
response tokens, finish reasons and commit counts are known in advance.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

import areno.engine.inference as inference_mod
from areno.engine.data import SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.inference import InferenceManager
from areno.models.base import CausalLMOutput
from tests.helpers import PatchedContext

VOCAB = 16
HIDDEN = 4


class _FakeSpecModel:
    def __init__(self, draft_shift: int):
        self.draft_shift = draft_shift
        self.commits: list[list[int]] = []
        self.draft_enabled = False

    def enable_mtp_draft(self, *, max_rows: int, tokens_per_seq: int) -> None:
        assert max_rows >= 1 and tokens_per_seq >= 2
        self.draft_enabled = True

    @staticmethod
    def _logits(tokens: torch.Tensor, shift: int) -> torch.Tensor:
        logits = torch.full((*tokens.shape, VOCAB), -10.0)
        return logits.scatter_(-1, ((tokens + shift) % VOCAB).unsqueeze(-1), 10.0)

    def __call__(self, *, input_ids, position_ids, infer_meta):
        if infer_meta.mode == "decode":
            assert input_ids.shape[1] == infer_meta.cache_seqlens.numel() * infer_meta.tokens_per_seq
        hidden = input_ids.unsqueeze(-1).expand(-1, -1, HIDDEN).float()
        return CausalLMOutput(logits_shard=self._logits(input_ids, 1), hidden_states=hidden)

    def mtp_draft_forward(self, *, input_ids, hidden_states, position_ids, infer_meta):
        assert hidden_states.shape[:2] == input_ids.shape
        return self._logits(input_ids, self.draft_shift), hidden_states

    def commit_speculative_state(self, committed: torch.Tensor, *, infer_meta) -> None:
        assert infer_meta.mode == "decode" and infer_meta.tokens_per_seq > 1
        self.commits.append(committed.tolist())


class _SpecManager(InferenceManager):
    def __init__(self, model: _FakeSpecModel, draft_tokens: int):
        super().__init__(SimpleNamespace())
        self.device = torch.device("cpu")
        self.model = model
        self.config = SimpleNamespace(
            runtime=SimpleNamespace(
                speculative_draft_tokens=draft_tokens,
                rollout_routing_replay=False,
                attn_backend="flash",
                eager_decode=False,
            ),
            model=SimpleNamespace(vocab_size=VOCAB),
            tp_size=1,
        )
        self._verify_graphs = {}
        self._draft_graphs = {}


def _rollout(prompts, *, draft_shift, max_new_tokens, eos=None, max_running=4, draft_tokens=2):
    model = _FakeSpecModel(draft_shift)
    manager = _SpecManager(model, draft_tokens)
    manager._infer_batch_size = max_running
    manager._enable_speculative_draft()
    state = InferenceBatchState(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        max_running_seqs=max_running,
        max_cache_len=64,
        max_prefill_tokens=64,
        kv_block_size=4,
        num_cache_blocks=64,
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1, rank=0, world_size=1)
    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state, SamplingParams(temperature=0.0), eos_token_id=eos, prompt_indices=list(range(len(prompts)))
        )
    assert model.draft_enabled
    return state, model


def _expected(prompt_last: int, length: int) -> list[int]:
    return [(prompt_last + 1 + i) % VOCAB for i in range(length)]


def test_accepted_drafts_fill_the_response_three_tokens_per_step():
    state, model = _rollout([[3], [7, 9]], draft_shift=1, max_new_tokens=7)
    assert state.generated == [_expected(3, 7), _expected(9, 7)]
    assert state.finish_reason == ["length", "length"]
    # Prefill gives 1 token, then 2 verify steps commit 3 fed tokens each.
    assert model.commits == [[3, 3], [3, 3]]
    assert state.metrics["decode_scheduled_tokens"] == 12.0
    assert state.metrics["spec_verify_rows"] == 4.0
    assert all(lp > -1e-6 for row in state.logprobs for lp in row)


def test_rejected_drafts_fall_back_to_one_token_per_step():
    state, model = _rollout([[3]], draft_shift=2, max_new_tokens=5)
    assert state.generated == [_expected(3, 5)]
    assert model.commits == [[1]] * 4


def test_stop_token_inside_accepted_drafts_truncates_the_row():
    # Tokens after 4: 5, 6, 7 ... ; stop at 7 arrives as the resampled bonus token.
    state, _ = _rollout([[3]], draft_shift=1, max_new_tokens=20, eos=7)
    assert state.generated == [[4, 5, 6, 7]]
    assert state.finish_reason == ["stop"]
    # Stop token as an accepted draft (second draft) also truncates.
    state, _ = _rollout([[3]], draft_shift=1, max_new_tokens=20, eos=6)
    assert state.generated == [[4, 5, 6]]
    assert state.finish_reason == ["stop"]


def test_length_cap_drops_surplus_tokens_of_the_last_step():
    state, _ = _rollout([[3]], draft_shift=1, max_new_tokens=5)
    assert state.generated == [_expected(3, 5)]
    assert state.finish_reason == ["length"]


def test_continuous_batching_admits_rows_into_a_running_speculative_batch():
    state, _ = _rollout([[1], [2], [3]], draft_shift=1, max_new_tokens=4, max_running=2)
    assert state.generated == [_expected(1, 4), _expected(2, 4), _expected(3, 4)]
    assert state.finish_reason == ["length"] * 3


def test_single_draft_needs_no_chain_step():
    state, model = _rollout([[3]], draft_shift=1, max_new_tokens=5, draft_tokens=1)
    assert state.generated == [_expected(3, 5)]
    assert model.commits == [[2], [2]]
