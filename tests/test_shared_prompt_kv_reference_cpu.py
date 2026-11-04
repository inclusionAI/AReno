"""CPU numerical oracle for issue #592; this does not claim CUDA backend coverage."""

from __future__ import annotations

import pytest
import torch

from areno.engine.runtime import logprobs as logprob_ops
from areno.engine.runtime.shared_kv_reference import SharedPromptKVCache, TinyCausalLM
from tests.helpers import PatchedContext, single_tp_context

ATOL = 1e-6
RTOL = 1e-6
_MAX_ABS_ERROR = {"logits": 0.0, "logprobs": 0.0}


@pytest.fixture(scope="module", autouse=True)
def report_max_abs_error():
    yield
    print(
        "shared-KV reference max abs deviation: "
        f"logits={_MAX_ABS_ERROR['logits']:.9g}, logprobs={_MAX_ABS_ERROR['logprobs']:.9g}"
    )


@pytest.fixture(autouse=True)
def inference_mode():
    with torch.inference_mode():
        yield


def make_model(*, max_seq_len: int = 64) -> TinyCausalLM:
    torch.manual_seed(592)
    return TinyCausalLM(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4, max_seq_len=max_seq_len).eval()


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, kind: str = "logits") -> None:
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)
    if actual.numel():
        error = (actual.float() - expected.float()).abs().max().item()
        _MAX_ABS_ERROR[kind] = max(_MAX_ABS_ERROR[kind], error)


def decode_and_compare(cache: SharedPromptKVCache, branch_index: int, token: int) -> torch.Tensor:
    actual = cache.branches[branch_index].decode(token)
    expected = cache.model(cache.branches[branch_index].tokens)[-1]
    assert_close(actual, expected)
    return actual


def snapshot_prefix(cache: SharedPromptKVCache) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple((key.clone(), value.clone()) for key, value in cache.prefix)


def assert_prefix_unchanged(cache: SharedPromptKVCache, snapshot) -> None:
    for (actual_key, actual_value), (expected_key, expected_value) in zip(cache.prefix, snapshot, strict=True):
        torch.testing.assert_close(actual_key, expected_key, atol=0, rtol=0)
        torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)


def assert_private_lengths(cache: SharedPromptKVCache) -> None:
    for branch in cache.branches:
        assert len(branch.private_suffix) == len(cache.prefix)
        for pair in branch.private_suffix:
            if branch.suffix_tokens:
                assert pair is not None
                assert pair[0].shape[1] == len(branch.suffix_tokens)
                assert pair[1].shape[1] == len(branch.suffix_tokens)
            else:
                assert pair is None


def test_shared_prompt_prefill_matches_independent_full_forwards():
    model = make_model()
    cache = SharedPromptKVCache(model)
    prompt = torch.tensor([3, 1, 4, 1, 5])
    branches = cache.start(prompt, num_samples=3)

    expected = model(prompt)
    assert_close(cache.prefill_logits, expected)
    assert len({id(branch.shared_prefix) for branch in branches}) == 1
    assert all(pair[0].shape[1] == prompt.numel() for pair in cache.prefix)
    assert_private_lengths(cache)


def test_interleaved_divergent_decode_is_isolated_and_matches_full_forward():
    model = make_model()
    cache = SharedPromptKVCache(model)
    prompt = torch.tensor([2, 7, 3, 8])
    branches = cache.start(prompt, num_samples=3)
    prefix_before = snapshot_prefix(cache)
    untouched = branches[1].tokens.clone()

    decode_and_compare(cache, 0, 11)
    torch.testing.assert_close(branches[1].tokens, untouched, atol=0, rtol=0)
    assert branches[1].suffix_tokens == ()
    assert all(pair is None for pair in branches[1].private_suffix)

    schedule = [(1, 21), (2, 31), (0, 12), (1, 22), (0, 13), (2, 32), (2, 33), (0, 14)]
    for branch_index, token in schedule:
        decode_and_compare(cache, branch_index, token)
        assert_prefix_unchanged(cache, prefix_before)

    assert branches[0].suffix_tokens == (11, 12, 13, 14)
    assert branches[1].suffix_tokens == (21, 22)
    assert branches[2].suffix_tokens == (31, 32, 33)
    assert_private_lengths(cache)
    assert all(pair[0].shape[1] == prompt.numel() for pair in cache.prefix)


@pytest.mark.parametrize(
    "suffixes",
    [
        [(5, 6, 7, 8)],
        [(), (11,), (21, 22, 23), (31, 32)],
    ],
    ids=["single-sample", "unequal-lengths"],
)
def test_variable_lengths_and_single_sample_degenerate_case(suffixes):
    model = make_model()
    cache = SharedPromptKVCache(model)
    cache.start(torch.tensor([9, 2, 6, 4]), num_samples=len(suffixes))
    prefix_before = snapshot_prefix(cache)

    for step in range(max(map(len, suffixes))):
        for branch_index, suffix in enumerate(suffixes):
            if step < len(suffix):
                decode_and_compare(cache, branch_index, suffix[step])

    for branch, suffix in zip(cache.branches, suffixes, strict=True):
        assert branch.suffix_tokens == suffix
        if not suffix:
            assert_close(cache.prefill_logits[-1], model(branch.tokens)[-1])
    assert_private_lengths(cache)
    assert_prefix_unchanged(cache, prefix_before)


def test_capacity_overflow_does_not_mutate_shared_or_private_kv():
    model = make_model(max_seq_len=6)
    cache = SharedPromptKVCache(model)
    cache.start(torch.tensor([2, 4, 6, 8]), num_samples=2)
    decode_and_compare(cache, 0, 10)
    decode_and_compare(cache, 0, 12)
    prefix_before = snapshot_prefix(cache)
    suffix_before = tuple(
        None if pair is None else (pair[0].clone(), pair[1].clone())
        for pair in cache.branches[0].private_suffix
    )

    with pytest.raises(ValueError, match="exceeds max_seq_len=6"):
        cache.branches[0].decode(14)

    assert cache.branches[0].suffix_tokens == (10, 12)
    assert cache.branches[1].suffix_tokens == ()
    for actual, expected in zip(cache.branches[0].private_suffix, suffix_before, strict=True):
        assert actual is not None and expected is not None
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
    assert_prefix_unchanged(cache, prefix_before)
    decode_and_compare(cache, 1, 20)


def test_reset_invalidates_old_branches_and_reuse_has_no_stale_state():
    model = make_model()
    cache = SharedPromptKVCache(model)
    old_branches = cache.start(torch.tensor([1, 2, 3, 4]), num_samples=2)
    decode_and_compare(cache, 0, 10)
    decode_and_compare(cache, 1, 20)
    old_prefix_storage = cache.prefix
    old_prefix = snapshot_prefix(cache)

    with pytest.raises(RuntimeError, match="reset is required"):
        cache.start(torch.tensor([4, 3, 2, 1]), num_samples=1)
    cache.reset()
    cache.reset()
    assert not cache.active
    assert cache.branches == ()
    assert cache.prefix == ()
    assert cache.prefill_logits.numel() == 0
    assert all(not branch.private_suffix and not branch.suffix_tokens for branch in old_branches)
    with pytest.raises(RuntimeError, match="reset shared-KV generation"):
        old_branches[0].decode(30)

    new_prompt = torch.tensor([4, 3, 2, 1])
    cache.start(new_prompt, num_samples=2)
    assert_close(cache.prefill_logits, model(new_prompt))
    new_prefix = snapshot_prefix(cache)
    decode_and_compare(cache, 0, 30)
    decode_and_compare(cache, 1, 40)
    assert_prefix_unchanged(cache, new_prefix)
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        old_prefix_storage, old_prefix, strict=True
    ):
        torch.testing.assert_close(actual_key, expected_key, atol=0, rtol=0)
        torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)


def test_shared_behavior_logprobs_match_training_side_full_forward():
    model = make_model()
    cache = SharedPromptKVCache(model)
    prompt = torch.tensor([6, 1, 8, 3, 9])
    cache.start(prompt, num_samples=2)
    suffixes = [(5, 7, 11), (12, 13)]

    for branch_index, suffix in enumerate(suffixes):
        predictors = [cache.prefill_logits[-1]]
        for token in suffix:
            predictors.append(decode_and_compare(cache, branch_index, token))

        branch = cache.branches[branch_index]
        full_tokens = branch.tokens.unsqueeze(0)
        full_logits = model(branch.tokens)
        behavior_logits = torch.stack(predictors[:-1])
        assert_close(behavior_logits, full_logits[prompt.numel() - 1 : -1])
        targets = torch.tensor(suffix, dtype=torch.long)
        actual = torch.log_softmax(behavior_logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        with PatchedContext(logprob_ops, get_tp_context=single_tp_context):
            training_logprobs = logprob_ops.next_token_logprobs(full_logits.unsqueeze(0), full_tokens)[0]
        expected = training_logprobs[prompt.numel() - 1 :]
        assert_close(actual, expected, kind="logprobs")
