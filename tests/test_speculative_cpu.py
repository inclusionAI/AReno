from __future__ import annotations

import torch

from areno.engine.data import SamplingParams
from areno.engine.data.sampling import _sample
from areno.engine.runtime.speculative import (
    mtp_input_tokens,
    new_token_mask,
    sampling_probs,
    verify_drafts,
)


def _logits(rows: int, vocab: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(rows, vocab, generator=torch.Generator().manual_seed(seed)) * 3


def test_sampling_probs_matches_single_token_sampler_without_truncation():
    logits = _logits(6, 50)
    params = SamplingParams(temperature=0.7)
    probs = sampling_probs(logits, params, (), torch.zeros(6, dtype=torch.long))
    drawn = torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(1)).squeeze(-1)
    reference = _sample(logits, params, torch.device("cpu"), generator=torch.Generator().manual_seed(1))
    assert torch.equal(drawn, reference)
    assert torch.allclose(probs.sum(-1), torch.ones(6))


def test_sampling_probs_greedy_is_one_hot_argmax():
    logits = _logits(4, 20)
    probs = sampling_probs(logits, SamplingParams(temperature=0.0), (), torch.zeros(4, dtype=torch.long))
    assert torch.equal(probs.argmax(-1), logits.argmax(-1))
    assert torch.equal(probs.max(-1).values, torch.ones(4))


def test_sampling_probs_top_k_top_p_match_reference_truncation():
    logits = _logits(5, 30)
    params = SamplingParams(temperature=1.0, top_k=8, top_p=0.9)
    probs = sampling_probs(logits, params, (), torch.zeros(5, dtype=torch.long))
    # Plain reference: keep top-k, then the smallest prefix whose mass first
    # exceeds top-p, renormalize in vocab order.
    full = torch.softmax(logits, -1)
    for row in range(5):
        order = full[row].argsort(descending=True)
        kept = order[:8]
        cumulative = torch.cumsum(full[row, kept], 0)
        kept = kept[(cumulative - full[row, kept]) <= 0.9]
        expected = torch.zeros(30)
        expected[kept] = full[row, kept] / full[row, kept].sum()
        assert torch.allclose(probs[row], expected, atol=1e-6)


def test_sampling_probs_masks_eos_per_row_until_min_new_tokens():
    logits = torch.zeros(3, 6)
    logits[:, 2] = 10.0  # EOS is the argmax everywhere
    params = SamplingParams(temperature=1.0, min_new_tokens=3)
    probs = sampling_probs(logits, params, (2,), torch.tensor([1, 2, 3]))
    assert probs[0, 2] == 0.0 and probs[1, 2] == 0.0
    assert probs[2, 2] > 0.9


def test_verify_drafts_greedy_is_exact():
    vocab = 10
    target = torch.zeros(3, 3, vocab)
    target[0, 0, 4] = target[0, 1, 5] = target[0, 2, 6] = 1.0  # accepts both drafts, bonus 6
    target[1, 0, 4] = target[1, 1, 9] = target[1, 2, 0] = 1.0  # rejects second draft -> 9
    target[2, 0, 7] = target[2, 1, 1] = target[2, 2, 0] = 1.0  # rejects first draft -> 7
    drafts = torch.tensor([[4, 5], [4, 5], [4, 5]])
    draft_probs = torch.zeros(3, 2, vocab)
    draft_probs[:, 0, 4] = 1.0
    draft_probs[:, 1, 5] = 1.0
    new_tokens, accepted = verify_drafts(target, drafts, draft_probs)
    assert accepted.tolist() == [2, 1, 0]
    assert new_tokens[0].tolist() == [4, 5, 6]
    assert new_tokens[1, :2].tolist() == [4, 9]
    assert new_tokens[2, 0].item() == 7


def test_verify_drafts_preserves_target_distribution():
    torch.manual_seed(0)
    vocab, trials = 5, 40000
    p = torch.softmax(torch.randn(vocab) * 2, 0)
    q = torch.softmax(torch.randn(vocab) * 2, 0)
    target = p.expand(trials, 2, vocab).clone()
    draft_probs = q.expand(trials, 1, vocab).clone()
    generator = torch.Generator().manual_seed(1)
    drafts = torch.multinomial(q, trials, replacement=True, generator=generator).unsqueeze(-1)
    new_tokens, accepted = verify_drafts(target, drafts, draft_probs, generator=generator)
    first = torch.bincount(new_tokens[:, 0], minlength=vocab).float() / trials
    assert torch.allclose(first, p, atol=0.015), (first, p)
    expected_alpha = torch.minimum(p, q).sum()
    assert abs(accepted.float().mean() - expected_alpha) < 0.015
    # Bonus tokens (all drafts accepted) also follow the target.
    bonus = new_tokens[accepted == 1, 1]
    bonus_hist = torch.bincount(bonus, minlength=vocab).float() / bonus.numel()
    assert torch.allclose(bonus_hist, p, atol=0.03)


def test_new_token_mask_truncates_at_stop_and_length_cap():
    new_tokens = torch.tensor([[1, 9, 3], [1, 2, 9], [1, 2, 3], [1, 2, 3]])
    accepted = torch.tensor([2, 2, 1, 2])
    stop = torch.tensor([9])
    write_pos = torch.tensor([0, 0, 0, 6])
    mask = new_token_mask(new_tokens, accepted, stop, write_pos, max_new_tokens=8)
    assert mask.tolist() == [
        [True, True, False],  # stop token kept, nothing after it
        [True, True, True],  # stop token is the last new token
        [True, True, False],  # only accepted + 1 columns
        [True, True, False],  # length cap at 8 leaves two slots
    ]


def test_mtp_input_tokens_shifts_and_inserts_sampled_token():
    fed = torch.tensor([[10, 11, 12], [20, 21, 22]])
    new_tokens = torch.tensor([[11, 12, 13], [21, 99, 0]])
    accepted = torch.tensor([2, 1])
    shifted = mtp_input_tokens(fed, new_tokens, accepted)
    assert shifted[0].tolist() == [11, 12, 13]
    assert shifted[1, :2].tolist() == [21, 99]
