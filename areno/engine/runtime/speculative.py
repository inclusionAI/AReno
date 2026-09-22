"""Speculative-decoding tensor math for the rollout engine.

Chain drafting verifies ``k`` draft tokens per sequence with one target
forward over ``S = k + 1`` positions: the already-sampled token followed by
the drafts. Rejection sampling (Leviathan et al. 2023; Chen et al. 2023)
keeps the sampled sequence distributed exactly as single-token decoding:

    accept draft d_j   with probability  min(1, p_j(d_j) / q_j(d_j))
    on first rejection sample from       norm(max(p_j - q_j, 0))
    if every draft passes sample from    p_k

where ``p_j`` is the target's *processed* distribution (temperature, top-k /
top-p, EOS and suppression masks) at fed position ``j`` and ``q_j`` the
draft's. Every helper here is pure tensor code so it runs and is tested on
CPU; the engine supplies gathered full-vocab logits.
"""

from __future__ import annotations

import torch

from areno.engine.data import SamplingParams


def sampling_probs(
    logits: torch.Tensor,
    params: SamplingParams,
    eos_token_ids: tuple[int, ...],
    sample_steps: torch.Tensor,
) -> torch.Tensor:
    """Return the full-vocab distribution single-token decode would sample from.

    Mirrors ``areno.engine.data.sampling._sample`` step for step, but returns
    the distribution instead of a draw and takes one ``sample_steps`` entry per
    row so tokens at different response offsets get the right ``min_new_tokens``
    EOS mask. Greedy decoding (temperature 0) is the one-hot argmax, which
    makes rejection sampling reduce to exact greedy verification.
    """

    scores = torch.nan_to_num(
        logits.float(),
        nan=float("-inf"),
        posinf=torch.finfo(torch.float32).max,
        neginf=float("-inf"),
    )
    vocab = scores.shape[-1]
    suppress_ids = [token_id for token_id in params.suppress_token_ids if 0 <= token_id < vocab]
    if suppress_ids:
        scores = scores.clone()
        scores[:, torch.tensor(suppress_ids, device=scores.device, dtype=torch.long)] = float("-inf")
    eos_ids = [token_id for token_id in eos_token_ids if 0 <= token_id < vocab]
    if eos_ids and params.min_new_tokens > 0:
        eos_mask = torch.zeros(vocab, device=scores.device, dtype=torch.bool)
        eos_mask[torch.tensor(eos_ids, device=scores.device, dtype=torch.long)] = True
        too_early = (sample_steps < params.min_new_tokens).unsqueeze(-1)
        scores = scores.masked_fill(too_early & eos_mask, float("-inf"))
    fallback = torch.argmax(scores, dim=-1)
    if params.temperature == 0.0:
        return torch.nn.functional.one_hot(fallback, vocab).to(torch.float32)
    probs = torch.softmax(scores / params.temperature, dim=-1)
    if params.top_k <= 0 and params.top_p >= 1.0:
        # Softmax rows are already normalized; only an all -inf row (NaN out)
        # needs the same argmax fallback `_sample` applies. Two passes instead
        # of the general renormalization below.
        degenerate = torch.isnan(probs[:, :1])
        return torch.where(degenerate, torch.nn.functional.one_hot(fallback, vocab).to(probs.dtype), probs)
    probs = _renormalize_or_one_hot(probs, fallback)
    if params.top_k > 0 or params.top_p < 1.0:
        # Same truncation rule as `_sample`: sort descending, drop tokens past
        # top-k and tokens whose exclusive prefix mass already exceeds top-p,
        # then scatter the renormalized tail back into vocab order.
        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
        if params.top_k > 0:
            probs_sort[:, min(params.top_k, vocab) :] = 0.0
        if params.top_p < 1.0:
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            probs_sort[(probs_sum - probs_sort) > params.top_p] = 0.0
        probs_sort = _renormalize_or_one_hot(probs_sort, torch.zeros_like(fallback))
        probs = torch.zeros_like(probs).scatter_(-1, probs_idx, probs_sort)
    return probs


def _renormalize_or_one_hot(probs: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    """Normalize rows; rows that lost all mass become one-hot on ``fallback``."""

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp_min_(0.0)
    row_sum = probs.sum(dim=-1, keepdim=True)
    normalized = probs / row_sum.clamp_min(torch.finfo(probs.dtype).tiny)
    one_hot = torch.zeros_like(probs).scatter_(-1, fallback.view(-1, 1), 1.0)
    return torch.where(row_sum > 0.0, normalized, one_hot)


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Draw one token per row from ``probs`` by the Gumbel-max trick.

    Equivalent in distribution to ``torch.multinomial(probs, 1)`` but two
    elementwise passes and an argmax instead of a prefix scan, which matters
    at 150k-token vocabularies inside the decode loop.
    """

    uniform = torch.rand(probs.shape, device=probs.device, generator=generator)
    gumbel = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(torch.float32).tiny)))
    return torch.argmax(torch.log(probs) + gumbel, dim=-1)


def selected_logprobs(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Raw ``log_softmax(logits)[token]`` per row for full-vocab logits ``(M, V)``."""

    scores = logits.float()
    return scores.gather(-1, tokens.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(scores, dim=-1)


def verify_drafts(
    target_probs: torch.Tensor,
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run chain rejection sampling for a batch of sequences.

    ``target_probs`` is ``(N, k + 1, V)``: position ``j`` is the target's
    distribution for the token that follows fed token ``j``. ``draft_tokens``
    ``(N, k)`` were drawn from ``draft_probs`` ``(N, k, V)``. Returns
    ``new_tokens`` ``(N, k + 1)`` holding the accepted drafts followed by the
    freshly sampled token (columns past that are padding) and ``accepted``
    ``(N,)``, the number of accepted drafts; row ``n`` produced
    ``accepted[n] + 1`` new tokens.
    """

    batch, num_drafts = draft_tokens.shape
    if num_drafts == 0:
        raise ValueError("verify_drafts needs at least one draft token per sequence")
    rows = torch.arange(batch, device=draft_tokens.device)
    p_draft = target_probs[:, :num_drafts].gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    q_draft = draft_probs.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)
    uniform = torch.rand(p_draft.shape, device=p_draft.device, generator=generator)
    # Accept iff u < p/q, written multiplicatively so q == 0 rejects and
    # one-hot (greedy) inputs accept exactly when both argmaxes agree.
    accept = uniform * q_draft < p_draft
    # Chain rule: a draft counts only if every earlier draft was accepted.
    accepted = torch.cumprod(accept.to(torch.long), dim=-1).sum(dim=-1)
    # Residual at the first rejected position; the bonus position (all drafts
    # accepted) samples from the target directly, i.e. residual with q = 0.
    p_resample = target_probs[rows, accepted]
    partial = (accepted < num_drafts).unsqueeze(-1)
    q_resample = torch.where(partial, draft_probs[rows, accepted.clamp(max=num_drafts - 1)], 0.0)
    residual = _renormalize_or_one_hot((p_resample - q_resample).clamp_min(0.0), p_resample.argmax(dim=-1))
    resampled = sample_from_probs(residual, generator)
    new_tokens = torch.zeros(batch, num_drafts + 1, device=draft_tokens.device, dtype=draft_tokens.dtype)
    new_tokens[:, :num_drafts] = draft_tokens
    new_tokens[rows, accepted] = resampled
    return new_tokens, accepted


def new_token_mask(
    new_tokens: torch.Tensor,
    accepted: torch.Tensor,
    stop_token_ids: torch.Tensor | None,
    write_pos: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Mark which columns of ``new_tokens`` enter the response buffer.

    A column is kept when it is one of the row's ``accepted + 1`` new tokens,
    it does not follow a stop token already kept in this step, and the row
    still has room under ``max_new_tokens`` at ``write_pos + column``.
    """

    columns = torch.arange(new_tokens.shape[1], device=new_tokens.device)
    valid = columns.unsqueeze(0) <= accepted.unsqueeze(-1)
    if stop_token_ids is not None:
        is_stop = new_tokens.unsqueeze(-1).eq(stop_token_ids).any(dim=-1) & valid
        # Keep everything up to and including the first stop token.
        stopped_before = torch.cumsum(is_stop.to(torch.long), dim=-1) - is_stop.to(torch.long)
        valid &= stopped_before == 0
    valid &= (write_pos.unsqueeze(-1) + columns.unsqueeze(0)) < max_new_tokens
    return valid


def mtp_input_tokens(fed_tokens: torch.Tensor, new_tokens: torch.Tensor, accepted: torch.Tensor) -> torch.Tensor:
    """Build the MTP layer's token inputs for the fed positions after a verify.

    An MTP layer at trunk position ``t`` consumes the embedding of token
    ``t + 1`` together with the trunk hidden state at ``t``. For the fed
    positions ``0..k`` that next token is the following fed token, except at
    the last committed position ``accepted`` where it is the freshly sampled
    token; positions past ``accepted`` are never read.
    """

    shifted = torch.cat([fed_tokens[:, 1:], new_tokens[:, -1:]], dim=-1)
    rows = torch.arange(fed_tokens.shape[0], device=fed_tokens.device)
    shifted[rows, accepted] = new_tokens[rows, accepted]
    return shifted
