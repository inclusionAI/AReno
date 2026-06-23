"""Supervised fine-tuning loss over assistant/target tokens."""

from __future__ import annotations

import torch


def sft_loss_fn(data_pack, logprobs):
    """Negative log-likelihood on non-prompt target tokens.

    The backend computes next-token logprobs for realized labels. `prompt_mask`
    is aligned with token positions, so target positions use `[:, 1:]` in the
    padded path and `packed_response_mask` in the packed path.
    """

    if "packed_response_mask" in data_pack:
        # Packed varlen layout: logprobs is 1D over all next-token positions;
        # packed_response_mask already excludes prompt and padding positions.
        response_mask = data_pack["packed_response_mask"].to(device=logprobs.device).bool()
        valid_count = response_mask.sum().clamp_min(1)
        logprob_sum = logprobs[response_mask].sum()
        # SFT is plain negative log-likelihood averaged over target tokens.
        loss = _sft_token_mean_loss(data_pack, logprob_sum, valid_count, logprobs)
        return loss, {
            "sft_loss": loss.detach(),
            "sft_target_tokens": valid_count.detach(),
            "sft_logprob_mean": (-loss).detach(),
        }

    # Padded layout: position t predicts token t+1, so prompt_mask[:, 1:]
    # aligns with the returned next-token logprobs tensor.
    response_mask = (~data_pack["prompt_mask"][:, 1:]).to(device=logprobs.device, dtype=logprobs.dtype)
    valid_count = response_mask.sum().clamp_min(1.0)
    logprob_sum = (logprobs * response_mask).sum()
    # Prompt and right-padding positions have zero weight in the loss.
    loss = _sft_token_mean_loss(data_pack, logprob_sum, valid_count, logprobs)
    return loss, {
        "sft_loss": loss.detach(),
        "sft_target_tokens": valid_count.detach(),
        "sft_logprob_mean": (-loss).detach(),
    }


def _sft_token_mean_loss(data_pack, logprob_sum, valid_count, logprobs):
    """Return local token mean, or global accumulation-group token mean when annotated."""

    total_target_tokens = data_pack.get("_sft_total_target_tokens")
    if total_target_tokens is None:
        return -(logprob_sum / valid_count.to(dtype=logprobs.dtype))
    denominator = torch.as_tensor(total_target_tokens, device=logprobs.device, dtype=logprobs.dtype).clamp_min(1)
    grad_scale = torch.as_tensor(data_pack.get("_sft_grad_scale", 1), device=logprobs.device, dtype=logprobs.dtype)
    # TrainingManager divides every microbatch loss by grad_scale before
    # backward. Multiplying here makes the summed accumulation-group gradient
    # equal to -sum(logprobs) / total_target_tokens.
    return -(logprob_sum / denominator) * grad_scale
