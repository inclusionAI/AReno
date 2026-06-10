"""Supervised fine-tuning loss over assistant/target tokens."""

from __future__ import annotations


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
        # SFT is plain negative log-likelihood averaged over target tokens.
        loss = -(logprobs[response_mask].sum() / valid_count)
        return loss, {
            "sft_loss": loss.detach(),
            "sft_target_tokens": valid_count.detach(),
            "sft_logprob_mean": (logprobs[response_mask].sum() / valid_count).detach(),
        }

    # Padded layout: position t predicts token t+1, so prompt_mask[:, 1:]
    # aligns with the returned next-token logprobs tensor.
    response_mask = (~data_pack["prompt_mask"][:, 1:]).to(device=logprobs.device, dtype=logprobs.dtype)
    valid_count = response_mask.sum().clamp_min(1.0)
    # Prompt and right-padding positions have zero weight in the loss.
    loss = -((logprobs * response_mask).sum() / valid_count)
    return loss, {
        "sft_loss": loss.detach(),
        "sft_target_tokens": valid_count.detach(),
        "sft_logprob_mean": ((logprobs * response_mask).sum() / valid_count).detach(),
    }
