"""Native Torch losses for canonical packed CUDA training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from areno.api.backend.common import TrainMetric

if TYPE_CHECKING:
    import torch


@dataclass(slots=True)
class ResponseLayout:
    """Response-token tensors from a canonical packed training batch."""

    response_mask: torch.Tensor
    valid_count: torch.Tensor
    response_len: torch.Tensor
    old_logprobs: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    ref_logprobs: torch.Tensor | None = None
    seq_ids: torch.Tensor | None = None
    num_sequences: int | None = None


def response_layout(
    data_pack: dict,
    logprobs: torch.Tensor,
    *,
    need_old_logprobs: bool = False,
    need_advantages: bool = False,
    need_ref_logprobs: bool = False,
    need_sequences: bool = False,
) -> ResponseLayout:
    """Return the response-token view from a canonical packed batch."""

    import torch

    device = logprobs.device
    response_mask = data_pack["packed_response_mask"].to(device=device, dtype=torch.float32)
    valid_count = response_mask.sum().clamp(min=1)

    seq_ids = None
    num_sequences = None
    if need_sequences:
        seq_ids = data_pack["packed_seq_ids"].to(device=device, dtype=torch.long)
        num_sequences = int(data_pack["packed_num_sequences"])
        response_len = torch.zeros(num_sequences, device=device, dtype=torch.float32)
        response_len.scatter_add_(0, seq_ids, response_mask)
        response_len = response_len.clamp(min=1)
    else:
        response_len = valid_count

    old_logprobs = data_pack["packed_logprobs"].to(device=device, dtype=torch.float32) if need_old_logprobs else None
    advantages = data_pack["packed_advantages"].to(device=device, dtype=torch.float32) if need_advantages else None
    ref_logprobs = (
        data_pack["packed_ref_logprobs"].to(device=device, dtype=torch.float32)
        if need_ref_logprobs and "packed_ref_logprobs" in data_pack
        else None
    )

    return ResponseLayout(
        response_mask=response_mask,
        valid_count=valid_count,
        response_len=response_len,
        old_logprobs=old_logprobs,
        advantages=advantages,
        ref_logprobs=ref_logprobs,
        seq_ids=seq_ids,
        num_sequences=num_sequences,
    )


def sequence_sum(values: torch.Tensor, layout: ResponseLayout) -> torch.Tensor:
    """Sum response-token values into one scalar per sequence."""

    import torch

    masked = values * layout.response_mask.to(dtype=values.dtype)
    if layout.seq_ids is None or layout.num_sequences is None:
        raise ValueError("packed sequence_sum requires seq_ids and num_sequences")
    out = torch.zeros(layout.num_sequences, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, layout.seq_ids, masked)
    return out


def masked_mean(values: torch.Tensor, layout: ResponseLayout) -> torch.Tensor:
    """Average values over response tokens only."""

    return (values * layout.response_mask.to(dtype=values.dtype)).sum() / layout.valid_count.to(dtype=values.dtype)


def sft_loss_fn(data_pack, logprobs):
    """Negative log-likelihood on non-prompt target tokens."""

    response_mask = data_pack["packed_response_mask"].to(device=logprobs.device).bool()
    valid_count = response_mask.sum().clamp_min(1)
    logprob_sum = logprobs[response_mask].sum()
    loss = _sft_token_mean_loss(data_pack, logprob_sum, valid_count, logprobs)
    return loss, {
        "sft_loss": loss.detach(),
        "sft_target_tokens": valid_count.detach(),
        "sft_logprob_mean": (-loss).detach(),
    }


def _sft_token_mean_loss(data_pack, logprob_sum, valid_count, logprobs):
    import torch

    total_target_tokens = data_pack.get("_sft_total_target_tokens")
    if total_target_tokens is None:
        return -(logprob_sum / valid_count.to(dtype=logprobs.dtype))
    denominator = torch.as_tensor(total_target_tokens, device=logprobs.device, dtype=logprobs.dtype).clamp_min(1)
    grad_scale = torch.as_tensor(data_pack.get("_sft_grad_scale", 1), device=logprobs.device, dtype=logprobs.dtype)
    return -(logprob_sum / denominator) * grad_scale


def dpo_loss_fn(data_pack, logprobs, *, beta: float = 0.1, label_smoothing: float = 0.0):
    """Pairwise Direct Preference Optimization loss."""

    layout = response_layout(data_pack, logprobs, need_ref_logprobs=True, need_sequences=True)
    num_sequences = int(layout.num_sequences)
    if num_sequences % 2 != 0:
        raise ValueError("DPO requires an even number of sequences per microbatch")
    ref_logprobs = layout.ref_logprobs.to(dtype=logprobs.dtype)
    return _dpo_from_sequence_logps(
        sequence_sum(logprobs, layout),
        sequence_sum(ref_logprobs, layout),
        layout.response_len.to(dtype=logprobs.dtype),
        float(beta),
        float(label_smoothing),
    )


def _dpo_from_sequence_logps(policy, reference, response_lens, beta: float, label_smoothing: float):
    import torch
    import torch.nn.functional as F

    chosen_policy, rejected_policy = policy[0::2], policy[1::2]
    chosen_ref, rejected_ref = reference[0::2], reference[1::2]
    logits = beta * ((chosen_policy - rejected_policy) - (chosen_ref - rejected_ref))
    losses = -(1.0 - label_smoothing) * F.logsigmoid(logits) - label_smoothing * F.logsigmoid(-logits)
    loss = losses.mean()
    chosen_rewards = beta * (chosen_policy - chosen_ref).detach()
    rejected_rewards = beta * (rejected_policy - rejected_ref).detach()
    return loss, {
        "dpo_loss": loss.detach(),
        "total_loss": loss.detach(),
        "dpo_accuracy": (logits > 0).to(dtype=torch.float32).mean().detach(),
        "dpo_margin": logits.mean().detach(),
        "dpo_reward_margin": (chosen_rewards - rejected_rewards).mean().detach(),
        "dpo_chosen_reward": chosen_rewards.mean().detach(),
        "dpo_rejected_reward": rejected_rewards.mean().detach(),
        "dpo_response_len": response_lens.clamp(min=1).mean().detach(),
    }


def grpo_loss_fn(data_pack, logprobs, *, clip_eps: float = 0.2):
    """Token-level clipped policy-gradient loss."""

    import torch

    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_sequences=True)
    ratio = torch.exp(logprobs - logprobs.detach())
    clipped = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    loss = (
        -torch.min(ratio * layout.advantages, clipped * layout.advantages) * layout.response_mask
    ).sum() / layout.valid_count
    valid_ratio = ratio[layout.response_mask.bool()]
    difference = layout.old_logprobs - logprobs.detach()
    return loss, {
        "policy_loss": loss.detach(),
        "total_loss": loss.detach(),
        TrainMetric.RATIO_MEAN: valid_ratio.mean().detach() if valid_ratio.numel() else ratio.mean().detach(),
        TrainMetric.RATIO_STD: valid_ratio.std().detach()
        if valid_ratio.numel() > 1
        else torch.zeros((), device=logprobs.device),
        "advantage_mean": masked_mean(layout.advantages, layout).detach(),
        "response_len": layout.response_len.mean().detach(),
        TrainMetric.ROLLOUT_LOGPROBS_MEAN: masked_mean(layout.old_logprobs, layout).detach(),
        TrainMetric.TRAIN_LOGPROBS_MEAN: masked_mean(logprobs.detach(), layout).detach(),
        TrainMetric.LOGP_DIFF_MEAN: masked_mean(difference, layout).detach(),
        TrainMetric.LOGP_ABS_DIFF_MEAN: masked_mean(difference.abs(), layout).detach(),
    }


def gspo_loss_fn(data_pack, logprobs, *, clip_eps: float = 3e-4):
    """Sequence-level clipped policy-gradient loss."""

    import torch

    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_sequences=True)
    seq_ratio = torch.exp(sequence_sum(logprobs - logprobs.detach(), layout) / layout.response_len)
    clipped = torch.clamp(seq_ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    seq_advantage = sequence_sum(layout.advantages, layout) / layout.response_len
    loss = -torch.min(seq_ratio * seq_advantage, clipped * seq_advantage).mean()
    difference = layout.old_logprobs - logprobs.detach()
    return loss, {
        "policy_loss": loss.detach(),
        "total_loss": loss.detach(),
        TrainMetric.RATIO_MEAN: seq_ratio.mean().detach(),
        TrainMetric.RATIO_STD: seq_ratio.std(unbiased=False).detach(),
        "advantage_mean": seq_advantage.mean().detach(),
        "response_len": layout.response_len.mean().detach(),
        TrainMetric.ROLLOUT_LOGPROBS_MEAN: masked_mean(layout.old_logprobs, layout).detach(),
        TrainMetric.TRAIN_LOGPROBS_MEAN: masked_mean(logprobs.detach(), layout).detach(),
        TrainMetric.LOGP_DIFF_MEAN: masked_mean(difference, layout).detach(),
        TrainMetric.LOGP_ABS_DIFF_MEAN: masked_mean(difference.abs(), layout).detach(),
    }


def ppo_loss_fn(
    data_pack,
    logprobs,
    *,
    clip_eps: float = 0.2,
    clip_ratio_c: float = 3.0,
    use_kl_loss: bool = False,
    kl_loss_coef: float = 0.001,
    kl_loss_type: str = "low_var_kl",
):
    """PPO clipped actor loss with optional reference KL penalty."""

    import torch

    layout = response_layout(data_pack, logprobs, need_old_logprobs=True, need_advantages=True, need_ref_logprobs=True)
    reference = layout.old_logprobs if layout.ref_logprobs is None else layout.ref_logprobs
    log_ratio = torch.clamp(logprobs - layout.old_logprobs, min=-20.0, max=20.0)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    losses1 = -ratio * layout.advantages
    losses2 = -clipped_ratio * layout.advantages
    clipped_losses = torch.maximum(losses1, losses2)
    dual_losses = -layout.advantages * float(clip_ratio_c)
    losses = torch.where(
        layout.advantages < 0.0,
        torch.minimum(dual_losses, clipped_losses),
        clipped_losses,
    )
    policy_loss = masked_mean(losses, layout)
    kl = masked_mean(_kl_penalty(logprobs, reference, kl_loss_type), layout)
    total_loss = policy_loss + (float(kl_loss_coef) * kl if use_kl_loss else 0.0)
    valid_ratio = ratio[layout.response_mask.bool()]
    return total_loss, {
        "policy_loss": policy_loss.detach(),
        "kl_loss": kl.detach(),
        "kl_coef": torch.tensor(float(kl_loss_coef) if use_kl_loss else 0.0, device=logprobs.device),
        "pg_clipfrac": masked_mean((losses2 > losses1).float(), layout).detach(),
        "pg_clipfrac_lower": masked_mean(
            ((clipped_losses > dual_losses) & (layout.advantages < 0.0)).float(), layout
        ).detach(),
        "ppo_kl": masked_mean(-log_ratio, layout).detach(),
        "total_loss": total_loss.detach(),
        TrainMetric.RATIO_MEAN: valid_ratio.mean().detach() if valid_ratio.numel() else ratio.mean().detach(),
        TrainMetric.RATIO_STD: valid_ratio.std().detach()
        if valid_ratio.numel() > 1
        else torch.zeros((), device=logprobs.device),
        "advantage_mean": masked_mean(layout.advantages, layout).detach(),
    }


def _kl_penalty(logprob, ref_logprob, kl_type: str):
    import torch

    difference = logprob - ref_logprob
    if kl_type in {"kl", "k1"}:
        return difference
    if kl_type == "abs":
        return difference.abs()
    if kl_type in {"mse", "k2"}:
        return 0.5 * difference.square()
    if kl_type in {"low_var_kl", "k3"}:
        reverse = torch.clamp(-difference, min=-20.0, max=20.0)
        return torch.clamp(torch.exp(reverse) - reverse - 1.0, min=-10.0, max=10.0)
    raise NotImplementedError(f"unsupported PPO kl_loss_type: {kl_type}")


def dispatch_loss(pack: dict, logprobs):
    """Invoke the loss callable carried by a CUDA training pack."""

    loss_fn = pack.get("_loss_fn")
    if not callable(loss_fn):
        raise ValueError("CUDA train data pack is missing callable _loss_fn")
    return loss_fn(pack, logprobs)


__all__ = [
    "dispatch_loss",
    "dpo_loss_fn",
    "grpo_loss_fn",
    "gspo_loss_fn",
    "masked_mean",
    "ppo_loss_fn",
    "response_layout",
    "sequence_sum",
    "sft_loss_fn",
]
