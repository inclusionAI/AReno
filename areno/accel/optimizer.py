"""Fused CUDA update for AReno's compact FP32-master AdamW."""

from __future__ import annotations

import torch

from areno.accel._extension import extension


@torch._dynamo.disable
@torch.no_grad()
def areno_adamw_fp32_master_step(
    model: torch.Tensor,
    low_bits: torch.Tensor,
    round_up_bits: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    state_offset: int,
    beta1: float,
    beta2: float,
    effective_lr: float,
    weight_decay: float,
    eps: float,
    step_size: float,
    bias_correction2_sqrt: float,
) -> None:
    """Update one contiguous model shard without FP32 temporary tensors."""

    tensors = (model, low_bits, round_up_bits, grad, exp_avg, exp_avg_sq)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused FP32-master AdamW requires CUDA tensors")
    if model.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused FP32-master AdamW requires bfloat16 or float32 model weights, got {model.dtype}")
    if grad.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused FP32-master AdamW requires bfloat16 or float32 gradients, got {grad.dtype}")
    if low_bits.dtype != torch.uint16 or round_up_bits.dtype != torch.uint8:
        raise TypeError("compact master metadata must use uint16 low bits and uint8 packed carries")
    if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
        raise TypeError("Adam moments must be float32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused FP32-master AdamW requires contiguous tensors")
    if model.numel() != grad.numel():
        raise ValueError("model and gradient shards must have the same number of elements")
    if state_offset < 0 or state_offset + model.numel() > low_bits.numel():
        raise ValueError("optimizer state slice is outside the compact master bucket")
    if low_bits.numel() != exp_avg.numel() or low_bits.numel() != exp_avg_sq.numel():
        raise ValueError("compact master metadata and Adam moments must have the same length")
    if round_up_bits.numel() < (low_bits.numel() + 7) // 8:
        raise ValueError("packed rounding-carry tensor is too short")
    extension().areno_adamw_fp32_master_step(
        model,
        low_bits,
        round_up_bits,
        grad,
        exp_avg,
        exp_avg_sq,
        state_offset,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt,
    )


__all__ = ["areno_adamw_fp32_master_step"]
