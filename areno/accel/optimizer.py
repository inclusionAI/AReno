"""Fused CUDA updates for AReno optimizers."""

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


@torch._dynamo.disable
@torch.no_grad()
def areno_adamw_8bit_step(
    model: torch.Tensor,
    grad: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_scale: torch.Tensor,
    exp_avg_sq_q: torch.Tensor,
    exp_avg_sq_scale: torch.Tensor,
    signed_codebook: torch.Tensor,
    unsigned_codebook: torch.Tensor,
    *,
    block_size: int,
    beta1: float,
    beta2: float,
    effective_lr: float,
    weight_decay: float,
    eps: float,
    step_size: float,
    bias_correction2_sqrt: float,
) -> None:
    """Update block-quantized AdamW state without full FP32 moments."""

    tensors = (
        model,
        grad,
        exp_avg_q,
        exp_avg_scale,
        exp_avg_sq_q,
        exp_avg_sq_scale,
        signed_codebook,
        unsigned_codebook,
    )
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused 8-bit AdamW requires CUDA tensors")
    if any(tensor.device != model.device for tensor in tensors[1:]):
        raise ValueError("fused 8-bit AdamW requires every tensor on the model device")
    if model.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused 8-bit AdamW requires bfloat16 or float32 model weights, got {model.dtype}")
    if grad.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused 8-bit AdamW requires bfloat16 or float32 gradients, got {grad.dtype}")
    if exp_avg_q.dtype != torch.uint8 or exp_avg_sq_q.dtype != torch.uint8:
        raise TypeError("quantized Adam moments must use uint8")
    if exp_avg_scale.dtype != torch.float32 or exp_avg_sq_scale.dtype != torch.float32:
        raise TypeError("quantized Adam scales must use float32")
    if signed_codebook.dtype != torch.float32 or unsigned_codebook.dtype != torch.float32:
        raise TypeError("dynamic quantization codebooks must use float32")
    if signed_codebook.numel() != 256 or unsigned_codebook.numel() != 256:
        raise ValueError("dynamic quantization codebooks must contain 256 entries")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused 8-bit AdamW requires contiguous tensors")
    if model.numel() != grad.numel() or model.numel() != exp_avg_q.numel():
        raise ValueError("model, gradient, and quantized moment tensors must have the same number of elements")
    if exp_avg_q.numel() != exp_avg_sq_q.numel():
        raise ValueError("first- and second-moment tensors must have the same number of elements")
    if block_size < 1 or block_size > 4096:
        raise ValueError(f"block_size must be between 1 and 4096, got {block_size}")
    block_count = (model.numel() + block_size - 1) // block_size
    if exp_avg_scale.numel() != block_count or exp_avg_sq_scale.numel() != block_count:
        raise ValueError("scale tensors must contain one value per quantization block")
    extension().areno_adamw_8bit_step(
        model,
        grad,
        exp_avg_q,
        exp_avg_scale,
        exp_avg_sq_q,
        exp_avg_sq_scale,
        signed_codebook,
        unsigned_codebook,
        block_size,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt,
    )


@torch._dynamo.disable
@torch.no_grad()
def areno_adamw_fp32_state_step(
    model: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    beta1: float,
    beta2: float,
    effective_lr: float,
    weight_decay: float,
    eps: float,
    step_size: float,
    bias_correction2_sqrt: float,
) -> None:
    """Update BF16/FP32 weights with persistent FP32 Adam moments."""

    tensors = (model, grad, exp_avg, exp_avg_sq)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused FP32-state AdamW requires CUDA tensors")
    if any(tensor.device != model.device for tensor in tensors[1:]):
        raise ValueError("fused FP32-state AdamW requires every tensor on the model device")
    if model.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused FP32-state AdamW requires bfloat16 or float32 model weights, got {model.dtype}")
    if grad.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused FP32-state AdamW requires bfloat16 or float32 gradients, got {grad.dtype}")
    if exp_avg.dtype != torch.float32 or exp_avg_sq.dtype != torch.float32:
        raise TypeError("FP32-state Adam moments must use float32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused FP32-state AdamW requires contiguous tensors")
    if any(tensor.numel() != model.numel() for tensor in tensors[1:]):
        raise ValueError("model, gradient, and FP32 moments must have the same number of elements")
    extension().areno_adamw_fp32_state_step(
        model,
        grad,
        exp_avg,
        exp_avg_sq,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt,
    )


@torch._dynamo.disable
@torch.no_grad()
def areno_adamw_4bit_step(
    model: torch.Tensor,
    grad: torch.Tensor,
    exp_avg_q: torch.Tensor,
    exp_avg_scale: torch.Tensor,
    exp_avg_sq_q: torch.Tensor,
    exp_avg_sq_scale: torch.Tensor,
    *,
    packed_offset: int,
    scale_offset: int,
    quant_block_size: int,
    beta1: float,
    beta2: float,
    effective_lr: float,
    weight_decay: float,
    eps: float,
    step_size: float,
    bias_correction2_sqrt: float,
) -> None:
    """Update one model shard directly from packed block-wise moments."""

    tensors = (model, grad, exp_avg_q, exp_avg_scale, exp_avg_sq_q, exp_avg_sq_scale)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("fused AdamW4bit requires CUDA tensors")
    if model.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused AdamW4bit requires bfloat16 or float32 model weights, got {model.dtype}")
    if grad.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"fused AdamW4bit requires bfloat16 or float32 gradients, got {grad.dtype}")
    if exp_avg_q.dtype != torch.uint8 or exp_avg_sq_q.dtype != torch.uint8:
        raise TypeError("packed AdamW4bit moments must use uint8 storage")
    if exp_avg_scale.dtype != torch.float32 or exp_avg_sq_scale.dtype != torch.float32:
        raise TypeError("AdamW4bit scales must use float32 storage")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused AdamW4bit requires contiguous tensors")
    if model.numel() != grad.numel():
        raise ValueError("model and gradient shards must have the same number of elements")
    if quant_block_size < 32 or quant_block_size > 1024 or quant_block_size & (quant_block_size - 1):
        raise ValueError("quant_block_size must be a power of two between 32 and 1024")
    packed_numel = (model.numel() + 1) // 2
    scale_numel = (model.numel() + quant_block_size - 1) // quant_block_size
    if packed_offset < 0 or packed_offset + packed_numel > exp_avg_q.numel():
        raise ValueError("packed AdamW4bit state slice is out of bounds")
    if exp_avg_q.numel() != exp_avg_sq_q.numel():
        raise ValueError("packed AdamW4bit moments must have the same length")
    if scale_offset < 0 or scale_offset + scale_numel > exp_avg_scale.numel():
        raise ValueError("AdamW4bit scale slice is out of bounds")
    if exp_avg_scale.numel() != exp_avg_sq_scale.numel():
        raise ValueError("AdamW4bit scale tensors must have the same length")
    extension().areno_adamw_4bit_step(
        model,
        grad,
        exp_avg_q,
        exp_avg_scale,
        exp_avg_sq_q,
        exp_avg_sq_scale,
        packed_offset,
        scale_offset,
        quant_block_size,
        beta1,
        beta2,
        effective_lr,
        weight_decay,
        eps,
        step_size,
        bias_correction2_sqrt,
    )


__all__ = [
    "areno_adamw_4bit_step",
    "areno_adamw_8bit_step",
    "areno_adamw_fp32_master_step",
    "areno_adamw_fp32_state_step",
]
