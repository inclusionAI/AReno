"""Fused MoE expert-alignment kernel used by the fused MoE forward path.

Given a flattened ``(tokens, top_k)`` matrix of expert ids, ``areno_moe_align``
produces the indirection arrays needed by the block-wise expert MLP kernel:
``sorted_token_ids`` orders tokens by destination expert (and optionally pads
each expert block up to ``block_size``), ``expert_ids`` records the expert
served by each output block, and ``num_tokens_post_pad`` reports the padded
total length so launch parameters can be sized correctly. The kernel writes
into caller-allocated buffers in place; ``cumsum_buffer`` provides scratch
space for the prefix-sum step and must be supplied.
"""
import torch

from areno.accel._extension import extension as _extension


@torch._dynamo.disable
def areno_moe_align(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    cumsum_buffer: torch.Tensor | None = None,
    pad_sorted_token_ids: bool = True,
) -> None:
    """Build ARENO MoE routing indirection grouped by expert and block padded.

    Output tensors are written in place. ``cumsum_buffer`` is mandatory scratch
    space sized to hold the per-expert offsets; passing ``None`` is rejected
    so that callers manage allocation lifetime explicitly.
    """
    if topk_ids.device.type != "cuda":
        raise ValueError("areno_moe_align expects CUDA tensors")
    if cumsum_buffer is None:
        raise ValueError("areno_moe_align requires cumsum_buffer")
    _extension().areno_moe_align(
        topk_ids,
        int(num_experts),
        int(block_size),
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        bool(pad_sorted_token_ids),
    )
