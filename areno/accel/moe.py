"""Fused MoE permute / unpermute kernels and their autograd glue.

These kernels move tokens between the natural token-major layout and the
expert-major layout consumed by ``areno_grouped_linear`` / the expert MLPs:

* :func:`areno_moe_permute` works from a dense routing map (one bool per
  expert per token) plus prob weights.
* :func:`areno_moe_topk_permute` consumes the more compact top-k indices /
  weights produced by the router and additionally returns the per-expert
  token count required by the grouped GEMM.
* :func:`areno_moe_unpermute` performs the scatter-add back to token-major,
  applying the route weights along the way.

Backward implementations reuse the inverse kernels so gradients are routed
through the same indirection tensors saved in the forward pass.
"""

import torch

from areno.accel._extension import extension as _extension


class _MoePermute(torch.autograd.Function):
    """Autograd glue for routing-map driven MoE permute."""

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor, num_out_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, route_weight, token_index = _extension().areno_moe_permute_forward(
            x.contiguous(),
            probs.contiguous(),
            routing_map.contiguous(),
            int(num_out_tokens),
        )
        ctx.save_for_backward(token_index)
        ctx.tokens = x.shape[0]
        ctx.hidden = x.shape[1]
        return out, route_weight, token_index

    @staticmethod
    def backward(
        ctx, grad_out: torch.Tensor, grad_route_weight: torch.Tensor, grad_token_index: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None]:
        del grad_route_weight, grad_token_index
        (token_index,) = ctx.saved_tensors
        grad_x = _extension().areno_moe_unpermute_forward(grad_out.contiguous(), token_index, ctx.tokens, ctx.hidden)
        return grad_x, None, None, None


class _MoeTopKPermute(torch.autograd.Function):
    """Autograd glue for top-k driven MoE permute (the common path)."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
        local_expert_start: int,
        local_num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out, route_weight, token_index, topk_position, tokens_per_expert = _extension().areno_moe_topk_permute_forward(
            x.contiguous(),
            topk_idx.contiguous(),
            topk_weight.contiguous(),
            int(local_expert_start),
            int(local_num_experts),
        )
        ctx.save_for_backward(token_index, topk_position)
        ctx.tokens = x.shape[0]
        ctx.hidden = x.shape[1]
        ctx.top_k = topk_idx.shape[1]
        return out, route_weight, token_index, tokens_per_expert

    @staticmethod
    def backward(
        ctx,
        grad_out: torch.Tensor,
        grad_route_weight: torch.Tensor,
        grad_token_index: torch.Tensor,
        grad_tokens_per_expert: torch.Tensor,
    ) -> tuple[torch.Tensor, None, torch.Tensor, None, None]:
        del grad_token_index, grad_tokens_per_expert
        token_index, topk_position = ctx.saved_tensors
        grad_x = _extension().areno_moe_unpermute_forward(grad_out.contiguous(), token_index, ctx.tokens, ctx.hidden)
        grad_topk_weight = _extension().areno_moe_topk_weight_backward(
            grad_route_weight.contiguous(),
            token_index,
            topk_position,
            ctx.tokens,
            ctx.top_k,
        )
        return grad_x, None, grad_topk_weight, None, None


class _MoeUnpermute(torch.autograd.Function):
    """Autograd glue for the expert-major -> token-major scatter."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, token_index: torch.Tensor, tokens: int, hidden: int) -> torch.Tensor:
        ctx.save_for_backward(token_index)
        return _extension().areno_moe_unpermute_forward(
            x.contiguous(), token_index.contiguous(), int(tokens), int(hidden)
        )

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        (token_index,) = ctx.saved_tensors
        return _extension().areno_moe_gather_by_token_index(grad_out.contiguous(), token_index), None, None, None


@torch._dynamo.disable
def areno_moe_permute(
    x: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor, num_out_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute MoE tokens in expert-major order and return route weights plus source token ids.

    ``x`` shape: ``(tokens, hidden)``. ``routing_map`` shape:
    ``(tokens, num_experts)`` bool. Returns
    ``(permuted_x, route_weight, token_index)`` where ``token_index`` records
    the source row of each expert-major output row.
    """
    if not x.is_cuda or not probs.is_cuda or not routing_map.is_cuda:
        raise RuntimeError("areno_moe_permute requires CUDA tensors")
    if routing_map.dtype != torch.bool:
        raise TypeError("areno_moe_permute routing_map must be bool")
    return _MoePermute.apply(x, probs, routing_map, int(num_out_tokens))


@torch._dynamo.disable
def areno_moe_topk_permute(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_weight: torch.Tensor,
    local_expert_start: int,
    local_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Permute MoE top-k routes directly into expert-major order.

    Only routes that land on this rank's expert shard
    (``[local_expert_start, local_expert_start + local_num_experts)``) are
    materialised. Returns ``(permuted_x, route_weight, token_index,
    tokens_per_expert)`` ready to feed into ``areno_grouped_linear``.
    """
    if not x.is_cuda or not topk_idx.is_cuda or not topk_weight.is_cuda:
        raise RuntimeError("areno_moe_topk_permute requires CUDA tensors")
    if topk_idx.dtype != torch.long:
        raise TypeError("areno_moe_topk_permute topk_idx must be int64")
    if topk_weight.dtype != torch.float32:
        raise TypeError("areno_moe_topk_permute topk_weight must be float32")
    return _MoeTopKPermute.apply(x, topk_idx, topk_weight, int(local_expert_start), int(local_num_experts))


@torch._dynamo.disable
def areno_moe_unpermute(x: torch.Tensor, token_index: torch.Tensor, restore_shape: tuple[int, int]) -> torch.Tensor:
    """Scatter-add expert-major rows back to token-major order.

    ``restore_shape`` is ``(tokens, hidden)`` of the pre-permute layout.
    Tokens that were routed to multiple experts accumulate via atomic add.
    """
    if not x.is_cuda or not token_index.is_cuda:
        raise RuntimeError("areno_moe_unpermute requires CUDA tensors")
    return _MoeUnpermute.apply(x, token_index, int(restore_shape[0]), int(restore_shape[1]))
