"""Fused linear and grouped-linear kernels for dense and MoE projections.

``areno_linear`` is a drop-in fused matmul + optional bias used where PyTorch's
``F.linear`` would otherwise dispatch to cuBLAS, and lets the autograd path
reuse the extension's tuned backward. ``areno_grouped_linear`` performs
per-expert matmuls over contiguous token groups (post-permute MoE layout)
without launching one kernel per expert.
"""

from collections.abc import Sequence

import torch

from areno.accel._extension import extension as _extension


class _Linear(torch.autograd.Function):
    """Autograd glue for the fused dense linear projection."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        use_bias = bias is not None
        # The kernel always takes a concrete bias tensor; pass an empty
        # placeholder when bias is unused.
        kernel_bias = bias if use_bias else torch.empty(0, device=x.device, dtype=x.dtype)
        out = _extension().areno_linear_forward(x.contiguous(), weight.contiguous(), kernel_bias.contiguous(), use_bias)
        ctx.save_for_backward(x, weight)
        ctx.use_bias = use_bias
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        x, weight = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = _extension().areno_linear_backward(
            grad_output.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            ctx.use_bias,
        )
        return grad_input, grad_weight, grad_bias if ctx.use_bias else None


@torch._dynamo.disable
def areno_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Apply an ARENO CUDA linear projection over the last input dimension.

    Weight layout matches ``torch.nn.Linear`` (``out_features``, ``in_features``).
    Returns a tensor with the trailing dim replaced by ``out_features``.
    """
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("areno_linear requires CUDA input and weight")
    if bias is not None and not bias.is_cuda:
        raise RuntimeError("areno_linear bias must be CUDA")
    return _Linear.apply(x, weight, bias)


class _GroupedLinear(torch.autograd.Function):
    """Autograd glue for batched per-expert matmuls over a contiguous token layout."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, tokens_per_expert: list[int]) -> torch.Tensor:
        counts = [int(count) for count in tokens_per_expert]
        out = _extension().areno_grouped_linear_forward(x.contiguous(), weight.contiguous(), counts)
        ctx.save_for_backward(x, weight)
        ctx.tokens_per_expert = counts
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight = ctx.saved_tensors
        grad_input, grad_weight = _extension().areno_grouped_linear_backward(
            grad_output.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            ctx.tokens_per_expert,
        )
        return grad_input, grad_weight, None


class _GroupedLinearCounts(torch.autograd.Function):
    """Grouped linear using a CUDA counts tensor to avoid Python count materialization."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, tokens_per_expert: torch.Tensor) -> torch.Tensor:
        out = _extension().areno_grouped_linear_forward_counts(
            x.contiguous(), weight.contiguous(), tokens_per_expert.contiguous()
        )
        ctx.save_for_backward(x, weight, tokens_per_expert)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None]:
        x, weight, tokens_per_expert = ctx.saved_tensors
        grad_input, grad_weight = _extension().areno_grouped_linear_backward_counts(
            grad_output.contiguous(),
            x.contiguous(),
            weight.contiguous(),
            tokens_per_expert.contiguous(),
        )
        return grad_input, grad_weight, None


@torch._dynamo.disable
def areno_grouped_linear(
    x: torch.Tensor, weight: torch.Tensor, tokens_per_expert: torch.Tensor | Sequence[int]
) -> torch.Tensor:
    """Apply per-expert linear projections to contiguous token groups.

    ``x`` is the permuted token matrix laid out expert-major and
    ``tokens_per_expert[i]`` gives the row count for expert ``i``. ``weight``
    has shape ``(num_experts, out_features, in_features)``. Returns the
    expert-major output matrix; the caller is responsible for unpermuting.
    """
    if not x.is_cuda or not weight.is_cuda:
        raise RuntimeError("areno_grouped_linear requires CUDA input and weight")
    if weight.dim() != 3:
        raise ValueError(f"areno_grouped_linear weight must be 3D, got {tuple(weight.shape)}")
    if isinstance(tokens_per_expert, torch.Tensor):
        if not tokens_per_expert.is_cuda:
            raise RuntimeError("areno_grouped_linear tensor tokens_per_expert must be CUDA")
        return _GroupedLinearCounts.apply(x, weight, tokens_per_expert)
    return _GroupedLinear.apply(x, weight, tokens_per_expert)
