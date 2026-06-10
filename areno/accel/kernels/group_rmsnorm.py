"""Fused group RMSNorm + SiLU/Swish-gate kernel.

This file implements ``y = (x / rms(x)) * w * sigmoid(gate)`` in a single
Triton kernel. The input is laid out as a 3D tensor of shape ``(M, G, N)``
where ``M`` rows of ``G`` groups each carry an ``N``-wide feature vector that
is independently RMS-normalised. Per-row, per-group reciprocal standard
deviations (``rstd``) are cached and returned so a matching backward kernel
can reuse them without recomputing the reduction.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# Autotune over (num_warps, num_stages) keyed on the feature width N. Picking
# num_warps here trades launch occupancy against per-program register usage;
# num_stages governs how aggressively Triton software-pipelines the load/use
# of input tiles. We sweep a broad range because the optimal point depends on
# both the GPU and N, which is fixed per autotune cache entry.
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8, 16, 32]
        for num_stages in [2, 3, 4]
    ],
    key=["N"],
)
@triton.jit
def _rms_norm_gate_kernel(
    # One program handles one (row, group) feature vector of length N.
    # Grid layout: program_id(0) = row index, program_id(1) = group index.
    # Computes y = (x * rstd) * w * sigmoid(gate) where rstd = 1/sqrt(mean(x^2)+eps).
    # rstd is stored back to RAM so the backward pass can avoid recomputing the
    # reduction. The entire feature dimension N is processed in a single tile
    # of BLOCK_N lanes; this is the reason for the cap in the Python wrapper.
    X,
    O,
    Y,
    W,
    Rstd,
    N,
    stride_x0,
    stride_x1,
    stride_x2,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_y0,
    stride_y1,
    stride_y2,
    stride_w0,
    stride_w1,
    stride_rstd0,
    stride_rstd1,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)
    # Advance every tensor pointer to the start of this program's slice.
    # stride_x0 = row stride, stride_x1 = group stride, stride_x2 = feature
    # stride (similarly for O/Y). W is per-group, so only the group stride
    # applies. Note feature offsets are added below via cols * stride_*2.
    X += row * stride_x0 + group * stride_x1
    O += row * stride_o0 + group * stride_o1
    Y += row * stride_y0 + group * stride_y1
    W += group * stride_w0

    # Lane indices over the feature dim; mask zeroes out lanes past N when
    # BLOCK_N (a power of two) is larger than the true feature width.
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    # Load x as fp32 for numerically stable variance accumulation. The second
    # ``where`` is defensive — ``other=0.0`` already zeros masked lanes, but
    # being explicit makes the reduction below trivially correct.
    x = tl.load(X + cols * stride_x2, mask=mask, other=0.0).to(tl.float32)
    x = tl.where(mask, x, 0.0)
    # RMSNorm: divide by sqrt(mean(x^2) + eps). The +eps is the standard
    # numerical-stability trick that protects against division by zero when an
    # entire feature vector is zero. The reciprocal (rstd) is what is cached.
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    tl.store(Rstd + row * stride_rstd0 + group * stride_rstd1, rstd)

    # Apply per-group affine weight and Swish-style gating in one pass:
    #     y = normalised_x * w * sigmoid(gate)
    # All compute is done in fp32 and downcast on store via the implicit dtype
    # promotion of the destination pointer.
    w = tl.load(W + cols * stride_w1, mask=mask).to(tl.float32)
    gate = tl.load(O + cols * stride_o2, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w * tl.sigmoid(gate)
    tl.store(Y + cols * stride_y2, y, mask=mask)


def rms_norm_gate_fwd(x: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for fused group RMSNorm + Swish gating.

    Args:
        x: Input activations with shape ``(M, G, N)`` (rows, groups, features).
        gate: Pre-activation gate tensor of identical shape to ``x``; the
            kernel applies ``sigmoid(gate)`` element-wise to gate the output.
        weight: Per-group affine scale of shape ``(G, N)``.
        eps: Small positive constant added inside the sqrt for numerical
            stability.

    Returns:
        Tuple ``(y, rstd)``. ``y`` has the same shape and dtype as ``x``;
        ``rstd`` is shape ``(M, G)`` fp32 and stores 1/RMS for use in
        backward.

    The kernel processes the entire feature dimension N in a single Triton
    program by using a BLOCK_N tile sized up to the next power of two. If the
    rounded BLOCK_N would exceed a fixed shared-memory budget, the call is
    rejected — there is no looping fallback in this implementation.
    """
    if x.ndim != 3:
        raise ValueError(f"rms_norm_gate_fwd expects x with shape (M, G, N), got {tuple(x.shape)}")
    m, groups, width = x.shape
    y = torch.empty_like(x)
    rstd = torch.empty((m, groups), dtype=torch.float32, device=x.device)
    # Cap the per-program tile at 64 KiB worth of feature lanes so we fit in
    # shared memory regardless of dtype; round the actual width up to a power
    # of two as Triton requires for constexpr block sizes.
    max_fused_size = 65536 // x.element_size()
    block_n = min(max_fused_size, triton.next_power_of_2(width))
    if width > block_n:
        raise RuntimeError("fused group RMSNorm does not support this feature size")
    # Grid: one program per (row, group). No further sharding across N.
    _rms_norm_gate_kernel[(m, groups)](
        x,
        gate,
        y,
        weight,
        rstd,
        width,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        gate.stride(0),
        gate.stride(1),
        gate.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        weight.stride(0),
        weight.stride(1),
        rstd.stride(0),
        rstd.stride(1),
        eps,
        block_n,
    )
    return y, rstd

