"""Small, CPU-only GaLore reference for experiments and correctness tests.

This module intentionally has no AReno trainer or accelerator dependencies.  It
implements the matrix form of GaLore: a gradient is projected onto the leading
left (or right) singular subspace, Adam moments are kept in that subspace, and
the update is projected back to the parameter space.

The projector is computed from the first non-None gradient and then frozen.
Freezing it makes the rank sweep isolate the effect of rank rather than the
frequency or timing of SVD refreshes.  At saturated rank the reference uses the
original (identity) coordinates instead of an SVD-rotated basis.  This is
important: AdamW's second moment and epsilon are coordinate-wise, so an
arbitrary orthogonal rotation does *not* generally preserve an AdamW update.
The identity path is the uncompressed GaLore limit and gives a useful exact
AdamW correctness anchor.

Only persistent moment storage is represented by :func:`galore_state_numel`.
The projector, parameters, gradients, and temporary SVD workspace are excluded.
This is a readable reference, not a production optimizer or a memory-measuring
implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class _Projector:
    """A fixed singular subspace for one matrix-shaped parameter."""

    rows: int
    cols: int
    rank: int
    left: bool
    basis: torch.Tensor | None = None

    @property
    def saturated(self) -> bool:
        return self.rank == min(self.rows, self.cols)

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.saturated:
            # See the module docstring: rotating coordinates breaks exact
            # coordinate-wise AdamW equivalence even when no rank is lost.
            return gradient
        if self.basis is None:
            u, _, vh = torch.linalg.svd(gradient, full_matrices=False)
            if self.left:
                self.basis = u[:, : self.rank].detach()
            else:
                self.basis = vh[: self.rank, :].transpose(0, 1).detach()
        if self.left:
            return self.basis.transpose(0, 1) @ gradient
        return gradient @ self.basis

    def unproject(self, update: torch.Tensor) -> torch.Tensor:
        if self.saturated:
            return update
        if self.basis is None:
            raise RuntimeError("projector must be initialized before unproject")
        if self.left:
            return self.basis @ update
        return update @ self.basis.transpose(0, 1)


@dataclass
class _State:
    step: int
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor


class GaLoreAdamWReference:
    """Apply AdamW to fixed low-rank GaLore projections of matrix gradients.

    Args:
        params: An iterable of two-dimensional floating-point tensors or
            parameters.  Non-matrix parameters are deliberately out of scope.
        rank: Number of singular vectors retained for every parameter.  It
            must be positive and no greater than ``min(parameter.shape)``.
        lr: Learning rate.
        betas: Adam first- and second-moment decay coefficients.
        eps: AdamW denominator epsilon, applied in projected coordinates for
            the compressed path.
        weight_decay: Decoupled AdamW weight decay, applied to the original
            full-space parameters before the gradient update.

    The class intentionally implements only the small optimizer surface needed
    by the standalone sweep: ``step()``, ``zero_grad()``, and ``state``.  It
    does not implement parameter groups, checkpointing, AMP, sparse gradients,
    or periodic projector refreshes.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        *,
        rank: int,
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.01,
    ) -> None:
        self.params = list(params)
        if not self.params:
            raise ValueError("params must contain at least one matrix")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("rank must be a positive integer")
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        self.rank = rank
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state: dict[torch.Tensor, _State] = {}
        self._projectors: dict[torch.Tensor, _Projector] = {}

        for param in self.params:
            if not isinstance(param, torch.Tensor) or param.ndim != 2:
                raise ValueError("GaLoreAdamWReference accepts only 2-D tensors")
            if not param.is_floating_point():
                raise TypeError("GaLoreAdamWReference requires floating-point tensors")
            rows, cols = param.shape
            if rank > min(rows, cols):
                raise ValueError(f"rank {rank} exceeds min(parameter.shape)={min(rows, cols)}")
            self._projectors[param] = _Projector(rows, cols, rank, left=rows <= cols)

    @torch.no_grad()
    def step(self) -> None:
        """Take one AdamW step for every parameter with a dense gradient."""

        beta1, beta2 = self.betas
        for param in self.params:
            gradient = param.grad
            if gradient is None:
                continue
            if gradient.is_sparse:
                raise ValueError("sparse gradients are not supported")
            if gradient.shape != param.shape:
                raise ValueError("gradient and parameter shapes must match")

            projector = self._projectors[param]
            projected_gradient = projector.project(gradient)
            state = self.state.get(param)
            if state is None:
                state = _State(
                    step=0,
                    exp_avg=torch.zeros_like(projected_gradient),
                    exp_avg_sq=torch.zeros_like(projected_gradient),
                )
                self.state[param] = state
            state.step += 1

            # Match torch.optim.AdamW's decoupled weight-decay ordering.
            if self.weight_decay != 0:
                param.mul_(1 - self.lr * self.weight_decay)

            state.exp_avg.lerp_(projected_gradient, 1 - beta1)
            state.exp_avg_sq.mul_(beta2).addcmul_(projected_gradient, projected_gradient, value=1 - beta2)

            bias_correction1 = 1 - beta1**state.step
            bias_correction2 = 1 - beta2**state.step
            step_size = self.lr / bias_correction1
            denom = (state.exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            if projector.saturated:
                # Keep the same addcdiv operation as single-tensor AdamW.
                param.addcdiv_(state.exp_avg, denom, value=-step_size)
            else:
                update = state.exp_avg / denom
                param.add_(projector.unproject(update), alpha=-step_size)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear gradients using the same convention as ``torch.optim``."""

        for param in self.params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()


def adamw_state_numel(shape: tuple[int, int]) -> int:
    """Return the number of FP32/FP64 scalars in two full Adam moments."""

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must be a non-empty 2-D shape")
    return 2 * math.prod(shape)


def galore_state_numel(shape: tuple[int, int], rank: int) -> int:
    """Return persistent moment scalars for a matrix GaLore state.

    For ``(m, n)`` with ``m <= n``, each moment has shape ``(rank, n)``;
    otherwise it has shape ``(m, rank)``.  Thus both orientations store
    ``2 * rank * max(m, n)`` scalars.  At saturated rank this equals
    :func:`adamw_state_numel`.
    """

    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("shape must be a non-empty 2-D shape")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0 or rank > min(shape):
        raise ValueError("rank must be positive and no greater than min(shape)")
    return 2 * rank * max(shape)
