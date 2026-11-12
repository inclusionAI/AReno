"""GaLore gradient low-rank projection optimizer.

Implements Algorithm 2 of "GaLore: Gradient Low-Rank Projection" (Zhao et al.,
2024, arXiv:2403.03507) for full-parameter training with low-rank optimizer
state. Each eligible 2D parameter keeps an orthonormal projector refreshed
every ``galore_refresh_interval`` steps from the SVD of the current gradient;
Adam moments live in the compact projected space and the bias-corrected update
is mapped back to the full weight, scaled by ``galore_alpha``.

Projection follows the paper's orientation: a wide matrix (``m <= n``) uses the
left singular subspace ``U[:, :rank]`` and projects with ``P.T @ grad``, so the
moments are ``rank x n``; a tall matrix uses the right subspace ``V[:, :rank]``
and projects with ``grad @ P``, giving ``m x rank`` moments. Parameters where
projection does not apply -- 1D tensors and matrices with
``min(m, n) <= rank`` -- fall back to ordinary full-space AdamW updates.

Coupling decisions worth knowing when comparing against the paper:

* Algorithm 2 writes plain Adam; like the other engine optimizers this class
  applies decoupled AdamW weight decay to the master weight before the update.
* The paper keeps moments across projector refreshes; this implementation does
  the same.
* FP32 master weights are kept per parameter and copied back into lower
  precision parameters after each step, matching ``AdamWFP32Master``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch


def galore_projector(grad: torch.Tensor, rank: int) -> torch.Tensor:
    """Return the rank-``rank`` SVD projector for a 2D gradient.

    Wide matrices (``m <= n``) return ``U[:, :rank]`` (``m x rank``); tall
    matrices return ``V[:, :rank]`` (``n x rank``). Callers pair this with
    :func:`project_gradient` / :func:`project_back`.
    """

    u, _, vh = torch.linalg.svd(grad, full_matrices=False)
    if grad.shape[0] <= grad.shape[1]:
        return u[:, :rank]
    return vh[:rank, :].transpose(0, 1).contiguous()


def _is_left_projected(grad: torch.Tensor) -> bool:
    return grad.shape[0] <= grad.shape[1]


def project_gradient(grad: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Project a 2D gradient into the compact subspace of ``proj``."""

    if _is_left_projected(grad):
        return proj.transpose(0, 1) @ grad
    return grad @ proj


def project_back(compact: torch.Tensor, proj: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Map a compact update back to the original gradient space."""

    if _is_left_projected(grad):
        return proj @ compact
    return compact @ proj.transpose(0, 1)


class GaloreAdamW(torch.optim.Optimizer):
    """Full-parameter AdamW with GaLore low-rank optimizer state.

    Constructor mirrors the engine optimizer call signature
    (:func:`areno.engine.modeling.build_optimizer` passes the same keywords);
    ``bucket_numel`` and the DP fields are accepted for call compatibility and
    the first delivery requires ``dp_size == 1``.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        bucket_numel: int | None = None,
        dp_rank: int = 0,
        dp_size: int = 1,
        dp_group: object | None = None,
        galore_rank: int = 128,
        galore_refresh_interval: int = 200,
        galore_alpha: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        del bucket_numel, dp_rank, dp_group
        if dp_size != 1:
            raise NotImplementedError("GaloreAdamW supports single-GPU training only (dp_size == 1)")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid betas: {betas}")
        if galore_rank < 1:
            raise ValueError("galore_rank must be >= 1")
        if galore_refresh_interval < 1:
            raise ValueError("galore_refresh_interval must be >= 1")
        if galore_alpha <= 0.0:
            raise ValueError("galore_alpha must be > 0")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)
        # Engine contract shared with AdamWFP32Master/AdamW8bit: the training
        # loop iterates ``optimizer.model_params`` to tag multimodal LR groups
        # and assigns ``optimizer.lr`` before every step.
        self.model_params = [
            param for group in self.param_groups for param in group["params"] if param.requires_grad
        ]
        self.galore_rank = galore_rank
        self.galore_refresh_interval = galore_refresh_interval
        self.galore_alpha = galore_alpha

    @property
    def lr(self) -> float:
        return self.param_groups[0]["lr"]

    @lr.setter
    def lr(self, value: float) -> None:
        for group in self.param_groups:
            group["lr"] = value

    def _eligible(self, grad: torch.Tensor) -> bool:
        return grad.ndim == 2 and min(grad.shape) > self.galore_rank

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                lr = float(getattr(param, "_areno_lr", group["lr"]))
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("GaloreAdamW does not support sparse gradients")
                state = self.state[param]
                if not state:
                    eligible = self._eligible(grad)
                    state["step"] = 0
                    state["galore"] = eligible
                    state["master"] = param.detach().clone().float()
                    if eligible:
                        compact_shape = (
                            (self.galore_rank, grad.shape[1])
                            if _is_left_projected(grad)
                            else (grad.shape[0], self.galore_rank)
                        )
                    else:
                        compact_shape = tuple(grad.shape)
                    state["exp_avg"] = torch.zeros(compact_shape, dtype=torch.float32, device=grad.device)
                    state["exp_avg_sq"] = torch.zeros(compact_shape, dtype=torch.float32, device=grad.device)

                t = state["step"] + 1
                master = state["master"]
                if weight_decay != 0.0:
                    master.mul_(1.0 - lr * weight_decay)
                grad32 = grad.float()

                proj = state.get("proj")
                if state["galore"]:
                    if proj is None or (t - 1) % self.galore_refresh_interval == 0:
                        proj = galore_projector(grad32, self.galore_rank)
                        state["proj"] = proj
                    compact = project_gradient(grad32, proj)
                else:
                    compact = grad32

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(compact, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(compact, compact, value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t
                update = (exp_avg / bias_correction1) / ((exp_avg_sq / bias_correction2).sqrt_().add_(eps))
                if state["galore"]:
                    update = self.galore_alpha * project_back(update, proj, grad32)
                master.add_(update, alpha=-lr)
                param.data.copy_(master.to(param.dtype))
                state["step"] = t
        return loss
