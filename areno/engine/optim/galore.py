"""GaLore AdamW: low-rank Adam moments for full-parameter training.

GaLore (Zhao et al., ICML 2024) keeps every weight trainable but stores the
Adam moments of each eligible weight matrix in a rank-``r`` subspace of its
gradient. Every ``update_proj_gap`` steps a matrix refreshes its projection
basis from the top-``r`` singular vectors of the current gradient. Every step
projects the full gradient into that basis, updates the low-rank moments, and
maps the normalized update back onto the full weight, scaled by ``scale``.

Eligible matrices are dense projection weights tagged by
``mark_linear_weight``. A fused weight such as ``[q; k; v]`` or ``[gate; up]``
is split into its logical row blocks, so each original projection keeps its
own basis, as when GaLore is applied to unfused modules. Embeddings, the LM
head, norms, biases and expert banks keep the inherited FP32-master AdamW
update. GaLore matrices keep the same lossless FP32 master as
``AdamWFP32Master``; only their Adam moments are low rank.

A basis needs the whole matrix gradient, so this optimizer supports only
unsharded optimizer state (``dp_size == 1``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist

from areno.engine.optim.adamw_fp32_master import (
    _DEFAULT_BUCKET_NUMEL,
    AdamWFP32Master,
    _MasterBucket,
    _param_grad,
)
from areno.engine.optim.master_storage import BF16MasterStorage, decode_fp32_master, encode_fp32_master

logger = logging.getLogger(__name__)

_STATE_FORMAT_VERSION = 1


@dataclass(slots=True)
class _GaLoreBlock:
    """Projection basis and low-rank moments for one logical matrix (row block)."""

    row_start: int
    rows: int
    cols: int
    # "left" projects the row space: basis (rows, r), moments (r, cols).
    # "right" projects the column space: basis (cols, r), moments (rows, r).
    # As in the reference implementation, the smaller side is projected.
    side: str
    basis: torch.Tensor | None = None
    exp_avg: torch.Tensor | None = None
    exp_avg_sq: torch.Tensor | None = None


@dataclass(slots=True)
class _GaLoreMatrix:
    """GaLore state for one model parameter split into its logical row blocks."""

    param: torch.nn.Parameter
    blocks: list[_GaLoreBlock]
    step: int = 0
    # Exact FP32 master for BF16 parameters; FP32 parameters are their own master.
    master_storage: BF16MasterStorage | None = None


def galore_blocks(param: torch.nn.Parameter, rank: int) -> list[_GaLoreBlock] | None:
    """Return the GaLore row blocks of ``param``, or ``None`` when it keeps AdamW.

    A parameter is eligible when it is a tagged 2-D linear weight and, for
    every logical block, a rank-``rank`` basis plus low-rank moments is smaller
    than the two full-rank Adam moments it replaces.
    """

    rows = getattr(param, "_areno_linear_rows", None)
    if rows is None or param.ndim != 2 or sum(rows) != param.shape[0]:
        return None
    if param.dtype not in (torch.bfloat16, torch.float32):
        return None
    cols = int(param.shape[1])
    blocks = []
    row_start = 0
    for block_rows in rows:
        small, large = sorted((int(block_rows), cols))
        if rank * (small + 2 * large) >= 2 * block_rows * cols:
            return None
        side = "right" if block_rows >= cols else "left"
        blocks.append(_GaLoreBlock(row_start=row_start, rows=int(block_rows), cols=cols, side=side))
        row_start += int(block_rows)
    return blocks


def top_singular_basis(gradient: torch.Tensor, rank: int, side: str) -> torch.Tensor:
    """Return the top-``rank`` singular vectors of ``gradient`` on ``side`` as columns.

    The left (right) singular vectors are the eigenvectors of ``G G^T``
    (``G^T G``). A symmetric eigensolver on that small Gram matrix finds the
    same subspace as a full SVD several times faster on GPUs, and it never
    forms the large singular-vector factor of the other side. Each vector is
    oriented so its largest-magnitude entry is positive, which makes the basis
    independent of the solver's sign convention and keeps moments carried
    across a refresh consistently oriented.
    """

    peak = float(gradient.abs().amax())
    if not math.isfinite(peak):
        raise FloatingPointError("GaLore projection refresh received a non-finite gradient")
    # Rescaling leaves singular vectors unchanged and keeps the Gram matrix in range.
    scaled = gradient / peak if peak > 0.0 else gradient
    gram = scaled @ scaled.transpose(0, 1) if side == "left" else scaled.transpose(0, 1) @ scaled
    _values, vectors = torch.linalg.eigh(gram)
    # eigh sorts eigenvalues in ascending order; keep the largest first.
    basis = vectors[:, -rank:].flip(1)
    pivot = basis.abs().argmax(dim=0, keepdim=True)
    return torch.where(basis.gather(0, pivot) < 0, -basis, basis).contiguous()


class GaLoreAdamW(AdamWFP32Master):
    """AdamW with GaLore low-rank moments for eligible linear weights.

    Ineligible parameters are bucketed and updated by ``AdamWFP32Master``
    exactly as without GaLore. For a GaLore matrix ``W`` with gradient ``G``
    and basis ``P`` (left side shown; the right side is the transpose), one
    step at optimizer step ``t`` with learning rate ``lr`` is::

        if (t - 1) % update_proj_gap == 0: P = top_singular_basis(G)
        R = P^T G
        m = beta1 * m + (1 - beta1) * R
        v = beta2 * v + (1 - beta2) * R * R
        N = (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
        W = W * (1 - lr * weight_decay) - lr * scale * P N

    Moments are kept across refreshes, as in the reference implementation.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float],
        weight_decay: float,
        rank: int,
        update_proj_gap: int,
        scale: float = 1.0,
        bucket_numel: int = _DEFAULT_BUCKET_NUMEL,
        dp_rank: int = 0,
        dp_size: int = 1,
        dp_group: dist.ProcessGroup | None = None,
    ):
        if rank < 1:
            raise ValueError(f"GaLore rank must be positive, got {rank}")
        if update_proj_gap < 1:
            raise ValueError(f"GaLore update_proj_gap must be positive, got {update_proj_gap}")
        if scale <= 0.0:
            raise ValueError(f"GaLore scale must be positive, got {scale}")
        if max(dp_size, 1) != 1:
            raise ValueError("GaLore needs whole-matrix gradients and supports only dp_size=1")
        params = [param for param in params if param.requires_grad]
        self.rank = int(rank)
        self.update_proj_gap = int(update_proj_gap)
        self.scale = float(scale)
        self._matrices = []
        for param in params:
            blocks = galore_blocks(param, self.rank)
            if blocks is not None:
                self._matrices.append(_GaLoreMatrix(param=param, blocks=blocks))
        self._matrix_ids = frozenset(id(matrix.param) for matrix in self._matrices)
        self._last_refreshed_matrices = 0
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            bucket_numel=bucket_numel,
            dp_rank=dp_rank,
            dp_size=dp_size,
            dp_group=dp_group,
        )
        galore_numel = sum(matrix.param.numel() for matrix in self._matrices)
        logger.info(
            "GaLore AdamW: rank=%d update_proj_gap=%d scale=%g matrices=%d galore_params=%d adamw_params=%d",
            self.rank,
            self.update_proj_gap,
            self.scale,
            sum(len(matrix.blocks) for matrix in self._matrices),
            galore_numel,
            sum(param.numel() for param in self.model_params) - galore_numel,
        )

    @torch.no_grad()
    def _build_buckets(self, params: list[torch.nn.Parameter], bucket_numel: int) -> list[_MasterBucket]:
        """Bucket only the parameters that keep the plain AdamW update."""

        return super()._build_buckets([param for param in params if id(param) not in self._matrix_ids], bucket_numel)

    @torch.no_grad()
    def step(self, closure=None):
        """Update bucketed tensors with AdamW, then each GaLore matrix."""

        if closure is not None:
            with torch.enable_grad():
                closure()
        super().step()
        self._last_refreshed_matrices = 0
        with _full_precision_fp32_matmul():
            for matrix in self._matrices:
                gradient = _param_grad(matrix.param)
                if gradient is None:
                    continue
                self._step_matrix(matrix, gradient)
                # Release the full gradient as soon as its matrix is updated.
                matrix.param.grad = None
                if isinstance(getattr(matrix.param, "main_grad", None), torch.Tensor):
                    matrix.param.main_grad = None
        return None

    @torch.no_grad()
    def _step_matrix(self, matrix: _GaLoreMatrix, gradient: torch.Tensor) -> None:
        """Apply one GaLore AdamW update to every row block of one parameter."""

        self._ensure_matrix_state(matrix)
        beta1, beta2 = self.betas
        matrix.step += 1
        refresh = (matrix.step - 1) % self.update_proj_gap == 0
        bias_correction1 = 1.0 - beta1**matrix.step
        bias_correction2_sqrt = (1.0 - beta2**matrix.step) ** 0.5
        effective_lr = float(getattr(matrix.param, "_areno_lr", self.lr))
        step_size = effective_lr * self.scale / bias_correction1
        master = self._matrix_master(matrix)
        if self.weight_decay != 0.0:
            master.mul_(1.0 - effective_lr * self.weight_decay)
        gradient = gradient.detach()
        for block in matrix.blocks:
            block_grad = gradient.narrow(0, block.row_start, block.rows).to(dtype=torch.float32)
            if refresh:
                block.basis = top_singular_basis(block_grad, self.rank, block.side)
                self._last_refreshed_matrices += 1
            assert block.basis is not None
            if block.side == "left":
                low_rank = block.basis.transpose(0, 1) @ block_grad
            else:
                low_rank = block_grad @ block.basis
            if block.exp_avg is None or block.exp_avg_sq is None:
                block.exp_avg = torch.zeros_like(low_rank)
                block.exp_avg_sq = torch.zeros_like(low_rank)
            block.exp_avg.mul_(beta1).add_(low_rank, alpha=1.0 - beta1)
            block.exp_avg_sq.mul_(beta2).addcmul_(low_rank, low_rank, value=1.0 - beta2)
            normalized = block.exp_avg / block.exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(self.eps)
            weight = master.narrow(0, block.row_start, block.rows)
            if block.side == "left":
                weight.addmm_(block.basis, normalized, alpha=-step_size)
            else:
                weight.addmm_(normalized, block.basis.transpose(0, 1), alpha=-step_size)
        self._commit_matrix_master(matrix, master)

    def _matrix_master(self, matrix: _GaLoreMatrix) -> torch.Tensor:
        """Return the logical FP32 master of one GaLore parameter."""

        param = matrix.param.detach()
        if param.dtype == torch.float32:
            # FP32 parameters are their own master and are updated in place.
            return param
        storage = matrix.master_storage
        if storage is None:
            # Nothing committed yet: the master is exactly the BF16 weight.
            return param.to(dtype=torch.float32)
        if storage.low_bits.device != param.device:
            # Checkpointing may read CPU-offloaded state next to device weights.
            storage = storage.to(param.device)
        return decode_fp32_master(param, storage)

    def _commit_matrix_master(self, matrix: _GaLoreMatrix, master: torch.Tensor) -> None:
        """Write an updated FP32 master back into the BF16 weight and its low bits."""

        if matrix.param.dtype == torch.bfloat16:
            model_values, matrix.master_storage = encode_fp32_master(master)
            matrix.param.detach().copy_(model_values)

    def _ensure_matrix_state(self, matrix: _GaLoreMatrix) -> None:
        """Bring offloaded GaLore state back to the parameter's device."""

        self._move_matrix_state(matrix, matrix.param.device)

    @staticmethod
    def _move_matrix_state(matrix: _GaLoreMatrix, device: torch.device | str) -> None:
        if matrix.master_storage is not None and matrix.master_storage.low_bits.device != torch.device(device):
            matrix.master_storage = matrix.master_storage.to(device)
        for block in matrix.blocks:
            for name in ("basis", "exp_avg", "exp_avg_sq"):
                value = getattr(block, name)
                if value is not None and value.device != torch.device(device):
                    setattr(block, name, value.to(device=device))

    def clear_state(self) -> None:
        """Drop AdamW and GaLore state and reset step counters."""

        super().clear_state()
        for matrix in self._matrices:
            matrix.step = 0
            matrix.master_storage = None
            for block in matrix.blocks:
                block.basis = None
                block.exp_avg = None
                block.exp_avg_sq = None
        self._last_refreshed_matrices = 0

    def configure_state_offload(self, mode: str, directory: str | None = None, batch_size: int = 1) -> None:
        """Accept CPU residency only; GaLore state has no disk-streaming path."""

        if mode == "disk":
            raise ValueError("GaLoreAdamW supports optimizer state offload to cpu only, not disk")
        super().configure_state_offload(mode=mode, directory=directory, batch_size=batch_size)

    @torch.no_grad()
    def offload_state(self, mode: str = "cpu", directory: str | None = None, batch_size: int = 1) -> None:
        """Move AdamW buckets and GaLore state to CPU."""

        super().offload_state(mode=mode, directory=directory, batch_size=batch_size)
        for matrix in self._matrices:
            self._move_matrix_state(matrix, "cpu")

    @torch.no_grad()
    def onload_state(self, device: torch.device) -> None:
        """Move AdamW buckets and GaLore state back onto ``device``."""

        super().onload_state(device)
        for matrix in self._matrices:
            self._move_matrix_state(matrix, device)

    def state_memory_metrics(self) -> dict[str, int]:
        """Return persistent optimizer-state bytes split into GaLore and AdamW parts."""

        projection_bytes = moment_bytes = master_bytes = 0
        for matrix in self._matrices:
            if matrix.master_storage is not None:
                master_bytes += matrix.master_storage.nbytes
            for block in matrix.blocks:
                projection_bytes += _nbytes(block.basis)
                moment_bytes += _nbytes(block.exp_avg) + _nbytes(block.exp_avg_sq)
        adamw_bytes = sum(
            _nbytes(bucket.exp_avg)
            + _nbytes(bucket.exp_avg_sq)
            + (bucket.master_storage.nbytes if bucket.master_storage is not None else 0)
            for bucket in self.buckets
        )
        return {
            "projection_bytes": projection_bytes,
            "moment_bytes": moment_bytes,
            "master_bytes": master_bytes,
            "adamw_state_bytes": adamw_bytes,
            "state_bytes": projection_bytes + moment_bytes + master_bytes + adamw_bytes,
            # FP32 Adam moments the GaLore matrices would need without projection.
            "full_rank_moment_bytes": 8 * sum(matrix.param.numel() for matrix in self._matrices),
            "matrices": sum(len(matrix.blocks) for matrix in self._matrices),
            "refreshed_matrices": self._last_refreshed_matrices,
        }

    def state_dict(self) -> dict:
        """Return AdamW bucket state plus GaLore bases, moments, steps and settings."""

        state = super().state_dict()
        parameter_index = {id(param): index for index, param in enumerate(self.model_params)}
        state["galore"] = {
            "format_version": _STATE_FORMAT_VERSION,
            "rank": self.rank,
            "update_proj_gap": self.update_proj_gap,
            "scale": self.scale,
            "matrices": [
                {
                    "parameter_index": parameter_index[id(matrix.param)],
                    "shape": list(matrix.param.shape),
                    "step": matrix.step,
                    # Like AdamWFP32Master, checkpoint the logical FP32 master
                    # so a restore reproduces the weights bit for bit.
                    "master": self._matrix_master(matrix).detach().to(device="cpu", copy=True),
                    "blocks": [
                        {
                            "rows": block.rows,
                            "cols": block.cols,
                            "side": block.side,
                            "basis": _optional_cpu_copy(block.basis),
                            "exp_avg": _optional_cpu_copy(block.exp_avg),
                            "exp_avg_sq": _optional_cpu_copy(block.exp_avg_sq),
                        }
                        for block in matrix.blocks
                    ],
                }
                for matrix in self._matrices
            ],
        }
        return state

    @torch.no_grad()
    def load_state_dict(self, state_dict: dict) -> None:
        """Restore a ``state_dict`` written by a GaLoreAdamW with the same settings."""

        saved = state_dict.get("galore")
        if not isinstance(saved, dict):
            raise ValueError("optimizer state has no GaLore section; it was not saved by GaLoreAdamW")
        if saved.get("format_version") != _STATE_FORMAT_VERSION:
            raise ValueError(f"unsupported GaLore state format version: {saved.get('format_version')}")
        settings = {"rank": self.rank, "update_proj_gap": self.update_proj_gap, "scale": self.scale}
        for name, value in settings.items():
            if saved.get(name) != value:
                raise ValueError(f"GaLore {name} mismatch: checkpoint has {saved.get(name)}, optimizer has {value}")
        matrices = saved.get("matrices", [])
        if len(matrices) != len(self._matrices):
            raise ValueError(f"checkpoint has {len(matrices)} GaLore matrices, optimizer has {len(self._matrices)}")
        parameter_index = {id(param): index for index, param in enumerate(self.model_params)}
        for saved_matrix, matrix in zip(matrices, self._matrices, strict=True):
            layout = [(block["rows"], block["cols"], block["side"]) for block in saved_matrix["blocks"]]
            expected = [(block.rows, block.cols, block.side) for block in matrix.blocks]
            if (
                saved_matrix["parameter_index"] != parameter_index[id(matrix.param)]
                or list(saved_matrix["shape"]) != list(matrix.param.shape)
                or layout != expected
            ):
                raise ValueError(f"GaLore matrix layout mismatch for parameter {parameter_index[id(matrix.param)]}")
        super().load_state_dict(state_dict)
        for saved_matrix, matrix in zip(matrices, self._matrices, strict=True):
            device = matrix.param.device
            matrix.step = int(saved_matrix["step"])
            master = saved_matrix["master"].detach().to(device=device, dtype=torch.float32)
            if matrix.param.dtype == torch.bfloat16:
                model_values, matrix.master_storage = encode_fp32_master(master.contiguous())
                matrix.param.detach().copy_(model_values)
            else:
                matrix.param.detach().copy_(master)
                matrix.master_storage = None
            for saved_block, block in zip(saved_matrix["blocks"], matrix.blocks, strict=True):
                for name in ("basis", "exp_avg", "exp_avg_sq"):
                    value = saved_block[name]
                    setattr(block, name, None if value is None else value.detach().to(device=device).clone())


@contextmanager
def _full_precision_fp32_matmul():
    """Run optimizer projections in IEEE FP32 even when workers enable TF32."""

    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def _nbytes(value: torch.Tensor | None) -> int:
    return 0 if value is None else value.numel() * value.element_size()


def _optional_cpu_copy(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach().to(device="cpu", copy=True)


__all__ = ["GaLoreAdamW", "galore_blocks", "top_singular_basis"]
