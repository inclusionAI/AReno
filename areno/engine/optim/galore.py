"""Single-device full-parameter AdamW with GaLore gradient projection.

The smaller singular-vector basis is refreshed every ``update_proj_gap``
updates of each eligible matrix. Moments remain in the projected coordinates
across refreshes, as in the original GaLore AdamW rule. Non-matrices and ranks
that would not save persistent state use ordinary FP32-master AdamW.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from areno.engine.optim.deferred_projection import DeferredProjectionMixin, backend_policy
from areno.engine.optim.master_storage import (
    BF16MasterStorage,
    decode_fp32_master_slice,
    encode_fp32_master,
)
from areno.engine.optim.recycled_subspace import LEGACY_POLICY, projector_policy, recycled_basis

BACKEND_POLICY = {
    "master": "bf16-rne-low16-packedcarry-finite-reject-v1",
    "writeback": "torch-mul-add-or-matched-cuda-fma-v1",
    "lift": "output-row-tiles-full-rank-reduction-v1",
    "gradient_lifetime": "post-global-clipping-post-consumer-release-v1",
    "transport": "bias-corrected-mean-factorized-second-clamp0-no-error-feedback-v1",
}

VOCAB_TARGET_POLICY = {"targets": "shape-eligible-except-vocab-module-identities-v1"}


class GaLoreAdamW(DeferredProjectionMixin):
    """GaLore with FP32 masters, moments and bases; no distributed projection."""

    stream_gradient_shards = False

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        rank: int = 128,
        update_proj_gap: int = 200,
        scale: float = 1.0,
        eps: float = 1e-8,
        projector_backend: str = "svd",
        projector_oversampling: int = 16,
        projector_iterations: int = 1,
        projector_seed: int = 0,
        projector_final_qr: bool = True,
        compact_master: bool = False,
        tile_rows: int = 0,
        release_gradients: bool = False,
        fused_writeback: bool = False,
        transport_moments: bool = False,
        exclude_vocab: bool = False,
    ):
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise ValueError("GaLore rank must be a positive integer")
        if isinstance(update_proj_gap, bool) or not isinstance(update_proj_gap, int) or update_proj_gap < 1:
            raise ValueError("GaLore update_proj_gap must be a positive integer")
        if not all(math.isfinite(value) for value in (lr, weight_decay, eps, scale, *betas)):
            raise ValueError("GaLore AdamW hyperparameters must be finite")
        if lr < 0 or weight_decay < 0 or eps <= 0 or scale <= 0 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError("invalid GaLore AdamW hyperparameters")
        if projector_backend not in {"svd", "recycled"}:
            raise ValueError("GaLore projector_backend must be svd or recycled")
        for name, value, minimum in (
            ("oversampling", projector_oversampling, 0),
            ("iterations", projector_iterations, 1),
            ("seed", projector_seed, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"invalid GaLore projector {name}")
        for name, value in (
            ("projector_final_qr", projector_final_qr),
            ("compact_master", compact_master),
            ("release_gradients", release_gradients),
            ("fused_writeback", fused_writeback),
            ("transport_moments", transport_moments),
            ("exclude_vocab", exclude_vocab),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
        if isinstance(tile_rows, bool) or not isinstance(tile_rows, int) or tile_rows < 0:
            raise ValueError("tile_rows must be a nonnegative integer")
        self.compact_master = compact_master
        self.tile_rows = ((tile_rows + 7) // 8) * 8
        self.release_gradients = release_gradients
        self.fused_writeback = fused_writeback
        self.transport_moments = transport_moments
        self.exclude_vocab = exclude_vocab
        self._writeback_invalid = None
        self.projector_backend = projector_backend
        self.projector_oversampling = projector_oversampling
        self.projector_iterations = projector_iterations
        self.projector_seed = projector_seed
        self.projector_final_qr = projector_final_qr
        self.model_params = [param for param in params if param.requires_grad]
        if any(param.dtype not in {torch.float32, torch.bfloat16} for param in self.model_params):
            raise TypeError("GaLore supports float32 or bfloat16 parameters")
        self.lr, self.betas, self.weight_decay, self.eps = lr, betas, weight_decay, eps
        self.rank, self.update_proj_gap, self.scale = rank, update_proj_gap, scale
        self.state: dict[torch.nn.Parameter, dict] = {}
        self.last_projection_refresh = False
        self._init_deferred_projection()

    def _checkpoint_backend_policy(self):
        return backend_policy(BACKEND_POLICY) if self._deferred_projection else BACKEND_POLICY.copy()

    def _projected(self, param: torch.Tensor) -> bool:
        if self.exclude_vocab and self._vocab_parameter(param):
            return False
        if param.ndim != 2 or self.rank >= min(param.shape):
            return False
        rows, cols = param.shape
        return self.rank * (2 * max(rows, cols) + min(rows, cols)) < 2 * param.numel()

    @staticmethod
    def _vocab_parameter(param: torch.Tensor) -> bool:
        return (
            bool(getattr(param, "_areno_galore_vocab", False))
            or getattr(param, "_areno_optimizer_role", None) == "token_embedding"
        )

    def step(self, closure=None):
        """Keep optimizer FP32 products strict even if the model enables TF32."""
        previous_precision = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            return self._step_impl(closure)
        finally:
            torch.set_float32_matmul_precision(previous_precision)

    @torch.no_grad()
    def _step_impl(self, closure=None):
        if self._deferred_projection:
            return self._step_deferred_impl(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.last_projection_refresh = False
        beta1, beta2 = self.betas
        if self.fused_writeback and self.model_params and self.model_params[0].is_cuda:
            self._writeback_invalid = torch.zeros((), dtype=torch.int32, device=self.model_params[0].device)
        for parameter_index, param in enumerate(self.model_params):
            grad = getattr(param, "main_grad", None)
            grad = param.grad if grad is None else grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("GaLore does not support sparse gradients")
            grad = grad.detach().float()
            state = self.state.setdefault(param, {})
            if not state:
                if self.compact_master and param.dtype == torch.bfloat16 and param.is_contiguous():
                    torch._assert_async(torch.isfinite(param).all(), "compact masters reject nonfinite values")
                    master = BF16MasterStorage.zeros(param.numel(), device=param.device)
                else:
                    master = param.detach().float().clone()
                state.update(step=0, master=master, basis=None)
            projected = self._projected(param)
            if projected:
                if state["step"] % self.update_proj_gap == 0:
                    if self.projector_backend == "recycled":
                        new_basis = recycled_basis(
                            grad,
                            state,
                            rank=self.rank,
                            oversampling=self.projector_oversampling,
                            iterations=self.projector_iterations,
                            seed=self.projector_seed,
                            parameter_index=parameter_index,
                            final_qr=self.projector_final_qr,
                        )
                    else:
                        left, _, right = torch.linalg.svd(grad, full_matrices=False)
                        new_basis = (
                            right[: self.rank].contiguous().clone()
                            if grad.shape[0] >= grad.shape[1]
                            else left[:, : self.rank].contiguous().clone()
                        )
                        del left, right
                    if self.transport_moments and state["basis"] is not None:
                        self._transport_history(state, new_basis, param.shape[0] >= param.shape[1])
                    state["basis"] = new_basis
                    self.last_projection_refresh = True
                basis = state["basis"]
                grad = grad @ basis.T if param.shape[0] >= param.shape[1] else basis.T @ grad
            if self.release_gradients and projected:
                # Global synchronization/clipping precedes step(); SVD/projection
                # consumed this full gradient, so only projected values remain.
                param.grad = None
                if hasattr(param, "main_grad"):
                    param.main_grad = None
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(grad)
                state["exp_avg_sq"] = torch.zeros_like(grad)
            state["step"] += 1
            step = state["step"]
            mean, variance = state["exp_avg"], state["exp_avg_sq"]
            mean.mul_(beta1).add_(grad, alpha=1 - beta1)
            variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            update = (mean / (1 - beta1**step)) / ((variance / (1 - beta2**step)).sqrt() + self.eps)
            if self.release_gradients:
                param.grad = None
                if hasattr(param, "main_grad"):
                    param.main_grad = None
                del grad
            lr = float(getattr(param, "_areno_lr", self.lr))
            if projected and self.tile_rows and param.is_contiguous():
                # Only output rows are tiled; the complete rank reduction is intact.
                for start in range(0, param.shape[0], self.tile_rows):
                    end = min(start + self.tile_rows, param.shape[0])
                    delta = update[start:end] @ basis if param.shape[0] >= param.shape[1] else basis[start:end] @ update
                    delta.mul_(self.scale)
                    self._write_update(param, state, delta, start * param.shape[1], lr)
                    del delta
            else:
                if projected:
                    update = update @ basis if param.shape[0] >= param.shape[1] else basis @ update
                    update.mul_(self.scale)
                self._write_update(param, state, update, 0, lr)
        if self._writeback_invalid is not None:
            torch._assert_async(self._writeback_invalid == 0, "nonfinite or unsupported compact writeback")
        return loss

    @staticmethod
    def _checked_encode(master: torch.Tensor):
        # Reject NaNs/infs explicitly: BF16 canonical NaN payloads cannot always
        # be represented by a single carry. Check conversion/flush behavior too.
        torch._assert_async(torch.isfinite(master).all(), "compact masters reject nonfinite values")
        model, compact = encode_fp32_master(master)
        words = master.contiguous().view(-1).view(torch.int32)
        high = (words >> 16) & 0xFFFF
        rounded = model.contiguous().view(-1).view(torch.int16).to(torch.int32) & 0xFFFF
        difference = (rounded - high) & 0xFFFF
        torch._assert_async((difference <= 1).all(), "unsupported BF16 conversion for compact master")
        return model, compact

    @torch.no_grad()
    def _write_update(self, param, state, update, offset, lr):
        model_flat = param.view(-1) if param.is_contiguous() else None
        master = state["master"]
        if not isinstance(master, BF16MasterStorage):
            if offset == 0 and update.shape == param.shape:
                master.mul_(1 - lr * self.weight_decay).add_(update, alpha=-lr)
                param.copy_(master)
            else:
                target = master.view(-1).narrow(0, offset, update.numel()).view(update.shape)
                target.mul_(1 - lr * self.weight_decay).add_(update, alpha=-lr)
                model_flat.narrow(0, offset, update.numel()).copy_(target.reshape(-1))
            return
        if self.fused_writeback and param.is_cuda:
            from areno.engine.optim.compact_writeback import compact_writeback

            compact_writeback(param, master, update, offset, 1 - lr * self.weight_decay, -lr, self._writeback_invalid)
            return
        # A bounded 4M-element decode also supports packing-only/fallback controls.
        flat_update = update.contiguous().view(-1)
        chunk_size = 4 * 1024 * 1024
        for start in range(0, flat_update.numel(), chunk_size):
            numel = min(chunk_size, flat_update.numel() - start)
            begin = offset + start
            model_slice = model_flat.narrow(0, begin, numel)
            decoded = decode_fp32_master_slice(model_slice, master, begin)
            decoded.mul_(1 - lr * self.weight_decay).add_(flat_update[start : start + numel], alpha=-lr)
            rounded, encoded = self._checked_encode(decoded)
            model_slice.copy_(rounded)
            master.low_bits.narrow(0, begin, numel).copy_(encoded.low_bits)
            if begin % 8 or (numel % 8 and begin + numel != master.numel):
                raise RuntimeError("compact writes require byte-aligned tile boundaries")
            master.round_up_bits.narrow(0, begin // 8, encoded.round_up_bits.numel()).copy_(encoded.round_up_bits)

    @staticmethod
    def _snapshot_master(param, compact):
        # Canonical FP32 checkpoint assembled on CPU without a full GPU decode.
        output = torch.empty(param.numel(), dtype=torch.float32)
        flat = param.detach().contiguous().view(-1)
        for start in range(0, param.numel(), 4 * 1024 * 1024):
            count = min(4 * 1024 * 1024, param.numel() - start)
            model_slice = flat.narrow(0, start, count).to(compact.low_bits.device)
            output[start : start + count].copy_(decode_fp32_master_slice(model_slice, compact, start).cpu())
        return output.view(param.shape)

    def _transport_history(self, state: dict, new_basis: torch.Tensor, right: bool) -> None:
        """LDAdam-inspired statistical transport, without error feedback.

        Work with bias-corrected statistics at the completed-step clock s.
        Approximate discarded cross-coordinate second moments by products of
        means, then restore the original bias factors before ordinary AdamW.
        Clamp only the resulting second moment, as specified by the rule.
        """
        step = state["step"]
        beta1, beta2 = self.betas
        correction1, correction2 = 1 - beta1**step, 1 - beta2**step
        old_mean = state["exp_avg"] / correction1
        old_second = state["exp_avg_sq"] / correction2
        if right:
            change = state["basis"] @ new_basis.T
            new_mean = old_mean @ change
            new_second = (old_second - old_mean.square()) @ change.square() + new_mean.square()
        else:
            change = new_basis.T @ state["basis"]
            new_mean = change @ old_mean
            new_second = change.square() @ (old_second - old_mean.square()) + new_mean.square()
        state["exp_avg"].copy_(new_mean * correction1)
        state["exp_avg_sq"].copy_(new_second.clamp_min(0) * correction2)
        state["transport_count"] = state.get("transport_count", 0) + 1

    def zero_grad(self, set_to_none: bool = True) -> None:
        for param in self.model_params:
            for name in ("grad", "main_grad"):
                grad = getattr(param, name, None)
                if grad is not None:
                    if set_to_none:
                        setattr(param, name, None)
                    else:
                        grad.zero_()

    def state_dict(self) -> dict:
        """Return an independent CPU snapshot, including authoritative masters."""

        if self._deferred_projection:
            self._assert_deferred_idle()
        payload = {
            "format": "areno-galore-targets-v1" if self.exclude_vocab else "areno-galore-v3",
            "backend_policy": self._checkpoint_backend_policy(),
            "projector_policy": projector_policy(self.projector_final_qr),
            "hyperparameters": {
                name: getattr(self, name)
                for name in (
                    "lr",
                    "betas",
                    "weight_decay",
                    "eps",
                    "rank",
                    "update_proj_gap",
                    "scale",
                    "projector_backend",
                    "projector_oversampling",
                    "projector_iterations",
                    "projector_seed",
                    "projector_final_qr",
                    "compact_master",
                    "tile_rows",
                    "release_gradients",
                    "fused_writeback",
                    "transport_moments",
                )
            },
            "shapes": [tuple(param.shape) for param in self.model_params],
            "state": [
                {
                    key: self._snapshot_master(param, value)
                    if isinstance(value, BF16MasterStorage)
                    else value.detach().cpu().clone()
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in self.state.get(param, {}).items()
                }
                for param in self.model_params
            ],
        }
        if self.exclude_vocab:
            payload["hyperparameters"]["exclude_vocab"] = True
            payload["projection_policy"] = VOCAB_TARGET_POLICY.copy()
            payload["projection_exclusions"] = [self._vocab_parameter(param) for param in self.model_params]
        return payload

    @torch.no_grad()
    def load_state_dict(self, payload: dict) -> None:
        if self._deferred_projection:
            self._assert_deferred_idle()
            if payload.get("format") not in {"areno-galore-v3", "areno-galore-targets-v1"}:
                raise ValueError("deferred precision requires explicit v3 policy")
            if payload.get("backend_policy") != self._checkpoint_backend_policy():
                raise ValueError("unsupported GaLore deferred backend policy")
        if "backend_policy" in payload and payload["backend_policy"] != self._checkpoint_backend_policy():
            raise ValueError("unsupported GaLore backend policy")
        targeted = payload.get("format") == "areno-galore-targets-v1"
        if payload.get("format") not in {
            "areno-galore-v1",
            "areno-galore-v2",
            "areno-galore-v3",
            "areno-galore-targets-v1",
        }:
            raise ValueError("unsupported GaLore checkpoint format")
        hyperparameters = dict(payload["hyperparameters"])
        if payload["format"] == "areno-galore-v2":
            if payload.get("projector_policy") != LEGACY_POLICY or hyperparameters.get("projector_final_qr", False):
                raise ValueError("unsupported GaLore projector policy")
            hyperparameters["projector_final_qr"] = False
        elif payload["format"] == "areno-galore-v3" or targeted:
            if payload.get("projector_policy") != projector_policy(hyperparameters.get("projector_final_qr", True)):
                raise ValueError("unsupported GaLore projector policy")
        if (payload["format"] == "areno-galore-v3" or targeted) and payload.get(
            "backend_policy"
        ) != self._checkpoint_backend_policy():
            raise ValueError("unsupported GaLore backend policy")
        if payload["shapes"] != [tuple(param.shape) for param in self.model_params]:
            raise ValueError("GaLore checkpoint parameter shapes/order do not match")
        if len(payload["state"]) != len(self.model_params):
            raise ValueError("GaLore checkpoint parameter count does not match")
        # Validate hyperparameters through the same constructor as a fresh run.
        validated = type(self)([], **hyperparameters)
        validated._deferred_projection = self._deferred_projection
        if targeted:
            expected = [self._vocab_parameter(param) for param in self.model_params]
            saved_exclusions = payload.get("projection_exclusions")
            if (
                not validated.exclude_vocab
                or payload.get("projection_policy") != VOCAB_TARGET_POLICY
                or not isinstance(saved_exclusions, list)
                or any(type(value) is not bool for value in saved_exclusions)
                or saved_exclusions != expected
            ):
                raise ValueError("GaLore vocabulary projection policy/identity mismatch")
        elif validated.exclude_vocab:
            raise ValueError("GaLore vocabulary exclusions require a versioned target checkpoint")
        restored = {}
        for parameter_index, (param, saved) in enumerate(zip(self.model_params, payload["state"], strict=True)):
            if not saved:
                continue
            projected = validated._projected(param)
            rows, cols = param.shape if param.ndim == 2 else (0, 0)
            moment_shape = (rows, validated.rank) if rows >= cols else (validated.rank, cols)
            moment_shape = moment_shape if projected else tuple(param.shape)
            basis_shape = (validated.rank, cols) if rows >= cols else (rows, validated.rank)
            if saved["master"].shape != param.shape or saved["exp_avg"].shape != moment_shape:
                raise ValueError("GaLore checkpoint master or moment shape mismatch")
            if saved["exp_avg_sq"].shape != moment_shape or int(saved["step"]) < 1:
                raise ValueError("GaLore checkpoint variance or step mismatch")
            if projected and (saved["basis"] is None or saved["basis"].shape != basis_shape):
                raise ValueError("GaLore checkpoint projection basis shape mismatch")
            if not projected and saved["basis"] is not None:
                raise ValueError("ordinary AdamW parameter must not have a projection basis")
            if projected and validated.projector_backend == "recycled":
                rng_state = saved.get("projector_rng_state")
                if not isinstance(rng_state, torch.Tensor) or rng_state.dtype != torch.uint8:
                    raise ValueError("missing or invalid GaLore projector RNG state")
                if saved.get("projector_parameter_index") != parameter_index:
                    raise ValueError("GaLore projector parameter order mismatch")
                expected_refreshes = (int(saved["step"]) - 1) // validated.update_proj_gap + 1
                if saved.get("projector_refreshes") != expected_refreshes:
                    raise ValueError("GaLore projector refresh counter mismatch")
                torch.Generator(device="cpu").set_state(rng_state.cpu())
            compact = validated.compact_master and param.dtype == torch.bfloat16 and param.is_contiguous()
            if compact:
                _, compact_state = self._checked_encode(saved["master"].cpu())
            if validated.transport_moments and projected:
                expected_transports = (int(saved["step"]) - 1) // validated.update_proj_gap
                if int(saved.get("transport_count", 0)) != expected_transports:
                    raise ValueError("GaLore transport counter mismatch")
            restored[param] = {
                key: compact_state.to(param.device)
                if key == "master" and compact
                else value.detach().cpu().clone()
                if key == "projector_rng_state"
                else value.detach().to(device=param.device, dtype=torch.float32).clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in saved.items()
            }
        for name in validated.state_dict()["hyperparameters"]:
            setattr(self, name, getattr(validated, name))
        self.exclude_vocab = validated.exclude_vocab
        self.state = restored
        self.last_projection_refresh = False
        self._writeback_invalid = None
        for param, saved in zip(self.model_params, payload["state"], strict=True):
            if saved:
                param.copy_(saved["master"])

    def state_memory_metrics(self) -> dict[str, int]:
        """Resident tensor bytes, including bases and FP32 master parameters."""

        totals = {"master_bytes": 0, "moment_bytes": 0, "basis_bytes": 0, "rng_bytes": 0}
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, BF16MasterStorage):
                    totals["master_bytes"] += value.nbytes
                elif isinstance(value, torch.Tensor):
                    category = (
                        "master_bytes"
                        if key == "master"
                        else "basis_bytes"
                        if key == "basis"
                        else "rng_bytes"
                        if key == "projector_rng_state"
                        else "moment_bytes"
                    )
                    totals[category] += value.numel() * value.element_size()
        totals["state_bytes"] = totals["moment_bytes"] + totals["basis_bytes"] + totals["rng_bytes"]
        totals["total_bytes"] = totals["state_bytes"] + totals["master_bytes"]
        if self._deferred_projection:
            totals.update(self._deferred_last)
        return totals

    def has_gradients(self) -> bool:
        return any(
            param.grad is not None or getattr(param, "main_grad", None) is not None for param in self.model_params
        )

    def offload_state(self, mode: str = "cpu", directory: str | None = None, batch_size: int = 1) -> None:
        """Honor the worker's temporary auxiliary-role swap to CPU."""

        if mode != "cpu":
            raise ValueError("GaLore supports only temporary CPU role swaps")
        self.onload_state(torch.device("cpu"))

    def onload_state(self, device: torch.device) -> None:
        for state in self.state.values():
            for key, value in state.items():
                if isinstance(value, (torch.Tensor, BF16MasterStorage)) and key != "projector_rng_state":
                    state[key] = value.to(device=device)
