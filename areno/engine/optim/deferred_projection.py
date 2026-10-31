"""Private single-microbatch deferred-gradient prototype; no public API contract."""

from __future__ import annotations

import os

import torch

from areno.engine.optim.master_storage import BF16MasterStorage
from areno.engine.optim.recycled_subspace import recycled_basis


def backend_policy(original):
    return {
        **original,
        "gradient_lifetime": "single-micro-native-leaf-postaccum-projection-release-v1",
        "basis_input": "preclip-native-leaf-promoted-fp32-v1",
        "clipping_order": "full-owner-vectornorm-ordered-global-reduce-then-scale-coordinates-v1",
        "state_publication": "after-whole-backward-and-global-clip-v1",
    }


class DeferredProjectionMixin:
    def _init_deferred_projection(self):
        self._deferred_projection = os.getenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION") == "1"
        self._fused_scale_writeback = (
            self._deferred_projection
            and os.getenv("ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK") == "1"
            and type(self.scale) in (bool, int, float)
            and (not isinstance(self.scale, int) or self.scale < 2**64)
        )
        self._deferred_active = False
        self._deferred_ready = False
        self._deferred_pending = {}
        self._deferred_versions = []
        self._deferred_last = {}
        self._deferred_handles = []
        if self._deferred_projection:
            for index, parameter in enumerate(self.model_params):

                def observed(param, index=index):
                    self._observe_deferred_gradient(param, index)

                self._deferred_handles.append(parameter.register_post_accumulate_grad_hook(observed))

    def _deferred_signature(self):
        return (
            id(self.state),
            tuple((id(p), id(s), tuple((k, id(v)) for k, v in s.items())) for p, s in self.state.items()),
        )

    def _assert_deferred_idle(self):
        if self._deferred_active or self._deferred_ready or self._deferred_pending:
            raise RuntimeError("deferred projection requires an empty step-boundary cache")

    def begin_deferred_projection(self, *, micro_batches=1, reentrant=False):
        if not self._deferred_projection:
            raise RuntimeError("deferred projection is disabled")
        if micro_batches != 1 or reentrant:
            raise ValueError("deferred prototype requires one nonreentrant microbatch")
        self._assert_deferred_idle()
        if any(p.grad is not None or getattr(p, "main_grad", None) is not None for p in self.model_params):
            raise RuntimeError("deferred backward requires cleared leaf/main gradients")
        tensors = list(self.model_params)
        for state in self.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    tensors.append(value)
                elif isinstance(value, BF16MasterStorage):
                    tensors.extend((value.low_bits, value.round_up_bits))
        self._deferred_versions = [(tensor, tensor._version) for tensor in tensors]
        self._deferred_signature_before = self._deferred_signature()
        self._deferred_active = True

    @torch.no_grad()
    def _observe_deferred_gradient(self, param, index):
        if not self._deferred_active:
            raise RuntimeError("deferred leaf hook outside the supported backward scope")
        if param in self._deferred_pending:
            raise RuntimeError("deferred owner hook fired more than once; partial/accumulated backward unsupported")
        if param.grad is None or param.grad.is_sparse:
            raise RuntimeError("deferred prototype requires dense native leaf gradients")
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            gradient = param.grad.detach().float()
            norm = torch.linalg.vector_norm(gradient)
            projected = self._projected(param)
            existing = self.state.get(param, {})
            refresh = projected and existing.get("step", 0) % self.update_proj_gap == 0
            updates = {}
            basis = existing.get("basis")
            if refresh:
                if self.projector_backend == "recycled":
                    scratch = {
                        k: existing[k]
                        for k in ("basis", "projector_rng_state", "projector_parameter_index", "projector_refreshes")
                        if k in existing
                    }
                    basis = recycled_basis(
                        gradient,
                        scratch,
                        rank=self.rank,
                        oversampling=self.projector_oversampling,
                        iterations=self.projector_iterations,
                        seed=self.projector_seed,
                        parameter_index=index,
                        final_qr=self.projector_final_qr,
                    )
                    updates = {k: v for k, v in scratch.items() if k != "basis"}
                else:
                    left, _, right = torch.linalg.svd(gradient, full_matrices=False)
                    basis = (
                        right[: self.rank].contiguous().clone()
                        if gradient.shape[0] >= gradient.shape[1]
                        else left[:, : self.rank].contiguous().clone()
                    )
            coordinates = (
                (gradient @ basis.T if param.shape[0] >= param.shape[1] else basis.T @ gradient)
                if projected
                else gradient
            )
            self._deferred_pending[param] = dict(
                norm=norm,
                gradient=coordinates,
                projected=projected,
                basis=basis,
                refresh=bool(refresh),
                projector_updates=updates,
                parameter_index=index,
            )
            param.grad = None
        finally:
            torch.set_float32_matmul_precision(previous)

    def finish_deferred_backward(self):
        if not self._deferred_active or self._deferred_ready:
            raise RuntimeError("deferred backward was not started or was already finished")
        if self._deferred_signature() != self._deferred_signature_before:
            raise RuntimeError("persistent optimizer state published before backward barrier")
        if any(tensor._version != version for tensor, version in self._deferred_versions):
            raise RuntimeError("model/optimizer tensor mutated before backward barrier")
        if any(p.grad is not None or getattr(p, "main_grad", None) is not None for p in self.model_params):
            raise RuntimeError("deferred backward retained full leaf/main gradient")
        rows = list(self._deferred_pending.values())
        self._deferred_last = {
            "deferred_owner_count": len(rows),
            "deferred_owners_once": 1,
            "deferred_barrier_versions_unchanged": 1,
            "deferred_full_leaf_bytes_after_backward": 0,
            "deferred_pending_projected_bytes": sum(
                r["gradient"].numel() * r["gradient"].element_size() for r in rows if r["projected"]
            ),
            "deferred_pending_fallback_bytes": sum(
                r["gradient"].numel() * r["gradient"].element_size() for r in rows if not r["projected"]
            ),
            "deferred_norm_scalar_bytes": sum(r["norm"].element_size() for r in rows),
        }
        self._deferred_versions.clear()
        self._deferred_active = False
        self._deferred_ready = True

    def deferred_grad_norms(self, groups=()):
        if not self._deferred_ready or groups:
            raise RuntimeError("deferred prototype requires completed backward without multimodal groups")
        values = [self._deferred_pending[p]["norm"] for p in self.model_params if p in self._deferred_pending]
        if not values:
            return {"global": 0.0}
        # The only host observation is after all owner norm scalars are reduced.
        return {"global": float(torch.stack(values).float().square().sum().sqrt().cpu())}

    @torch.no_grad()
    def clip_deferred_gradients(self, grad_norm, max_norm):
        if not self._deferred_ready:
            raise RuntimeError("deferred clip before backward barrier")
        if grad_norm <= 0:
            return
        coefficient = float(max_norm) / (float(grad_norm) + 1e-6)
        if coefficient >= 1:
            return
        values = [self._deferred_pending[p]["gradient"] for p in self.model_params if p in self._deferred_pending]
        if values:
            scalar = torch.tensor(coefficient, device=values[0].device, dtype=torch.float32)
            for value in values:
                value.mul_(scalar)

    @torch.no_grad()
    def _step_deferred_impl(self, closure=None):
        if closure is not None:
            raise ValueError("deferred prototype does not support closures")
        if not self._deferred_ready:
            raise RuntimeError("deferred optimizer step before completed backward/global clip")
        self.last_projection_refresh = False
        beta1, beta2 = self.betas
        if self.fused_writeback and self.model_params and self.model_params[0].is_cuda:
            self._writeback_invalid = torch.zeros((), dtype=torch.int32, device=self.model_params[0].device)
        for param in self.model_params:
            row = self._deferred_pending.pop(param, None)
            if row is None:
                continue
            gradient = row["gradient"]
            state = self.state.setdefault(param, {})
            if not state:
                if self.compact_master and param.dtype == torch.bfloat16 and param.is_contiguous():
                    torch._assert_async(torch.isfinite(param).all(), "compact masters reject nonfinite values")
                    master = BF16MasterStorage.zeros(param.numel(), device=param.device)
                else:
                    master = param.detach().float().clone()
                state.update(step=0, master=master, basis=None)
            projected = row["projected"]
            if row["refresh"]:
                if self.transport_moments and state["basis"] is not None:
                    self._transport_history(state, row["basis"], param.shape[0] >= param.shape[1])
                state["basis"] = row["basis"]
                state.update(row["projector_updates"])
                self.last_projection_refresh = True
            basis = state["basis"]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(gradient)
                state["exp_avg_sq"] = torch.zeros_like(gradient)
            state["step"] += 1
            step = state["step"]
            mean, variance = state["exp_avg"], state["exp_avg_sq"]
            mean.mul_(beta1).add_(gradient, alpha=1 - beta1)
            variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
            update = (mean / (1 - beta1**step)) / ((variance / (1 - beta2**step)).sqrt() + self.eps)
            lr = float(getattr(param, "_areno_lr", self.lr))
            if projected and self.tile_rows and param.is_contiguous():
                for start in range(0, param.shape[0], self.tile_rows):
                    end = min(start + self.tile_rows, param.shape[0])
                    delta = update[start:end] @ basis if param.shape[0] >= param.shape[1] else basis[start:end] @ update
                    self._write_projected_update(param, state, delta, start * param.shape[1], lr)
                    del delta
            else:
                if projected:
                    update = update @ basis if param.shape[0] >= param.shape[1] else basis @ update
                    self._write_projected_update(param, state, update, 0, lr)
                else:
                    self._write_update(param, state, update, 0, lr)
            del update, gradient, row
        if self._writeback_invalid is not None:
            torch._assert_async(self._writeback_invalid == 0, "nonfinite or unsupported compact writeback")
        self.abort_deferred_projection()
        return None

    def abort_deferred_projection(self):
        self._deferred_pending.clear()
        self._deferred_versions.clear()
        self._deferred_active = False
        self._deferred_ready = False

    def _write_projected_update(self, param, state, update, offset, lr):
        if (
            self._fused_scale_writeback
            and self.fused_writeback
            and param.is_cuda
            and isinstance(state["master"], BF16MasterStorage)
        ):
            from areno.engine.optim.compact_writeback import compact_writeback

            compact_writeback(
                param,
                state["master"],
                update,
                offset,
                1 - lr * self.weight_decay,
                -lr,
                self._writeback_invalid,
                scale=self.scale,
            )
        else:
            update.mul_(self.scale)
            self._write_update(param, state, update, offset, lr)
