"""Independent preclip recycled + transport reference, not old trajectory equivalence."""

from __future__ import annotations

import copy
import io
import os

import pytest
import torch

from areno.engine.optim.galore import GaLoreAdamW
from tests.test_galore_transport_cpu import dense_transport
from tests.test_recycled_galore_cpu import IndependentReference, reference_qr


class PreclipReference(IndependentReference):
    """Reuse independent QR/Ritz + torch AdamW, plus dense covariance transport."""

    def __init__(self, initial, *, lr, betas, decay, eps):
        super().__init__(initial, oversampling=16)
        self.lr, self.betas, self.decay = lr, betas, decay
        self.optimizer.param_groups[0].update(lr=lr, betas=betas, eps=eps)

    def _update(self, gradient):
        old_weight = self.weight.clone()
        old_basis = self.basis.clone() if self.basis is not None else None
        completed = self.step
        a = gradient.T if self.right else gradient
        if completed % 3 == 0:
            width = min(2 + self.oversampling, a.shape[0])
            noise = torch.randn(
                a.shape[0],
                width if old_basis is None else width - 2,
                generator=self.generator,
            ).to(a.device)
            if old_basis is not None:
                for _ in range(2):
                    noise -= old_basis @ (old_basis.T @ noise)
                q = reference_qr(torch.cat([old_basis, noise], dim=1))
            else:
                q = reference_qr(noise)
            q = reference_qr(a @ reference_qr(a.T @ q))
            # Only the registered QR(A.T Q)->SVD(R.T) coordinate rule is
            # authoritative. Direct compressed-B SVD is a different FP32 rule.
            _, triangular = torch.linalg.qr(a.T @ q, mode="r")
            left, _, _ = torch.linalg.svd(triangular.T, full_matrices=False, driver="gesvd" if a.is_cuda else None)
            self.basis = reference_qr(q @ left[:, :2])
            for column in range(2):
                pivot = self.basis[:, column].abs().argmax()
                if self.basis[pivot, column] < 0:
                    self.basis[:, column].neg_()
            if old_basis is not None:
                state = self.optimizer.state[self.coordinates]
                mean, second = dense_transport(
                    dict(
                        step=completed,
                        basis=old_basis.T if self.right else old_basis,
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                    ),
                    self.basis.T if self.right else self.basis,
                    self.right,
                    self.betas,
                )
                state["exp_avg"].copy_(mean)
                state["exp_avg_sq"].copy_(second)
        projected = (self.basis.T @ a).T if self.right else self.basis.T @ a
        projected.mul_(self.clip_coefficient)
        with torch.no_grad():
            self.coordinates.zero_()
        self.coordinates.grad = projected
        self.optimizer.step()
        lifted = self.basis @ self.coordinates.detach().T if self.right else self.basis @ self.coordinates.detach()
        self.weight = old_weight * (1 - self.lr * self.decay) + 1.0 * (lifted.T if self.right else lifted)
        self.step += 1


def compare(a, b):
    # Portable state_dict is CPU-resident; move only for same-device comparison.
    a = a.to(b.device)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-4)
    err = (a - b).abs()
    return dict(
        max_abs=float(err.max()),
        max_normalized=float((err / (1e-6 + 1e-4 * b.abs())).max()),
    )


def exact(a, b):
    if isinstance(a, torch.Tensor):
        assert a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    elif hasattr(a, "low_bits"):
        exact(a.low_bits, b.low_bits)
        exact(a.round_up_bits, b.round_up_bits)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            exact(x, y)
    else:
        assert a == b


def snapshot(opt):
    return copy.deepcopy(opt.state), [p.detach().clone() for p in opt.model_params]


def case(shape, dtype, stress, device="cpu"):
    os.environ["ARENO_EXPERIMENT_DEFERRED_PROJECTION"] = "1"
    gen = torch.Generator().manual_seed(593)
    initial = torch.randn(shape, generator=gen).to(device=device, dtype=dtype)
    vector_initial = torch.randn(5, generator=gen).to(device)
    p = torch.nn.Parameter(initial.clone())
    v = torch.nn.Parameter(vector_initial.clone())
    options = dict(
        lr=0.003 if stress else 1e-5,
        betas=(0.9, 0.95),
        weight_decay=0.1 if stress else 0.01,
        eps=1e-4 if stress else 1e-8,
        rank=2,
        update_proj_gap=3,
        scale=1,
        projector_backend="recycled",
        projector_oversampling=16,
        projector_seed=17,
        compact_master=True,
        tile_rows=16384,
        release_gradients=True,
        fused_writeback=True,
        transport_moments=True,
    )
    opt = GaLoreAdamW([p, v], **options)
    ref = PreclipReference(
        initial.float(),
        lr=options["lr"],
        betas=options["betas"],
        decay=options["weight_decay"],
        eps=options["eps"],
    )
    vp = torch.nn.Parameter(vector_initial.clone())
    vo = torch.optim.AdamW(
        [vp],
        lr=options["lr"],
        betas=options["betas"],
        weight_decay=options["weight_decay"],
        eps=options["eps"],
        foreach=False,
    )
    restored = None
    rows = []
    refresh = []
    for step in range(10):
        raw = torch.randn(shape, generator=gen).to(device)
        rawv = torch.randn(5, generator=gen).to(device)
        native = raw.to(dtype).float()
        norm = float(
            torch.stack((torch.linalg.vector_norm(native), torch.linalg.vector_norm(rawv))).square().sum().sqrt()
        )
        coef = min(1.0, 0.7 / (norm + 1e-6))
        ref.clip_coefficient = coef
        before = [copy.deepcopy(opt.state.get(parameter, {})) for parameter in opt.model_params]
        before_weights = [p.detach().clone(), v.detach().clone()]
        opt.begin_deferred_projection()
        (p.float().mul(raw).sum() + v.mul(rawv).sum()).backward()
        exact(before, [opt.state.get(parameter, {}) for parameter in opt.model_params])
        for x, y in zip(before_weights, (p, v), strict=True):
            assert torch.equal(x, y)
        assert p.grad is None and v.grad is None
        try:
            opt.state_dict()
        except RuntimeError as e:
            assert "empty step-boundary cache" in str(e)
        else:
            raise AssertionError("pending state snapshot accepted")
        opt.finish_deferred_backward()
        got = opt.deferred_grad_norms()["global"]
        assert abs(got - norm) <= 1e-6 + 1e-4 * abs(norm)
        opt.clip_deferred_gradients(got, 0.7)
        opt.step()
        ref.update(native)
        vp.grad = rawv * torch.tensor(coef, device=device)
        vo.step()
        vo.zero_grad(set_to_none=True)
        state = opt.state_dict()["state"][0]
        record = dict(
            step=step,
            norm=got,
            master=compare(state["master"], ref.weight),
            model=compare(p.float(), ref.weight.to(dtype).float()),
            mean=compare(state["exp_avg"], ref.optimizer.state[ref.coordinates]["exp_avg"]),
            variance=compare(state["exp_avg_sq"], ref.optimizer.state[ref.coordinates]["exp_avg_sq"]),
            basis=compare(state["basis"], ref.basis.T if ref.right else ref.basis),
            vector=compare(v, vp),
            owners=opt._deferred_last["deferred_owner_count"],
            barrier=True,
        )
        assert torch.equal(state["projector_rng_state"], ref.generator.get_state())
        if opt.last_projection_refresh:
            refresh.append(step)
        if restored is not None:
            other, op, ov = restored
            other.begin_deferred_projection()
            (op.float().mul(raw).sum() + ov.mul(rawv).sum()).backward()
            other.finish_deferred_backward()
            gn = other.deferred_grad_norms()["global"]
            other.clip_deferred_gradients(gn, 0.7)
            other.step()
            exact(opt.state_dict(), other.state_dict())
            assert torch.equal(p, op) and torch.equal(v, ov)
            record["resumed_exact"] = True
        rows.append(record)
        if step == 2:
            stream = io.BytesIO()
            torch.save(opt.state_dict(), stream)
            stream.seek(0)
            payload = torch.load(stream, weights_only=True)
            op = torch.nn.Parameter(torch.zeros_like(p))
            ov = torch.nn.Parameter(torch.zeros_like(v))
            other = GaLoreAdamW([op, ov], lr=9, rank=1)
            other.load_state_dict(payload)
            exact(opt.state_dict(), other.state_dict())
            restored = (other, op, ov)
    assert refresh == [0, 3, 6, 9]
    return dict(
        shape=shape,
        dtype=str(dtype),
        device=device,
        stress=stress,
        refresh_indices=refresh,
        rows=rows,
    )


@pytest.mark.parametrize("shape", [(9, 9), (13, 9), (9, 13)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("stress", [False, True])
def test_deferred_independent_preclip_reference_cpu(monkeypatch, shape, dtype, stress):
    monkeypatch.setenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", "1")
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        case(shape, dtype, stress, "cpu")
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
