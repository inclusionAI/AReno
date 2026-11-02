"""Independent projected-AdamW and lifecycle tests for recycled Ritz refresh."""

from __future__ import annotations

import copy

import pytest
import torch

from areno.engine.config import OptimizerConfig
from areno.engine.optim.galore import GaLoreAdamW


def reference_qr(x):
    q, r = torch.linalg.qr(x)
    return q * torch.where(r.diag() < 0, -1.0, 1.0)


class IndependentReference:
    """Registered QR/Ritz sequence + torch AdamW; no production helper imports."""

    def __init__(self, initial, oversampling=2):
        self.weight = initial.clone()
        self.right = initial.shape[0] >= initial.shape[1]
        shape = (initial.shape[0], 2) if self.right else (2, initial.shape[1])
        self.coordinates = torch.nn.Parameter(initial.new_zeros(shape))
        self.optimizer = torch.optim.AdamW(
            [self.coordinates], lr=0.003, betas=(0.8, 0.9), eps=1e-8, weight_decay=0, foreach=False
        )
        self.generator = torch.Generator().manual_seed(17)
        self.basis = None
        self.step = 0
        self.oversampling = oversampling

    def update(self, gradient):
        previous = torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision("highest")
            self._update(gradient)
        finally:
            torch.set_float32_matmul_precision(previous)

    def _update(self, gradient):
        a = gradient.T if self.right else gradient
        if self.step % 3 == 0:
            width = min(2 + self.oversampling, a.shape[0])
            noise = torch.randn(a.shape[0], width if self.basis is None else width - 2, generator=self.generator).to(
                a.device
            )
            if self.basis is not None:
                for _ in range(2):
                    noise -= self.basis @ (self.basis.T @ noise)
                q = reference_qr(torch.cat([self.basis, noise], dim=1))
            else:
                q = reference_qr(noise)
            q = reference_qr(a @ reference_qr(a.T @ q))
            # Follow the registered factorization rule; direct B SVD is only
            # mathematically equivalent and can perturb Adam coordinates in FP32.
            _, triangular = torch.linalg.qr(a.T @ q, mode="r")
            left, _, _ = torch.linalg.svd(
                triangular.T, full_matrices=False, driver="gesvd" if triangular.is_cuda else None
            )
            self.basis = reference_qr(q @ left[:, :2])
            for column in range(2):
                pivot = self.basis[:, column].abs().argmax()
                if self.basis[pivot, column] < 0:
                    self.basis[:, column].neg_()
        projected = (self.basis.T @ a).T if self.right else self.basis.T @ a
        with torch.no_grad():
            self.coordinates.zero_()
        self.coordinates.grad = projected
        self.optimizer.step()
        lifted = self.basis @ self.coordinates.detach().T if self.right else self.basis @ self.coordinates.detach()
        if self.right:
            lifted = lifted.T
        self.weight = self.weight * (1 - 0.003 * 0.1) + 0.7 * lifted
        self.step += 1


@pytest.fixture(params=["highest", "high"])
def worker_precision(request):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(request.param)
    yield request.param
    torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("shape", [(11, 7), (7, 11), (9, 9)])
@pytest.mark.parametrize("oversampling", [0, 2, 20])
def test_new_rule_independent_reference(shape, oversampling, device, worker_precision):
    generator = torch.Generator().manual_seed(593)
    param = torch.nn.Parameter(torch.randn(shape, generator=generator).to(device))
    reference = IndependentReference(param.detach(), oversampling)
    optim = GaLoreAdamW(
        [param],
        lr=0.003,
        betas=(0.8, 0.9),
        weight_decay=0.1,
        rank=2,
        update_proj_gap=3,
        scale=0.7,
        projector_backend="recycled",
        projector_oversampling=oversampling,
        projector_seed=17,
    )
    for step in range(10):
        gradient = torch.randn(shape, generator=generator).to(device)
        param.grad = gradient.clone()
        optim.step()
        reference.update(gradient)
        torch.testing.assert_close(param, reference.weight, atol=1e-6, rtol=1e-4)
        basis = optim.state[param]["basis"].T if reference.right else optim.state[param]["basis"]
        torch.testing.assert_close(basis, reference.basis, atol=1e-6, rtol=1e-4)
        expected_state = reference.optimizer.state[reference.coordinates]
        torch.testing.assert_close(optim.state[param]["exp_avg"], expected_state["exp_avg"], atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(optim.state[param]["exp_avg_sq"], expected_state["exp_avg_sq"], atol=1e-6, rtol=1e-4)
        assert optim.last_projection_refresh == (step % 3 == 0)
        assert torch.get_float32_matmul_precision() == worker_precision


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
@pytest.mark.parametrize("save_after", [0, 1, 3, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_refresh_resume_exact_and_global_rng_untouched(save_after, dtype, device):
    generator = torch.Generator().manual_seed(99)
    params = [
        torch.nn.Parameter(torch.randn(13, 9, generator=generator).to(device=device, dtype=dtype)) for _ in range(2)
    ]
    optim = GaLoreAdamW(params, lr=0.003, rank=2, update_proj_gap=3, projector_backend="recycled", projector_seed=19)
    gradients = [torch.randn(13, 9, generator=generator).to(device) for _ in range(10)]
    for index, gradient in enumerate(gradients[:save_after]):
        params[0].main_grad = gradient
        params[1].main_grad = gradient if index % 2 == 0 else None
        optim.step()
        optim.zero_grad()
    snapshot = optim.state_dict()
    targets = [torch.nn.Parameter(p.detach().clone()) for p in params]
    resumed = GaLoreAdamW(targets, lr=9.0, rank=1)
    resumed.load_state_dict(snapshot)
    assert resumed.projector_backend == "recycled" and resumed.projector_seed == 19
    rng_before = torch.get_rng_state().clone()
    for index in range(save_after, 10):
        for original, target in zip(params, targets, strict=True):
            gradient = gradients[index] if original is params[0] or index % 2 == 0 else None
            original.main_grad, target.main_grad = gradient, gradient
        optim.step()
        resumed.step()
        for original, target in zip(params, targets, strict=True):
            torch.testing.assert_close(original, target, atol=0, rtol=0)
            for key, value in optim.state[original].items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(value, resumed.state[target][key], atol=0, rtol=0)
                else:
                    assert value == resumed.state[target][key]
        optim.zero_grad()
        resumed.zero_grad()
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert snapshot["state"][0].get("step", 0) == save_after
    assert optim.state[params[0]]["projector_rng_state"].dtype == torch.uint8


@pytest.mark.parametrize("kind", ["zero", "rank_one", "full"])
@pytest.mark.parametrize("shape", [(19, 7), (7, 19), (8, 8)])
def test_shapes_orthogonality_rank_deficiency_and_small_svd(monkeypatch, shape, kind):
    generator = torch.Generator().manual_seed(57)
    param = torch.nn.Parameter(torch.zeros(shape))
    gradient = torch.randn(shape, generator=generator)
    if kind == "zero":
        gradient.zero_()
    elif kind == "rank_one":
        gradient = torch.randn(shape[0], 1, generator=generator) @ torch.randn(1, shape[1], generator=generator)
    calls = []
    original_svd = torch.linalg.svd

    def observed_svd(matrix, *args, **kwargs):
        calls.append(tuple(matrix.shape))
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", observed_svd)
    optim = GaLoreAdamW(
        [param], lr=0.001, rank=2, update_proj_gap=1, projector_backend="recycled", projector_oversampling=2
    )
    for _ in range(3):
        param.grad = gradient.clone()
        optim.step()
        basis = optim.state[param]["basis"]
        basis = basis.T if shape[0] >= shape[1] else basis
        assert basis.shape == (min(shape), 2)
        torch.testing.assert_close(basis.T @ basis, torch.eye(2), atol=1e-6, rtol=1e-4)
        assert torch.isfinite(param).all()
    assert calls == [(4, 4)] * 3
    assert optim.state_memory_metrics()["rng_bytes"] > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("projector_policy", {}),
        ("projector_rng_state", torch.zeros(4)),
        ("projector_parameter_index", 5),
        ("projector_refreshes", 20),
    ],
)
def test_malformed_checkpoint_is_atomic(field, value):
    param = torch.nn.Parameter(torch.randn(13, 9))
    optim = GaLoreAdamW([param], lr=0.001, rank=2, projector_backend="recycled")
    param.grad = torch.randn_like(param)
    optim.step()
    saved = copy.deepcopy(optim.state_dict())
    target = saved if field == "projector_policy" else saved["state"][0]
    target[field] = value
    before = param.detach().clone()
    with pytest.raises(ValueError):
        optim.load_state_dict(saved)
    assert torch.equal(param, before)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"galore_projector_backend": "bad"},
        {"galore_projector_oversampling": -1},
        {"galore_projector_iterations": 0},
        {"galore_projector_seed": -1},
    ],
)
def test_invalid_engine_config(kwargs):
    with pytest.raises(ValueError):
        OptimizerConfig(**kwargs)


def test_old_svd_checkpoint_resets_new_options():
    p = torch.nn.Parameter(torch.randn(11, 7))
    original = GaLoreAdamW([p], lr=0.001, rank=2)
    p.grad = torch.randn_like(p)
    original.step()
    saved = original.state_dict()
    saved["format"] = "areno-galore-v1"
    saved.pop("projector_policy")
    for key in list(saved["hyperparameters"]):
        if key.startswith("projector_"):
            saved["hyperparameters"].pop(key)
    resumed = GaLoreAdamW([p], lr=0.003, rank=2, projector_backend="recycled")
    resumed.load_state_dict(saved)
    assert resumed.projector_backend == "svd"


def test_engine_factory_wires_recycled_options():
    from types import SimpleNamespace

    from areno.engine.modeling import build_optimizer

    config = OptimizerConfig(
        galore_rank=2,
        galore_projector_backend="recycled",
        galore_projector_oversampling=3,
        galore_projector_iterations=2,
        galore_projector_seed=31,
    )
    param = torch.nn.Parameter(torch.zeros(11, 7))
    optim = build_optimizer([param], config, SimpleNamespace(world_size=1, dp_size=1))
    assert (
        optim.projector_backend,
        optim.projector_oversampling,
        optim.projector_iterations,
        optim.projector_seed,
    ) == ("recycled", 3, 2, 31)


@pytest.mark.parametrize("shape,rank", [((7,), 2), ((4, 5), 4), ((4, 4), 3)])
def test_recycled_backend_keeps_ordinary_adamw_fallback(shape, rank):
    param = torch.nn.Parameter(torch.ones(shape))
    reference = torch.nn.Parameter(torch.ones(shape))
    actual = GaLoreAdamW([param], lr=0.003, rank=rank, weight_decay=0.1, projector_backend="recycled")
    expected = torch.optim.AdamW([reference], lr=0.003, betas=(0.9, 0.95), weight_decay=0.1)
    for _ in range(3):
        gradient = torch.randn(shape)
        param.grad, reference.grad = gradient.clone(), gradient.clone()
        actual.step()
        expected.step()
        torch.testing.assert_close(param, reference, atol=1e-6, rtol=1e-4)
    assert actual.state[param]["basis"] is None
    assert "projector_rng_state" not in actual.state[param]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_worker_tf32_setting_does_not_corrupt_basis_and_is_restored():
    previous = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        generator = torch.Generator().manual_seed(593)
        param = torch.nn.Parameter(torch.randn(512, 256, generator=generator).cuda())
        optim = GaLoreAdamW([param], lr=0.001, rank=16, projector_backend="recycled", update_proj_gap=1)
        for _ in range(3):
            param.grad = torch.randn(512, 256, generator=generator).cuda()
            optim.step()
            basis = optim.state[param]["basis"].cpu()
            torch.testing.assert_close(basis @ basis.T, torch.eye(16), atol=1e-6, rtol=1e-4)
            assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)
