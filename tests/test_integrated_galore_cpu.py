"""Combined backend exactness, versioned QR policy, and independent transport."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from areno.engine.config import OptimizerConfig
from areno.engine.modeling import build_optimizer
from areno.engine.optim.galore import GaLoreAdamW
from areno.engine.optim.master_storage import BF16MasterStorage
from areno.engine.optim.recycled_subspace import LEGACY_POLICY, POLICY
from tests.test_galore_transport_cpu import dense_transport
from tests.test_recycled_galore_cpu import IndependentReference

DEVICE = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def assert_states_exact(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], torch.Tensor):
            assert first[key].dtype == second[key].dtype
            assert torch.equal(first[key], second[key]), key
        else:
            assert first[key] == second[key], key


@pytest.mark.parametrize("device", DEVICE)
@pytest.mark.parametrize("precision", ["highest", "high"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32, 16), (16, 32)])
@pytest.mark.parametrize("transport", [False, True])
def test_combined_matches_uncompressed_and_exact_resume(device, precision, dtype, shape, transport):
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    generator = torch.Generator().manual_seed(593)
    initial = [torch.randn(shape, generator=generator), torch.randn(7, generator=generator)]
    params = [torch.nn.Parameter(value.to(device=device, dtype=dtype)) for value in initial]
    references = [torch.nn.Parameter(value.detach().clone()) for value in params]
    settings = dict(
        lr=0.003,
        betas=(0.8, 0.9),
        weight_decay=0.1,
        rank=2,
        update_proj_gap=3,
        scale=0.7,
        projector_backend="recycled",
        projector_oversampling=2,
        projector_seed=17,
        transport_moments=transport,
    )
    actual = GaLoreAdamW(
        params, compact_master=True, tile_rows=8, release_gradients=True, fused_writeback=True, **settings
    )
    reference = GaLoreAdamW(references, **settings)
    gradients = [[torch.randn(value.shape, generator=generator).to(device) for value in initial] for _ in range(8)]
    global_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    try:
        for index, gradients_for_step in enumerate(gradients):
            for j, (param, other, grad) in enumerate(zip(params, references, gradients_for_step, strict=True)):
                # The matrix's first update is skipped; its private clock stays zero.
                use_gradient = not (j == 0 and index == 0) and not (j == 1 and index % 2)
                param.main_grad = grad.clone() if use_gradient else None
                other.main_grad = grad.clone() if use_gradient else None
            actual.step()
            reference.step()
            assert torch.get_float32_matmul_precision() == precision
            for j, (param, other) in enumerate(zip(params, references, strict=True)):
                assert torch.equal(param, other)
                left = actual.state_dict()["state"][j]
                right = reference.state_dict()["state"][j]
                assert_states_exact(left, right)
                if left and j == 0:
                    rng = actual.state[param]["projector_rng_state"]
                    assert rng.device.type == "cpu" and rng.dtype == torch.uint8
                assert param.main_grad is None
            if index in (0, 3, 4):
                snapshot = actual.state_dict()
                replacements = [torch.nn.Parameter(param.detach().clone()) for param in params]
                resumed = GaLoreAdamW(replacements, lr=0.9)
                resumed.load_state_dict(snapshot)
                assert resumed.projector_final_qr and resumed.compact_master and resumed.fused_writeback
                assert resumed.release_gradients and resumed.transport_moments == transport
                actual, params = resumed, replacements
        assert torch.equal(torch.get_rng_state(), global_rng)
        if cuda_rng is not None:
            assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
        if dtype == torch.bfloat16:
            assert all(isinstance(actual.state[p]["master"], BF16MasterStorage) for p in params)
            assert actual.state_memory_metrics()["master_bytes"] == sum(
                p.numel() * 2 + (p.numel() + 7) // 8 for p in params
            )
        actual.offload_state()
        actual.onload_state(torch.device(device))
        assert actual.state[params[0]]["projector_rng_state"].device.type == "cpu"
        assert_states_exact(actual.state_dict()["state"][0], reference.state_dict()["state"][0])
    finally:
        torch.set_float32_matmul_precision(previous)


class TransportReference(IndependentReference):
    """Independent QR/Ritz and torch AdamW with full-covariance transport."""

    def _update(self, gradient):
        old_basis = None if self.basis is None else self.basis.clone()
        completed_step = self.step
        original_step = self.optimizer.step

        def transported_step():
            if completed_step % 3 == 0 and old_basis is not None:
                state = self.optimizer.state[self.coordinates]
                mean, second = dense_transport(
                    dict(
                        step=completed_step,
                        basis=old_basis.T if self.right else old_basis,
                        exp_avg=state["exp_avg"],
                        exp_avg_sq=state["exp_avg_sq"],
                    ),
                    self.basis.T if self.right else self.basis,
                    self.right,
                    (0.8, 0.9),
                )
                state["exp_avg"].copy_(mean)
                state["exp_avg_sq"].copy_(second)
            return original_step()

        self.optimizer.step = transported_step
        try:
            super()._update(gradient)
        finally:
            self.optimizer.step = original_step


@pytest.mark.parametrize("device", DEVICE)
@pytest.mark.parametrize("shape", [(11, 7), (7, 11)])
def test_combined_transport_independent_reference(device, shape):
    generator = torch.Generator().manual_seed(593)
    param = torch.nn.Parameter(torch.randn(shape, generator=generator).to(device))
    reference = TransportReference(param.detach())
    actual = GaLoreAdamW(
        [param],
        lr=0.003,
        betas=(0.8, 0.9),
        weight_decay=0.1,
        rank=2,
        update_proj_gap=3,
        scale=0.7,
        projector_backend="recycled",
        projector_oversampling=2,
        projector_seed=17,
        compact_master=True,
        tile_rows=8,
        release_gradients=True,
        fused_writeback=True,
        transport_moments=True,
    )
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        for _ in range(10):
            gradient = torch.randn(shape, generator=generator).to(device)
            param.main_grad = gradient.clone()
            actual.step()
            reference.update(gradient)
            torch.testing.assert_close(param, reference.weight, atol=1e-6, rtol=1e-4)
            for key in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    actual.state[param][key],
                    reference.optimizer.state[reference.coordinates][key],
                    atol=1e-6,
                    rtol=1e-4,
                )
            assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("backend", ["svd", "recycled"])
def test_highest_scope_restores_after_normal_step_and_exception(monkeypatch, backend):
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = GaLoreAdamW([p], lr=0.003, rank=2, projector_backend=backend, transport_moments=True)
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    observed = []
    original = torch.Tensor.__matmul__

    def checked_matmul(left, right):
        observed.append(torch.get_float32_matmul_precision())
        return original(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", checked_matmul)
    try:
        for _ in range(4):
            p.main_grad = torch.randn_like(p)
            opt.step()
        assert observed and set(observed) == {"highest"}
        assert torch.get_float32_matmul_precision() == "high"

        def closure():
            assert torch.get_float32_matmul_precision() == "highest"
            raise RuntimeError("intentional scope test")

        with pytest.raises(RuntimeError, match="intentional"):
            opt.step(closure)
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous)


def test_legacy_v2_policy_restores_legacy_qr_rule_and_resumes_exactly():
    generator = torch.Generator().manual_seed(593)
    p = torch.nn.Parameter(torch.randn(32, 16, generator=generator))
    old = GaLoreAdamW([p], lr=0.003, rank=2, update_proj_gap=2, projector_backend="recycled", projector_final_qr=False)
    for _ in range(3):
        p.main_grad = torch.randn(32, 16, generator=generator)
        old.step()
    saved = old.state_dict()
    saved["format"] = "areno-galore-v2"
    saved.pop("backend_policy")
    for key in (
        "projector_final_qr",
        "compact_master",
        "tile_rows",
        "release_gradients",
        "fused_writeback",
        "transport_moments",
    ):
        saved["hyperparameters"].pop(key)
    assert saved["projector_policy"] == LEGACY_POLICY
    q = torch.nn.Parameter(torch.zeros_like(p))
    resumed = GaLoreAdamW([q], lr=0.8, compact_master=True)
    resumed.load_state_dict(saved)
    assert not resumed.projector_final_qr and not resumed.compact_master
    for _ in range(4):
        gradient = torch.randn(32, 16, generator=generator)
        p.main_grad = gradient
        q.main_grad = gradient.clone()
        old.step()
        resumed.step()
        assert torch.equal(p, q)
        assert_states_exact(old.state_dict()["state"][0], resumed.state_dict()["state"][0])
    assert POLICY != LEGACY_POLICY


@pytest.mark.parametrize("field", ["backend_policy", "projector_policy"])
def test_incompatible_integrated_policy_rejected_atomically(field):
    p = torch.nn.Parameter(torch.randn(32, 16))
    opt = GaLoreAdamW([p], lr=0.003, rank=2, projector_backend="recycled")
    p.grad = torch.randn_like(p)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    saved[field] = {}
    before = p.detach().clone()
    with pytest.raises(ValueError, match="policy"):
        opt.load_state_dict(saved)
    assert torch.equal(before, p)


def test_internal_factory_defaults_and_combined_flags():
    default = OptimizerConfig(galore_rank=2)
    config = OptimizerConfig(
        galore_rank=2,
        galore_projector_backend="recycled",
        galore_compact_master=True,
        galore_tile_rows=16,
        galore_release_gradients=True,
        galore_fused_writeback=True,
        galore_transport_moments=True,
    )
    ctx = SimpleNamespace(world_size=1, dp_size=1)
    plain = build_optimizer([torch.nn.Parameter(torch.zeros(32, 16))], default, ctx)
    combined = build_optimizer([torch.nn.Parameter(torch.zeros(32, 16))], config, ctx)
    assert plain.projector_backend == "svd" and not plain.compact_master and not plain.transport_moments
    assert combined.projector_final_qr and combined.compact_master and combined.fused_writeback
    assert combined.release_gradients and combined.transport_moments and combined.tile_rows == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_final_qr_strict_gram_representative_shapes_under_worker_high():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    generator = torch.Generator().manual_seed(593)
    try:
        for shape in ((2048, 1024), (1024, 2048)):
            p = torch.nn.Parameter(torch.zeros(shape, device="cuda", dtype=torch.bfloat16))
            opt = GaLoreAdamW(
                [p],
                lr=1e-5,
                rank=128,
                update_proj_gap=1,
                projector_backend="recycled",
                compact_master=True,
                tile_rows=512,
                release_gradients=True,
                fused_writeback=True,
            )
            for _ in range(3):
                p.main_grad = torch.randn(shape, generator=generator).cuda()
                opt.step()
                assert torch.get_float32_matmul_precision() == "high"
                basis = opt.state[p]["basis"].cpu()
                gram = basis @ basis.T if shape[0] >= shape[1] else basis.T @ basis
                torch.testing.assert_close(gram, torch.eye(128), atol=1e-6, rtol=1e-4)
    finally:
        torch.set_float32_matmul_precision(previous)
