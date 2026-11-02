"""Independent AdamW reference, refresh/resume, and persistent-byte contracts."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from areno.engine.checkpoints.training_state import load_training_state, save_training_state
from areno.engine.config import EngineConfig, ModelConfig, OptimizerConfig, RuntimeConfig
from areno.engine.optim.galore import GaLoreAdamW


class Reference:
    """Use torch.optim.AdamW on virtual coordinates, independently of production math."""

    def __init__(self, initial, *, rank=2, gap=3, scale=0.7):
        self.weight = initial.clone()
        self.rank, self.gap, self.scale = rank, gap, scale
        rows, cols = initial.shape
        self.virtual = torch.nn.Parameter(initial.new_zeros((rows, rank) if rows >= cols else (rank, cols)))
        self.optim = torch.optim.AdamW(
            [self.virtual], lr=0.003, betas=(0.8, 0.9), eps=1e-8, weight_decay=0, foreach=False
        )
        self.step = 0

    def update(self, grad):
        if self.step % self.gap == 0:
            u, _, vh = torch.linalg.svd(grad, full_matrices=True)
            self.basis = vh[: self.rank].clone() if grad.shape[0] >= grad.shape[1] else u[:, : self.rank].clone()
        if grad.shape[0] >= grad.shape[1]:
            projected = torch.einsum("ij,kj->ik", grad, self.basis)
        else:
            projected = torch.einsum("ik,ij->kj", self.basis, grad)
        with torch.no_grad():
            self.virtual.zero_()
        self.virtual.grad = projected
        self.optim.step()
        if grad.shape[0] >= grad.shape[1]:
            delta = torch.einsum("ik,kj->ij", self.virtual.detach(), self.basis)
        else:
            delta = torch.einsum("ik,kj->ij", self.basis, self.virtual.detach())
        self.weight = self.weight * (1 - 0.003 * 0.1) + self.scale * delta
        self.step += 1


@pytest.mark.parametrize("shape", [(11, 7), (7, 11), (9, 9)])
def test_matches_independent_reference_across_refresh(shape):
    torch.manual_seed(593)
    param = torch.nn.Parameter(torch.randn(shape))
    ref = Reference(param.detach())
    optim = GaLoreAdamW([param], lr=0.003, betas=(0.8, 0.9), weight_decay=0.1, rank=2, update_proj_gap=3, scale=0.7)
    for step in range(10):
        grad = torch.randn_like(param)
        param.grad = grad.clone()
        optim.step()
        ref.update(grad)
        torch.testing.assert_close(param, ref.weight, atol=1e-6, rtol=1e-4)
        assert optim.last_projection_refresh == (step % 3 == 0)


@pytest.mark.parametrize("shape,rank", [((7,), 2), ((4, 5), 4), ((2, 3, 4), 1), ((4, 4), 3)])
def test_ineligible_matches_torch_adamw(shape, rank):
    torch.manual_seed(42)
    param = torch.nn.Parameter(torch.randn(shape))
    expected = torch.nn.Parameter(param.detach().clone())
    actual_optim = GaLoreAdamW([param], lr=0.01, weight_decay=0.1, betas=(0.9, 0.95), rank=rank)
    expected_optim = torch.optim.AdamW([expected], lr=0.01, weight_decay=0.1, betas=(0.9, 0.95))
    for _ in range(5):
        grad = torch.randn_like(param)
        param.grad, expected.grad = grad.clone(), grad.clone()
        actual_optim.step()
        expected_optim.step()
        torch.testing.assert_close(param, expected, atol=1e-6, rtol=1e-4)
    assert not actual_optim.last_projection_refresh


@pytest.mark.parametrize("save_after", [1, 3, 4, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_resume_before_and_after_refresh(save_after, dtype):
    torch.manual_seed(2026)
    param = torch.nn.Parameter(torch.randn(11, 7).to(dtype))
    optim = GaLoreAdamW([param], lr=0.003, rank=2, update_proj_gap=3)
    grads = [torch.randn_like(param) for _ in range(10)]
    for grad in grads[:save_after]:
        param.main_grad = grad.float()
        optim.step()
        optim.zero_grad()
    snapshot = optim.state_dict()
    resumed_param = torch.nn.Parameter(torch.zeros_like(param))
    resumed = GaLoreAdamW([resumed_param], lr=0.1, rank=1, update_proj_gap=17)
    resumed.load_state_dict(snapshot)
    assert resumed.rank == 2 and resumed.update_proj_gap == 3 and resumed.lr == 0.003
    for grad in grads[save_after:]:
        param.main_grad = grad.float()
        resumed_param.main_grad = grad.float()
        optim.step()
        resumed.step()
        torch.testing.assert_close(param, resumed_param, atol=0, rtol=0)
        for key in ("master", "exp_avg", "exp_avg_sq", "basis"):
            torch.testing.assert_close(optim.state[param][key], resumed.state[resumed_param][key], atol=0, rtol=0)
    # Snapshots cannot alias live tensors.
    assert snapshot["state"][0]["step"] == save_after


def test_skipped_gradient_main_grad_and_zero_grad():
    param = torch.nn.Parameter(torch.ones(11, 7))
    frozen = torch.nn.Parameter(torch.ones(11, 7), requires_grad=False)
    optim = GaLoreAdamW([param, frozen], lr=0.01, rank=2)
    optim.step()
    assert not optim.state
    param.grad, param.main_grad = torch.zeros_like(param), torch.randn_like(param)
    optim.step()
    assert optim.state[param]["step"] == 1
    assert len(optim.model_params) == 1
    optim.zero_grad()
    assert param.grad is None and param.main_grad is None


def test_temporary_role_swap_and_gradient_detection():
    param = torch.nn.Parameter(torch.randn(11, 7))
    optim = GaLoreAdamW([param], lr=0.003, rank=2)
    assert not optim.has_gradients()
    param.main_grad = torch.randn_like(param)
    assert optim.has_gradients()
    optim.step()
    expected = optim.state_dict()
    optim.offload_state()
    optim.onload_state(torch.device("cpu"))
    for key in ("master", "exp_avg", "exp_avg_sq", "basis"):
        assert torch.equal(expected["state"][0][key], optim.state[param][key])
    with pytest.raises(ValueError, match="temporary CPU"):
        optim.offload_state(mode="disk")


def test_persistent_bytes_include_projection_basis():
    torch.manual_seed(593)
    param = torch.nn.Parameter(torch.zeros(1024, 1024))
    param.grad = torch.randn_like(param)
    optim = GaLoreAdamW([param], lr=0.001, rank=128)
    optim.step()
    metrics = optim.state_memory_metrics()
    assert metrics["basis_bytes"] == 128 * 1024 * 4
    assert metrics["state_bytes"] == 3 * 128 * 1024 * 4
    assert metrics["state_bytes"] < 2 * 1024 * 1024 * 4
    assert metrics["total_bytes"] == metrics["state_bytes"] + param.numel() * 4


def test_malformed_checkpoint_rejected_without_mutation():
    param = torch.nn.Parameter(torch.randn(11, 7))
    optim = GaLoreAdamW([param], lr=0.01, rank=2)
    param.grad = torch.randn_like(param)
    optim.step()
    snapshot = optim.state_dict()
    snapshot["state"][0]["basis"] = torch.ones(2, 8)
    before = param.detach().clone()
    with pytest.raises(ValueError, match="basis"):
        optim.load_state_dict(snapshot)
    assert torch.equal(before, param)


def test_training_sidecar_roundtrip_restores_schedule_rng_and_master(tmp_path):
    model = torch.nn.Linear(7, 11, bias=False)
    optim = GaLoreAdamW(model.parameters(), lr=0.003, rank=2, update_proj_gap=3)
    model.weight.grad = torch.randn_like(model.weight)
    optim.step()
    worker = SimpleNamespace(
        model=model,
        optimizer=optim,
        device=torch.device("cpu"),
        _global_step=1,
        base_lr=0.003,
        min_lr=0.0001,
        lr_decay_steps=10,
        lr_warmup_steps=2,
        lr_decay_style="cosine",
    )
    save_training_state(worker, str(tmp_path))
    expected_random = torch.rand(4)
    with torch.no_grad():
        model.weight.zero_()
    worker._global_step, worker.base_lr = 0, 1.0
    load_training_state(worker, str(tmp_path))
    assert worker._global_step == 1 and worker.base_lr == 0.003
    assert torch.equal(torch.rand(4), expected_random)
    torch.testing.assert_close(model.weight, optim.state[model.weight]["master"], atol=0, rtol=0)


@pytest.mark.parametrize(
    "kwargs",
    [{"galore_rank": 0}, {"galore_rank": 2, "adam_8bit": True}, {"galore_update_proj_gap": 0}, {"galore_scale": 0}],
)
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        OptimizerConfig(**kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_config_rejects_nonfinite_projection_scale(value):
    with pytest.raises(ValueError):
        OptimizerConfig(galore_scale=value)


@pytest.mark.parametrize("field", ["lr", "weight_decay", "eps", "scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_constructor_rejects_nonfinite_scalar_hyperparameters(field, value):
    settings = {"lr": 0.003, field: value}
    with pytest.raises(ValueError, match="finite"):
        GaLoreAdamW([], **settings)


@pytest.mark.parametrize("field", ["lr", "weight_decay", "eps", "scale"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_checkpoint_hyperparameters_rejected_before_mutation(field, value):
    params = [torch.nn.Parameter(torch.ones(11, 7)), torch.nn.Parameter(torch.ones(3))]
    optimizer = GaLoreAdamW(params, lr=0.003, rank=2)
    for parameter in params:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    original = copy.deepcopy(optimizer.state_dict())
    before_parameters = [parameter.detach().clone() for parameter in params]
    bad = copy.deepcopy(original)
    bad["hyperparameters"][field] = value
    bad["state"][0]["master"].fill_(100)
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="finite"):
        optimizer.load_state_dict(bad)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert optimizer.state_dict()["hyperparameters"] == original["hyperparameters"]
    for parameter, before, saved in zip(params, before_parameters, original["state"], strict=True):
        assert torch.equal(parameter, before)
        for key in ("master", "exp_avg", "exp_avg_sq", "basis"):
            actual = optimizer.state[parameter][key]
            assert actual is None if saved[key] is None else torch.equal(actual, saved[key])


def test_engine_rejects_unsupported_galore_modes(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = dict(model=ModelConfig(), devices=[0], optimizer=OptimizerConfig(galore_rank=2))
    EngineConfig(**config)
    with pytest.raises(ValueError, match="resident"):
        EngineConfig(**config, runtime=RuntimeConfig(keep_rollout_state=False))
    with pytest.raises(ValueError, match="single-GPU"):
        EngineConfig(**{**config, "devices": [0, 1]})
    with pytest.raises(ValueError, match="dense"):
        EngineConfig(**{**config, "model": ModelConfig(num_experts=4)})
