"""Private deferred prototype policy, barrier and preflight invariants."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from areno.engine.optim.galore import GaLoreAdamW
from areno.engine.training import TrainingManager


def make(monkeypatch, enabled):
    monkeypatch.setenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", str(int(enabled)))
    p = torch.nn.Parameter(torch.arange(48, dtype=torch.float32).reshape(8, 6) / 100)
    opt = GaLoreAdamW(
        [p],
        lr=0.003,
        rank=2,
        update_proj_gap=2,
        projector_backend="recycled",
        projector_seed=17,
        transport_moments=True,
    )
    if enabled:
        opt.begin_deferred_projection()
        p.square().sum().backward()
        opt.finish_deferred_backward()
        norm = opt.deferred_grad_norms()["global"]
        opt.clip_deferred_gradients(norm, 1)
        opt.step()
    else:
        p.main_grad = 2 * p.detach()
        opt.step()
        opt.zero_grad()
    return p, opt


def identical(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            identical(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            identical(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("schema", ["areno-galore-v1", "areno-galore-v2", "areno-galore-v3"])
def test_cross_policy_rejected_before_mutation(monkeypatch, enabled, schema):
    _, source = make(monkeypatch, enabled)
    payload = source.state_dict()
    payload["format"] = schema
    payload["state"][0]["master"].fill_(100)
    p, target = make(monkeypatch, not enabled)
    before = target.state_dict()
    weight = p.detach().clone()
    with pytest.raises(ValueError):
        target.load_state_dict(payload)
    identical(before, target.state_dict())
    assert torch.equal(p, weight)


@pytest.mark.parametrize("micro,reentrant", [(2, False), (1, True), (0, False)])
def test_begin_rejects_unsupported_before_backward(monkeypatch, micro, reentrant):
    p, opt = make(monkeypatch, True)
    before = opt.state_dict()
    weight = p.detach().clone()
    with pytest.raises(ValueError):
        opt.begin_deferred_projection(micro_batches=micro, reentrant=reentrant)
    identical(before, opt.state_dict())
    assert torch.equal(p, weight)


def test_pending_snapshot_and_early_step_rejected(monkeypatch):
    p, opt = make(monkeypatch, True)
    payload = opt.state_dict()
    opt.begin_deferred_projection()
    with pytest.raises(RuntimeError, match="empty step-boundary"):
        opt.state_dict()
    with pytest.raises(RuntimeError, match="completed backward"):
        opt.step()
    p.square().sum().backward()
    with pytest.raises(RuntimeError, match="empty step-boundary"):
        opt.load_state_dict(payload)
    opt.abort_deferred_projection()


def test_duplicate_owner_is_not_silently_accumulated(monkeypatch):
    p, opt = make(monkeypatch, True)
    before = opt.state_dict()
    weight = p.detach().clone()
    opt.begin_deferred_projection()
    p.square().sum().backward()
    with pytest.raises(RuntimeError, match="more than once"):
        p.square().sum().backward()
    opt.abort_deferred_projection()
    opt.zero_grad()
    identical(before, opt.state_dict())
    assert torch.equal(p, weight)


def test_barrier_detects_early_mutation(monkeypatch):
    p, opt = make(monkeypatch, True)
    opt.begin_deferred_projection()
    p.square().sum().backward()
    with torch.no_grad():
        p.add_(1)
    with pytest.raises(RuntimeError, match="mutated before"):
        opt.finish_deferred_backward()
    opt.abort_deferred_projection()


def test_barrier_detects_early_state_publication(monkeypatch):
    p, opt = make(monkeypatch, True)
    opt.begin_deferred_projection()
    p.square().sum().backward()
    opt.state[p] = copy.deepcopy(opt.state[p])
    with pytest.raises(RuntimeError, match="published before"):
        opt.finish_deferred_backward()
    opt.abort_deferred_projection()


def test_missing_gradient_keeps_original_projector_owner_index(monkeypatch):
    monkeypatch.setenv("ARENO_EXPERIMENT_DEFERRED_PROJECTION", "1")
    p0 = torch.nn.Parameter(torch.randn(8, 6))
    v = torch.nn.Parameter(torch.ones(3))
    p2 = torch.nn.Parameter(torch.randn(9, 7))
    opt = GaLoreAdamW([p0, v, p2], lr=0.003, rank=2, projector_backend="recycled", projector_seed=17)
    opt.begin_deferred_projection()
    (v.square().sum() + p2.square().sum()).backward()
    assert not opt.state
    opt.finish_deferred_backward()
    opt.deferred_grad_norms()
    opt.step()
    assert p0 not in opt.state and opt.state[p2]["projector_parameter_index"] == 2
    assert opt._deferred_last["deferred_owner_count"] == 2


@pytest.mark.parametrize("allow,scale", [(False, 1), (True, 2)])
def test_invalid_train_step_rejected_before_model_work(allow, scale):
    worker = SimpleNamespace(optimizer=SimpleNamespace(_deferred_projection=True, _deferred_active=True))
    with pytest.raises(ValueError, match="one complete supported backward"):
        TrainingManager(worker)._train_step([], allow_step=allow, grad_scale=scale)


def test_multi_pack_rejected_before_zero_grad():
    calls = []
    runtime = SimpleNamespace(optimizer_state_offload="none", keep_rollout_state=True, compile_model=False)
    worker = SimpleNamespace(
        config=SimpleNamespace(runtime=runtime, model=SimpleNamespace(model_type="qwen3")),
        _train_state_ready=True,
        multimodal_lr_schedules={},
        optimizer=SimpleNamespace(_deferred_projection=True, zero_grad=lambda **kw: calls.append("zero")),
    )
    payload = SimpleNamespace(data_packs_by_dp=[{}, {}], gradient_accumulation_steps=2)
    with pytest.raises(ValueError, match="one microbatch before training starts"):
        TrainingManager(worker).train(payload)
    assert not calls


@pytest.mark.parametrize("group", [None, "tower", "projector"])
@pytest.mark.parametrize("compiled", [False, True])
def test_default_multimodal_schedules_use_actual_owner_groups(group, compiled):
    calls = []
    parameter = torch.nn.Parameter(torch.ones(3))
    if group is not None:
        parameter._areno_lr_group = group
    runtime = SimpleNamespace(optimizer_state_offload="none", keep_rollout_state=True, compile_model=compiled)
    worker = SimpleNamespace(
        config=SimpleNamespace(runtime=runtime, model=SimpleNamespace(model_type="qwen3")),
        model=SimpleNamespace(parameters=lambda: iter([parameter])),
        _train_state_ready=True,
        multimodal_lr_schedules={"tower": {"lr": 1e-5}, "projector": {"lr": 1e-5}},
        optimizer=SimpleNamespace(
            _deferred_projection=True,
            zero_grad=lambda **kw: calls.append("zero"),
            begin_deferred_projection=lambda **kw: calls.append("begin"),
            abort_deferred_projection=lambda: calls.append("abort"),
        ),
    )
    manager = TrainingManager(worker)
    manager._train_step = lambda *args, **kw: calls.append("backward")
    payload = SimpleNamespace(data_packs_by_dp=[{}], gradient_accumulation_steps=1)
    if group is not None or compiled:
        with pytest.raises(ValueError, match="compilation, multimodal groups and offload"):
            manager.train(payload)
        assert not calls
    else:
        assert manager.train(payload) == [None]
        assert calls == ["zero", "begin", "backward", "abort"]
