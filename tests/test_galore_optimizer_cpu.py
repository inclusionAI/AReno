"""CPU tests for the GaLore optimizer (issue #593).

Every update check compares ``GaloreAdamW`` against an independent slow
reference that implements Algorithm 2 of arXiv:2403.03507 with explicit
tensor math, so a bug in the production step cannot hide behind shared code.
"""

from __future__ import annotations

import copy

import pytest
import torch

from areno.engine.optim.galore import GaloreAdamW, galore_projector


def _ref_galore_step(state, grad, lr, betas, weight_decay, rank, refresh_interval, alpha, eps=1e-8):
    """Literal Algorithm 2 (plus decoupled AdamW decay), written out slowly."""

    beta1, beta2 = betas
    t = state["step"] + 1
    master = state["master"]
    if weight_decay != 0.0:
        master = master * (1.0 - lr * weight_decay)
    left = grad.shape[0] <= grad.shape[1]
    proj = state.get("proj")
    if proj is None or (t - 1) % refresh_interval == 0:
        u, _, vh = torch.linalg.svd(grad, full_matrices=False)
        proj = u[:, :rank] if left else vh[:rank, :].transpose(0, 1)
    compact = proj.transpose(0, 1) @ grad if left else grad @ proj
    m = beta1 * state["m"] + (1.0 - beta1) * compact
    v = beta2 * state["v"] + (1.0 - beta2) * compact * compact
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    n = m_hat / (v_hat.sqrt() + eps)
    update = alpha * (proj @ n if left else n @ proj.transpose(0, 1))
    master = master - lr * update
    state.update(master=master, m=m, v=v, proj=proj, step=t)


def _ref_adamw_step(state, grad, lr, betas, weight_decay, eps=1e-8):
    """Full-space AdamW fallback reference (same decay placement)."""

    beta1, beta2 = betas
    t = state["step"] + 1
    master = state["master"]
    if weight_decay != 0.0:
        master = master * (1.0 - lr * weight_decay)
    m = beta1 * state["m"] + (1.0 - beta1) * grad
    v = beta2 * state["v"] + (1.0 - beta2) * grad * grad
    m_hat = m / (1.0 - beta1**t)
    v_hat = v / (1.0 - beta2**t)
    n = m_hat / (v_hat.sqrt() + eps)
    master = master - lr * n
    state.update(master=master, m=m, v=v, step=t)


def _fresh_state(master, compact_shape):
    return {
        "master": master.clone(),
        "m": torch.zeros(compact_shape, dtype=torch.float32),
        "v": torch.zeros(compact_shape, dtype=torch.float32),
        "step": 0,
    }


def _compact_shape(grad, rank):
    if grad.shape[0] <= grad.shape[1]:
        return (rank, grad.shape[1])
    return (grad.shape[0], rank)


def _drive(optimizer, param, grads):
    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()


def _make_pair(shape, rank, **kwargs):
    torch.manual_seed(sum(shape))
    param = torch.nn.Parameter(torch.randn(*shape))
    optimizer = GaloreAdamW([param], lr=kwargs.pop("lr", 1e-3), galore_rank=rank, **kwargs)
    return param, optimizer


GRAD_SPECS = {
    "wide": (6, 10),
    "tall": (10, 6),
    "square": (8, 8),
}


@pytest.mark.parametrize("shape_name", ["wide", "tall", "square"])
def test_update_matches_reference(shape_name):
    shape = GRAD_SPECS[shape_name]
    rank = 3
    param, optimizer = _make_pair(shape, rank, galore_refresh_interval=2, weight_decay=0.05)
    grads = [torch.randn(*shape) for _ in range(5)]
    ref = _fresh_state(param.detach().float(), _compact_shape(param.detach(), rank))
    for grad in grads:
        param.grad = grad.clone()
        optimizer.step()
        _ref_galore_step(ref, grad, 1e-3, (0.9, 0.95), 0.05, rank, 2, 1.0)
    assert torch.allclose(param.detach().float(), ref["master"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(optimizer.state[param]["exp_avg"], ref["m"], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("alpha", [0.5, 2.0])
def test_alpha_scaling_matches_reference(alpha):
    rank = 2
    param, optimizer = _make_pair((5, 9), rank, galore_alpha=alpha, galore_refresh_interval=1)
    grad = torch.randn(5, 9)
    ref = _fresh_state(param.detach().float(), _compact_shape(param.detach(), rank))
    param.grad = grad.clone()
    optimizer.step()
    _ref_galore_step(ref, grad, 1e-3, (0.9, 0.95), 0.0, rank, 1, alpha)
    assert torch.allclose(param.detach().float(), ref["master"], atol=1e-6, rtol=1e-5)


def test_projector_refreshes_only_at_interval():
    rank, interval = 2, 3
    param, optimizer = _make_pair((6, 8), rank, galore_refresh_interval=interval)
    captured = []
    for _ in range(7):
        param.grad = torch.randn(6, 8)
        optimizer.step()
        captured.append(optimizer.state[param]["proj"])
    for index in range(1, len(captured)):
        refreshed = (index) % interval == 0  # steps 1-based: refresh at 1, 4, 7
        assert (captured[index] is not captured[index - 1]) is refreshed
    # The refreshed basis is exactly the SVD projector of that step's gradient.
    # Steps 8-9 reuse the basis (7 % 3 == 0 was the last refresh); step 10
    # refreshes because (10 - 1) % 3 == 0.
    for _ in range(2):
        param.grad = torch.randn(6, 8)
        optimizer.step()
    grad3 = torch.randn(6, 8)
    param.grad = grad3.clone()
    optimizer.step()  # step 10: refresh
    assert torch.allclose(optimizer.state[param]["proj"], galore_projector(grad3, rank), atol=1e-6)


def test_ineligible_params_fall_back_to_full_adamw():
    param_1d = torch.nn.Parameter(torch.randn(5))
    optimizer = GaloreAdamW([param_1d], lr=1e-3, galore_rank=2, weight_decay=0.1)
    ref = _fresh_state(param_1d.detach().float(), (5,))
    for _ in range(3):
        grad = torch.randn(5)
        param_1d.grad = grad.clone()
        optimizer.step()
        _ref_adamw_step(ref, grad, 1e-3, (0.9, 0.95), 0.1)
    assert optimizer.state[param_1d]["galore"] is False
    assert torch.allclose(param_1d.detach().float(), ref["master"], atol=1e-6, rtol=1e-5)

    # rank not strictly below both matrix dims -> no projection, ordinary update.
    param_square = torch.nn.Parameter(torch.randn(8, 8))
    optimizer = GaloreAdamW([param_square], lr=1e-3, galore_rank=8)
    assert optimizer.state == {}  # lazy state: eligibility decided on first step
    param_square.grad = torch.randn(8, 8)
    optimizer.step()
    assert optimizer.state[param_square]["galore"] is False


def test_projector_shapes_and_compact_moments():
    rank = 2
    for shape in ((6, 10), (10, 6)):
        param, optimizer = _make_pair(shape, rank)
        param.grad = torch.randn(*shape)
        optimizer.step()
        state = optimizer.state[param]
        proj = state["proj"]
        if shape[0] <= shape[1]:
            assert proj.shape == (shape[0], rank)
            assert state["exp_avg"].shape == (rank, shape[1])
        else:
            assert proj.shape == (shape[1], rank)
            assert state["exp_avg"].shape == (shape[0], rank)
        assert state["exp_avg"].numel() < param.numel()
        assert state["exp_avg_sq"].shape == state["exp_avg"].shape


def test_checkpoint_resume_matches_uninterrupted():
    rank, interval = 3, 2
    torch.manual_seed(7)
    init = torch.randn(6, 10)
    grads = [torch.randn(6, 10) for _ in range(5)]

    param_a = torch.nn.Parameter(init.clone())
    opt_a = GaloreAdamW([param_a], lr=1e-3, galore_rank=rank, galore_refresh_interval=interval)
    for grad in grads[:3]:
        param_a.grad = grad.clone()
        opt_a.step()
    saved = copy.deepcopy(opt_a.state_dict())

    # Resumed run: fresh optimizer with the saved state, then the remaining grads.
    param_b = torch.nn.Parameter(param_a.detach().clone())
    opt_b = GaloreAdamW([param_b], lr=1e-3, galore_rank=rank, galore_refresh_interval=interval)
    opt_b.load_state_dict(saved)
    for grad in grads[3:]:
        param_b.grad = grad.clone()
        opt_b.step()

    # Uninterrupted run: same init, all five grads in one optimizer.
    param_c = torch.nn.Parameter(init.clone())
    opt_c = GaloreAdamW([param_c], lr=1e-3, galore_rank=rank, galore_refresh_interval=interval)
    for grad in grads:
        param_c.grad = grad.clone()
        opt_c.step()

    assert torch.allclose(param_b.detach().float(), param_c.detach().float(), atol=1e-6, rtol=1e-5)
    state_b, state_c = opt_b.state[param_b], opt_c.state[param_c]
    assert state_b["step"] == state_c["step"] == 5
    assert torch.allclose(state_b["exp_avg"], state_c["exp_avg"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(state_b["proj"], state_c["proj"], atol=1e-6)


def test_master_weight_copies_back_into_low_precision_param():
    param = torch.nn.Parameter(torch.randn(6, 10).to(torch.bfloat16))
    optimizer = GaloreAdamW([param], lr=1e-3, galore_rank=2)
    param.grad = torch.randn(6, 10).to(torch.bfloat16)
    optimizer.step()
    master = optimizer.state[param]["master"]
    assert master.dtype == torch.float32
    assert torch.equal(param.data, master.to(torch.bfloat16))


def test_constructor_validation():
    param = torch.nn.Parameter(torch.randn(4, 4))
    with pytest.raises(ValueError, match="galore_rank"):
        GaloreAdamW([param], galore_rank=0)
    with pytest.raises(ValueError, match="galore_refresh_interval"):
        GaloreAdamW([param], galore_refresh_interval=0)
    with pytest.raises(ValueError, match="galore_alpha"):
        GaloreAdamW([param], galore_alpha=0.0)
    with pytest.raises(NotImplementedError, match="single-GPU"):
        GaloreAdamW([param], dp_size=2)


def test_optimizer_config_galore_validation():
    from areno.engine.config import OptimizerConfig

    OptimizerConfig(galore=True, galore_rank=4, galore_refresh_interval=8)
    with pytest.raises(ValueError, match="mutually exclusive"):
        OptimizerConfig(galore=True, adam_8bit=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        OptimizerConfig(galore=True, paged=True)
    with pytest.raises(ValueError, match="galore_rank"):
        OptimizerConfig(galore=True, galore_rank=0)
    with pytest.raises(ValueError, match="galore_refresh_interval"):
        OptimizerConfig(galore=True, galore_refresh_interval=0)
    with pytest.raises(ValueError, match="galore_alpha"):
        OptimizerConfig(galore=True, galore_alpha=0.0)


def test_engine_attribute_surface():
    """Training loop iterates ``optimizer.model_params`` and assigns ``optimizer.lr``."""
    weight = torch.nn.Parameter(torch.randn(4, 6))
    frozen = torch.nn.Parameter(torch.randn(3, 3), requires_grad=False)
    optimizer = GaloreAdamW([weight, frozen], lr=1e-3, galore_rank=2)
    assert optimizer.model_params == [weight]
    optimizer.lr = 2e-3
    assert optimizer.lr == 2e-3
    assert all(group["lr"] == 2e-3 for group in optimizer.param_groups)


def test_step_honors_engine_lr_and_areno_override():
    rank = 2
    base_lr, override_lr = 4e-3, 7e-3
    torch.manual_seed(3)
    default = torch.nn.Parameter(torch.randn(5, 9))
    tagged = torch.nn.Parameter(torch.randn(9, 5))
    tagged._areno_lr = override_lr
    optimizer = GaloreAdamW(
        [default, tagged],
        lr=1e-9,
        galore_rank=rank,
        galore_refresh_interval=2,
        weight_decay=0.05,
    )
    optimizer.lr = base_lr  # the engine writes the scheduled lr before each step
    grads = {default: torch.randn(5, 9), tagged: torch.randn(9, 5)}
    refs = {
        default: _fresh_state(default.detach().float(), _compact_shape(default.detach(), rank)),
        tagged: _fresh_state(tagged.detach().float(), _compact_shape(tagged.detach(), rank)),
    }
    lrs = {default: base_lr, tagged: override_lr}
    for _ in range(3):
        for param, grad in grads.items():
            param.grad = grad.clone()
        optimizer.step()
        for param, grad in grads.items():
            _ref_galore_step(refs[param], grad, lrs[param], (0.9, 0.95), 0.05, rank, 2, 1.0)
    for param in (default, tagged):
        assert torch.allclose(param.detach().float(), refs[param]["master"], atol=1e-6, rtol=1e-5)
