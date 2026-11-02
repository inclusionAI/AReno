"""Transport reference uses explicit small full covariance matrices."""

import pytest
import torch

from areno.engine.optim.galore import GaLoreAdamW


def dense_transport(state, new_basis, right, betas):
    step = state["step"]
    c1, c2 = 1 - betas[0] ** step, 1 - betas[1] ** step
    old_basis = state["basis"].T if right else state["basis"]
    new = new_basis.T if right else new_basis
    change = new.T @ old_basis
    means = (state["exp_avg"] / c1).T if right else state["exp_avg"] / c1
    seconds = (state["exp_avg_sq"] / c2).T if right else state["exp_avg_sq"] / c2
    result_mean = torch.empty_like(means)
    result_second = torch.empty_like(seconds)
    for col in range(means.shape[1]):
        mean = means[:, col]
        second = torch.diag(seconds[:, col] - mean.square()) + torch.outer(mean, mean)
        result_mean[:, col] = change @ mean
        result_second[:, col] = torch.diag(change @ second @ change.T).clamp_min(0)
    if right:
        result_mean, result_second = result_mean.T, result_second.T
    return result_mean * c1, result_second * c2


@pytest.mark.parametrize("shape", [(11, 7), (7, 11), (9, 9)])
def test_transport_update_independent_adamw_reference(shape):
    torch.manual_seed(593)
    p = torch.nn.Parameter(torch.randn(shape))
    expected = p.detach().clone()
    rank, gap, betas, rate, scale, decay = 2, 3, (0.8, 0.9), 0.003, 0.7, 0.1
    right = shape[0] >= shape[1]
    projected_shape = (shape[0], rank) if right else (rank, shape[1])
    virtual = torch.nn.Parameter(torch.zeros(projected_shape))
    reference = torch.optim.AdamW([virtual], lr=rate, betas=betas, weight_decay=0, eps=1e-8, foreach=False)
    actual = GaLoreAdamW(
        [p],
        lr=rate,
        betas=betas,
        weight_decay=decay,
        rank=rank,
        update_proj_gap=gap,
        scale=scale,
        transport_moments=True,
    )
    basis = None
    for step in range(10):
        grad = torch.randn_like(p)
        if step % gap == 0:
            u, _, vh = torch.linalg.svd(grad, full_matrices=True)
            new_basis = vh[:rank].clone() if right else u[:, :rank].clone()
            if basis is not None:
                state = reference.state[virtual]
                m, v = dense_transport(
                    dict(step=step, basis=basis, exp_avg=state["exp_avg"], exp_avg_sq=state["exp_avg_sq"]),
                    new_basis,
                    right,
                    betas,
                )
                state["exp_avg"].copy_(m)
                state["exp_avg_sq"].copy_(v)
            basis = new_basis
        projected = torch.einsum("ij,kj->ik", grad, basis) if right else torch.einsum("ik,ij->kj", basis, grad)
        with torch.no_grad():
            virtual.zero_()
        virtual.grad = projected
        reference.step()
        delta = virtual.detach() @ basis if right else basis @ virtual.detach()
        expected = expected * (1 - rate * decay) + scale * delta
        p.grad = grad.clone()
        actual.step()
        torch.testing.assert_close(p, expected, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("right", [False, True])
def test_signed_permutation_preserves_lifted_statistics(right):
    torch.manual_seed(5)
    basis, _ = torch.linalg.qr(torch.randn(7, 2))
    basis = basis.T if right else basis
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    new = rotation @ basis if right else basis @ rotation
    p = torch.nn.Parameter(torch.zeros(11, 7) if right else torch.zeros(7, 11))
    optim = GaLoreAdamW([p], lr=0.01, rank=2, transport_moments=True)
    mean = torch.randn(11, 2) if right else torch.randn(2, 11)
    second = mean.square() + torch.rand_like(mean)
    state = dict(step=4, basis=basis, exp_avg=mean.clone(), exp_avg_sq=second.clone())
    expected_mean, expected_second = dense_transport(state, new, right, optim.betas)
    optim._transport_history(state, new, right)
    torch.testing.assert_close(state["exp_avg"], expected_mean, atol=1e-6, rtol=1e-4)
    torch.testing.assert_close(state["exp_avg_sq"], expected_second, atol=1e-6, rtol=1e-4)
    before = mean @ basis if right else basis @ mean
    after = state["exp_avg"] @ new if right else new @ state["exp_avg"]
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize("save_after", [1, 3, 4, 6])
def test_transport_resume_exact(save_after):
    torch.manual_seed(42)
    p = torch.nn.Parameter(torch.randn(11, 7, dtype=torch.bfloat16))
    optim = GaLoreAdamW([p], lr=0.003, rank=2, update_proj_gap=3, transport_moments=True)
    grads = [torch.randn_like(p, dtype=torch.float32) for _ in range(10)]
    for grad in grads[:save_after]:
        p.main_grad = grad
        optim.step()
    other_p = torch.nn.Parameter(torch.zeros_like(p))
    other = GaLoreAdamW([other_p], lr=1, rank=1, transport_moments=False)
    other.load_state_dict(optim.state_dict())
    assert other.transport_moments
    for grad in grads[save_after:]:
        p.main_grad = grad
        other_p.main_grad = grad.clone()
        optim.step()
        other.step()
        torch.testing.assert_close(p, other_p, atol=0, rtol=0)
        for name in ("master", "exp_avg", "exp_avg_sq", "basis"):
            torch.testing.assert_close(optim.state[p][name], other.state[other_p][name], atol=0, rtol=0)
