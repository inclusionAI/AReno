from __future__ import annotations

import pytest
import torch

from areno.optim.galore_reference import GaLoreAdamWReference, adamw_state_numel, galore_state_numel


@pytest.mark.parametrize("shape", [(8, 12), (12, 8), (8, 8)])
def test_saturated_rank_matches_adamw_for_multiple_steps(shape: tuple[int, int]) -> None:
    """The uncompressed path must agree with AdamW, including weight decay."""

    dtype = torch.float64
    generator = torch.Generator().manual_seed(593)
    initial = torch.randn(shape, generator=generator, dtype=dtype)
    galore_param = torch.nn.Parameter(initial.clone())
    adamw_param = torch.nn.Parameter(initial.clone())
    kwargs = dict(lr=0.017, betas=(0.83, 0.97), eps=1.0e-9, weight_decay=0.04)
    galore = GaLoreAdamWReference([galore_param], rank=min(shape), **kwargs)
    adamw = torch.optim.AdamW([adamw_param], foreach=False, **kwargs)

    for _ in range(15):
        gradient = torch.randn(shape, generator=generator, dtype=dtype) * 0.37
        galore_param.grad = gradient.clone()
        adamw_param.grad = gradient.clone()
        galore.step()
        adamw.step()
        torch.testing.assert_close(galore_param, adamw_param, rtol=0.0, atol=1.0e-6)

    assert galore.state[galore_param].exp_avg.shape == shape
    assert galore_state_numel(shape, min(shape)) == adamw_state_numel(shape)


@pytest.mark.parametrize(
    ("shape", "expected_moment_shape"),
    [((12, 20), (4, 20)), ((20, 12), (20, 4))],
)
def test_low_rank_svd_state_has_expected_shape_and_orthonormal_basis(
    shape: tuple[int, int], expected_moment_shape: tuple[int, int]
) -> None:
    generator = torch.Generator().manual_seed(17)
    param = torch.nn.Parameter(torch.randn(shape, generator=generator))
    optimizer = GaLoreAdamWReference([param], rank=4, lr=0.01, weight_decay=0.0)
    param.grad = torch.randn(shape, generator=generator)

    optimizer.step()

    state = optimizer.state[param]
    assert state.exp_avg.shape == expected_moment_shape
    assert state.exp_avg_sq.shape == expected_moment_shape
    basis = optimizer._projectors[param].basis
    assert basis is not None
    expected_basis_shape = (shape[0], 4) if shape[0] <= shape[1] else (shape[1], 4)
    assert basis.shape == expected_basis_shape
    torch.testing.assert_close(basis.transpose(0, 1) @ basis, torch.eye(4), rtol=0.0, atol=1.0e-6)
    assert 2 * state.exp_avg.numel() == galore_state_numel(shape, 4)


def test_low_rank_projector_is_frozen_after_the_first_gradient() -> None:
    generator = torch.Generator().manual_seed(23)
    param = torch.nn.Parameter(torch.randn(10, 14, generator=generator))
    optimizer = GaLoreAdamWReference([param], rank=3, weight_decay=0.0)
    param.grad = torch.randn(10, 14, generator=generator)
    optimizer.step()
    first_basis = optimizer._projectors[param].basis.clone()

    param.grad = torch.randn(10, 14, generator=generator)
    optimizer.step()

    torch.testing.assert_close(optimizer._projectors[param].basis, first_basis, rtol=0.0, atol=0.0)


def test_memory_formula_covers_both_matrix_orientations() -> None:
    assert adamw_state_numel((12, 20)) == 480
    assert galore_state_numel((12, 20), 4) == 160
    assert galore_state_numel((20, 12), 4) == 160
    assert galore_state_numel((12, 20), 12) == adamw_state_numel((12, 20))


def test_invalid_inputs_fail_early() -> None:
    with pytest.raises(ValueError, match="only 2-D"):
        GaLoreAdamWReference([torch.nn.Parameter(torch.ones(4))], rank=1)
    with pytest.raises(ValueError, match="exceeds"):
        GaLoreAdamWReference([torch.nn.Parameter(torch.ones(4, 6))], rank=5)
    with pytest.raises(ValueError, match="positive integer"):
        GaLoreAdamWReference([torch.nn.Parameter(torch.ones(4, 6))], rank=0)
    with pytest.raises(ValueError, match="no greater than"):
        galore_state_numel((4, 6), 7)
