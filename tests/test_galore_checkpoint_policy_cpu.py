"""Explicit checkpoint precision policies survive legacy format labels."""

import copy

import pytest
import torch

from areno.engine.optim.galore import BACKEND_POLICY, GaLoreAdamW
from areno.engine.optim.recycled_subspace import LEGACY_POLICY


def _optimizer(initial):
    parameter = torch.nn.Parameter(torch.full((16, 8), initial, dtype=torch.bfloat16))
    optimizer = GaLoreAdamW([parameter], lr=0.003, rank=2, compact_master=True)
    parameter.main_grad = torch.ones_like(parameter, dtype=torch.float32)
    optimizer.step()
    return parameter, optimizer


def _assert_same(first, second):
    if isinstance(first, torch.Tensor):
        assert isinstance(second, torch.Tensor) and first.dtype == second.dtype
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_same(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert type(first) is type(second) and len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            _assert_same(a, b)
    else:
        assert first == second


@pytest.mark.parametrize("legacy", ["areno-galore-v1", "areno-galore-v2"])
@pytest.mark.parametrize("wrong_policy", [None, {}, {"master": "model-owned-native"}, {"master": "unknown"}])
def test_explicit_incompatible_legacy_policy_rejected_atomically(legacy, wrong_policy):
    _, source = _optimizer(1)
    payload = source.state_dict()
    payload["format"] = legacy
    payload["backend_policy"] = wrong_policy
    if legacy == "areno-galore-v2":
        payload["projector_policy"] = LEGACY_POLICY.copy()
        payload["hyperparameters"]["projector_final_qr"] = False
    parameter, target = _optimizer(3)
    before_parameter, before_state = parameter.detach().clone(), target.state_dict()
    before_rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="unsupported GaLore backend policy"):
        target.load_state_dict(payload)
    assert torch.equal(parameter, before_parameter)
    _assert_same(target.state_dict(), before_state)
    assert torch.equal(torch.get_rng_state(), before_rng)


@pytest.mark.parametrize("include_policy", [False, True])
def test_supported_legacy_v1_restores_and_continues_exactly(include_policy):
    parameter, source = _optimizer(1)
    payload = copy.deepcopy(source.state_dict())
    payload["format"] = "areno-galore-v1"
    if include_policy:
        payload["backend_policy"] = BACKEND_POLICY.copy()
    else:
        payload.pop("backend_policy")
    target_parameter, target = _optimizer(3)
    target.load_state_dict(payload)
    assert torch.equal(parameter, target_parameter)
    _assert_same(source.state_dict(), target.state_dict())
    gradient = torch.arange(128, dtype=torch.float32).reshape(16, 8) / 128
    for _ in range(3):
        parameter.main_grad = gradient.clone()
        target_parameter.main_grad = gradient.clone()
        source.step()
        target.step()
        assert torch.equal(parameter, target_parameter)
        _assert_same(source.state_dict(), target.state_dict())
