"""Identity-based vocabulary fallback keeps the same ordinary AdamW rule."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from areno.engine.config import OptimizerConfig
from areno.engine.layers.vocab import VocabParallelEmbedding, VocabParallelLMHead
from areno.engine.modeling import build_optimizer, mark_galore_vocabulary_parameters
from areno.engine.optim.galore import GaLoreAdamW

DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


def tagged(value):
    value._areno_galore_vocab = True
    return value


def exact(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            exact(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("tied", [False, True])
def test_module_identities_target_heads_and_inputs_without_shape_guess(tied):
    model = torch.nn.Module()
    model.unusual_input_name = VocabParallelEmbedding(32, 16)
    model.unusual_output_name = VocabParallelLMHead(16, 32)
    model.same_shape_linear = torch.nn.Linear(16, 32, bias=False)
    if tied:
        model.unusual_output_name.weight = model.unusual_input_name.weight
    before = list(model.parameters())
    mark_galore_vocabulary_parameters(model)
    assert list(model.parameters()) == before
    opt = GaLoreAdamW(model.parameters(), lr=0.003, rank=2, exclude_vocab=True)
    assert not opt._projected(model.unusual_input_name.weight)
    assert not opt._projected(model.unusual_output_name.weight)
    assert opt._projected(model.same_shape_linear.weight)
    assert len(opt.model_params) == (2 if tied else 3)
    if tied:
        assert model.unusual_output_name.weight is model.unusual_input_name.weight


@pytest.mark.parametrize("device", DEVICES)
def test_full_vocab_fallback_matches_independent_adamw_and_decay_once(device):
    generator = torch.Generator().manual_seed(593)
    parameter = tagged(torch.nn.Parameter(torch.randn(32, 16, generator=generator).to(device)))
    reference = torch.nn.Parameter(parameter.detach().clone())
    settings = dict(lr=0.003, betas=(0.8, 0.9), weight_decay=0.1, eps=1e-8)
    actual = GaLoreAdamW([parameter], rank=2, scale=4, exclude_vocab=True, transport_moments=True, **settings)
    expected = torch.optim.AdamW([reference], **settings)
    for index in range(8):
        gradient = (
            torch.zeros_like(parameter) if index == 5 else torch.randn(parameter.shape, generator=generator).to(device)
        )
        parameter.main_grad = None if index == 3 else gradient.clone()
        reference.grad = None if index == 3 else gradient.clone()
        actual.step()
        expected.step()
        torch.testing.assert_close(parameter, reference, atol=1e-6, rtol=1e-4)
        state = actual.state[parameter]
        assert state["basis"] is None and "transport_count" not in state
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(state[key], expected.state[reference][key], atol=1e-6, rtol=1e-4)
    assert actual.state_memory_metrics()["moment_bytes"] == parameter.numel() * 8


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_mixed_target_exact_resume_across_refresh_boundary(device, dtype):
    generator = torch.Generator().manual_seed(593)
    params = [
        tagged(torch.nn.Parameter(torch.randn(32, 16, generator=generator).to(device=device, dtype=dtype))),
        torch.nn.Parameter(torch.randn(16, 32, generator=generator).to(device=device, dtype=dtype)),
    ]
    settings = dict(
        lr=0.003,
        rank=2,
        update_proj_gap=3,
        scale=4,
        projector_backend="recycled",
        projector_oversampling=2,
        exclude_vocab=True,
        compact_master=True,
        tile_rows=8,
        fused_writeback=True,
        release_gradients=True,
        transport_moments=True,
    )
    actual = GaLoreAdamW(params, **settings)
    other_params = [
        tagged(torch.nn.Parameter(params[0].detach().clone())),
        torch.nn.Parameter(params[1].detach().clone()),
    ]
    other = GaLoreAdamW(other_params, **settings)
    for index in range(8):
        for position, (first, second) in enumerate(zip(params, other_params, strict=True)):
            grad = torch.randn(first.shape, generator=generator).to(device)
            first.main_grad = grad.clone() if not (position == 0 and index == 4) else None
            second.main_grad = grad.clone() if not (position == 0 and index == 4) else None
        actual.step()
        other.step()
        exact(actual.state_dict(), other.state_dict())
        for a, b in zip(params, other_params, strict=True):
            assert torch.equal(a, b)
        assert actual.state[params[0]]["basis"] is None
        assert actual.state[params[1]]["basis"] is not None
        if index in (1, 2, 3):
            payload = other.state_dict()
            other_params = [
                tagged(torch.nn.Parameter(params[0].detach().clone())),
                torch.nn.Parameter(params[1].detach().clone()),
            ]
            other = GaLoreAdamW(other_params, lr=0.9)
            other.load_state_dict(payload)
            assert other.exclude_vocab
            exact(actual.state_dict(), other.state_dict())


@pytest.mark.parametrize("change", ["policy", "exclusions", "role", "boolean"])
def test_bad_target_checkpoint_rejected_before_mutation(change):
    p = tagged(torch.nn.Parameter(torch.randn(32, 16)))
    opt = GaLoreAdamW([p], lr=0.003, rank=2, exclude_vocab=True)
    p.main_grad = torch.randn_like(p)
    opt.step()
    payload = copy.deepcopy(opt.state_dict())
    target = tagged(torch.nn.Parameter(torch.randn_like(p)))
    if change == "policy":
        payload["projection_policy"]["targets"] = "unknown"
    elif change == "exclusions":
        payload["projection_exclusions"] = [False]
    elif change == "boolean":
        payload["projection_exclusions"] = [1]
    else:
        del target._areno_galore_vocab
    before = target.detach().clone()
    with pytest.raises(ValueError, match="policy/identity"):
        GaLoreAdamW([target], lr=0.9).load_state_dict(payload)
    assert torch.equal(before, target)


def test_existing_token_role_and_engine_config_factory():
    p = torch.nn.Parameter(torch.randn(32, 16))
    p._areno_optimizer_role = "token_embedding"
    config = OptimizerConfig(galore_rank=2, galore_exclude_vocab=True)
    optimizer = build_optimizer([p], config, SimpleNamespace(world_size=1, dp_size=1))
    assert optimizer.exclude_vocab and not optimizer._projected(p)
    legacy = GaLoreAdamW([p], lr=0.003, rank=2)
    assert legacy._projected(p)
    assert legacy.state_dict()["format"] == "areno-galore-v3"
