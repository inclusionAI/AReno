"""CPU regression tests for the proposed Stage 4 sampled-token OPD objective.

Run: python -m pytest tests/test_opd_loss_cpu.py -v
No model download, CUDA, or external service is required.
"""

from __future__ import annotations

import math

import pytest
import torch

from areno.api.loss_fns.opd import opd_loss_fn
from areno.api.algorithms import get_algorithm
from areno.api.models import TrainSequence
from areno.api.trainer_config import OPDTrainerConfig


def _padded(*, old=(-1.0, -1.0), teacher=(-0.5, -2.0), current=(-0.8, -1.2)):
    # Four token IDs: two prompts + two sampled responses.  Three training
    # next-token action positions: one prompt, two response tokens.
    pack = {
        "prompt_mask": torch.tensor([[True, True, False, False]]),
        "logprobs": torch.tensor([[0.0, 0.0, *old]], dtype=torch.float32),
        "ref_logprobs": torch.tensor([[0.0, 0.0, *teacher]], dtype=torch.float32),
    }
    current_logprobs = torch.tensor(
        [[-9.0, *current]], dtype=torch.float32, requires_grad=True
    )
    return pack, current_logprobs


def _packed(*, old=(-1.0, -1.0), teacher=(-0.5, -2.0), current=(-0.8, -1.2)):
    pack = {
        "packed_response_mask": torch.tensor([False, True, True]),
        "packed_logprobs": torch.tensor([0.0, *old]),
        "packed_ref_logprobs": torch.tensor([0.0, *teacher]),
    }
    logprobs = torch.tensor([-9.0, *current], requires_grad=True)
    return pack, logprobs


def test_hand_calculated_loss_and_gradients():
    pack, current = _padded()
    loss, stats = opd_loss_fn(pack, current)
    r1 = math.exp(0.2)  # current - old = -0.8 - (-1.0)
    r2 = math.exp(-0.2)
    expected = -(r1 * 0.5 + r2 * -1.0) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-6)
    assert stats["opd_advantage_mean"].item() == pytest.approx(-0.25)
    assert stats["ratio_mean"].item() == pytest.approx((r1 + r2) / 2, rel=1e-6)
    loss.backward()
    assert current.grad[0, 0].item() == 0.0
    assert current.grad[0, 1].item() == pytest.approx(-r1 * 0.5 / 2, rel=1e-6)
    assert current.grad[0, 2].item() == pytest.approx(r2 / 2, rel=1e-6)


def test_packed_matches_padded_value_and_gradient():
    padded_pack, padded_logp = _padded()
    packed_pack, packed_logp = _packed()
    lp, sp = opd_loss_fn(padded_pack, padded_logp)
    lk, sk = opd_loss_fn(packed_pack, packed_logp)
    assert lk.item() == pytest.approx(lp.item(), abs=1e-7)
    lp.backward()
    lk.backward()
    torch.testing.assert_close(padded_logp.grad[0], packed_logp.grad)
    for key in ("ratio_mean", "opd_advantage_mean", "teacher_logprobs_mean", "opd_response_tokens"):
        assert sp[key].item() == pytest.approx(sk[key].item())


@pytest.mark.parametrize("coef", [0.5, 2.0, 3.0])
def test_coefficient_scales_loss_and_gradients(coef):
    pack, base = _padded()
    _, unscaled = opd_loss_fn(pack, base)
    scaled_loss, scaled_stats = opd_loss_fn(pack, base, opd_coef=coef)
    assert scaled_loss.item() == pytest.approx(coef * unscaled["opd_loss"].item())
    scaled_loss.backward()
    r1, r2 = math.exp(0.2), math.exp(-0.2)
    assert base.grad[0, 1].item() == pytest.approx(-coef * r1 * .5 / 2)
    assert base.grad[0, 2].item() == pytest.approx(coef * r2 / 2)
    assert scaled_stats["opd_loss"].item() == pytest.approx(scaled_loss.item())


def test_positive_teacher_signal_increases_logprob_and_negative_decreases():
    # Here -d(loss)/d(logp) is the optimizer's direction for the sample.
    pack, current = _padded(old=(-1.0, -1.0), teacher=(-0.1, -2.0), current=(-1.0, -1.0))
    loss, _ = opd_loss_fn(pack, current)
    loss.backward()
    assert current.grad[0, 1] < 0  # gradient descent raises student logprob
    assert current.grad[0, 2] > 0  # gradient descent lowers student logprob


def test_equal_teacher_and_rollout_scores_yield_zero_loss_and_gradients():
    pack, current = _padded(teacher=(-1.0, -1.0))
    loss, stats = opd_loss_fn(pack, current)
    assert loss.item() == pytest.approx(0)
    assert stats["opd_advantage_mean"].item() == pytest.approx(0)
    loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_teacher_and_rollout_tensors_are_detached():
    pack, current = _padded()
    old = pack["logprobs"].clone().requires_grad_()
    teacher = pack["ref_logprobs"].clone().requires_grad_()
    pack["logprobs"], pack["ref_logprobs"] = old, teacher
    loss, stats = opd_loss_fn(pack, current)
    assert all(not value.requires_grad for value in stats.values())
    loss.backward()
    assert old.grad is None
    assert teacher.grad is None
    assert current.grad is not None and current.grad.abs().sum().item() > 0


def test_prompt_and_loss_mask_exclusions_have_zero_gradient():
    pack = {
        "prompt_mask": torch.tensor([[True, False, False, False, True]]),
        "loss_mask": torch.tensor([[False, True, False, True, False]]),
        "logprobs": torch.tensor([[0.0, -1.0, -2.0, -3.0, 0.0]]),
        "ref_logprobs": torch.tensor([[0.0, -0.5, -0.1, -3.5, 0.0]]),
    }
    current = torch.tensor([[-1.0, -200.0, -3.0, -300.0]], requires_grad=True)
    loss, stats = opd_loss_fn(pack, current)
    # Only response action positions 0 and 2 survive loss_mask.
    assert loss.item() == pytest.approx(0.0)
    assert stats["opd_response_tokens"].item() == 2
    loss.backward()
    assert current.grad[0, 0].item() == pytest.approx(-0.5 / 2)
    assert current.grad[0, 1].item() == 0.0
    assert current.grad[0, 2].item() == pytest.approx(+0.5 / 2)
    assert current.grad[0, 3].item() == 0.0


def test_padded_masked_nan_values_are_ignored_without_poisoning_loss():
    pack, current = _padded()
    pack["logprobs"][0, 1] = float("nan")
    pack["ref_logprobs"][0, 1] = float("inf")
    with torch.no_grad():
        current[0, 0] = float("nan")
    loss, stats = opd_loss_fn(pack, current)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in stats.values())
    loss.backward()
    assert current.grad[0, 0].item() == 0.0
    assert torch.isfinite(current.grad).all()


def test_packed_masked_nan_values_are_ignored():
    pack, current = _packed()
    pack["packed_logprobs"][0] = float("nan")
    pack["packed_ref_logprobs"][0] = float("inf")
    with torch.no_grad():
        current[0] = float("nan")
    loss, _ = opd_loss_fn(pack, current)
    assert torch.isfinite(loss)
    loss.backward()
    assert current.grad[0].item() == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("name", ["student", "rollout", "teacher"])
def test_nonfinite_active_scores_fail_fast(bad, name):
    pack, current = _padded()
    with torch.no_grad():
        if name == "student":
            current[0, 1] = bad
        else:
            pack["logprobs" if name == "rollout" else "ref_logprobs"][0, 2] = bad
    with pytest.raises(ValueError, match=f"OPD {name} contains non-finite"):
        opd_loss_fn(pack, current)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("missing", ["old", "teacher"])
def test_required_score_fields_are_not_silently_defaulted(packed, missing):
    pack, logp = _packed() if packed else _padded()
    key = (
        ("packed_logprobs" if packed else "logprobs")
        if missing == "old" else ("packed_ref_logprobs" if packed else "ref_logprobs")
    )
    del pack[key]
    with pytest.raises(ValueError, match="requires .*logprobs|requires teacher scores"):
        opd_loss_fn(pack, logp)


@pytest.mark.parametrize("packed", [False, True])
def test_shape_mismatch_fails_clearly(packed):
    pack, logp = _packed() if packed else _padded()
    key = "packed_ref_logprobs" if packed else "ref_logprobs"
    pack[key] = pack[key][..., :-1]
    with pytest.raises(ValueError, match="shapes must match"):
        opd_loss_fn(pack, logp)


@pytest.mark.parametrize("packed", [False, True])
def test_all_response_tokens_masked_fails_fast(packed):
    pack, logp = _packed() if packed else _padded()
    if packed:
        pack["packed_response_mask"][:] = False
    else:
        pack["prompt_mask"][:] = True
    with pytest.raises(ValueError, match="at least one unmasked"):
        opd_loss_fn(pack, logp)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_loss_coefficient_is_rejected(bad):
    pack, logp = _padded()
    with pytest.raises(ValueError, match="opd_coef"):
        opd_loss_fn(pack, logp, opd_coef=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, 21.0, float("nan"), float("inf")])
def test_invalid_clamp_bound_is_rejected(bad):
    pack, logp = _padded()
    with pytest.raises(ValueError, match="max_log_ratio"):
        opd_loss_fn(pack, logp, max_log_ratio=bad)


def test_extreme_ratios_are_bounded_and_clamp_fraction_measured():
    pack, current = _padded(old=(-1000.0, -1.0), teacher=(-999.0, -2.0), current=(-1.0, -1000.0))
    loss, stats = opd_loss_fn(pack, current, max_log_ratio=2.0)
    assert torch.isfinite(loss)
    assert stats["opd_ratio_clamped_fraction"].item() == pytest.approx(1.0)
    loss.backward()
    assert torch.isfinite(current.grad).all()
    assert current.grad[0, 1].item() == pytest.approx(0.0)
    assert current.grad[0, 2].item() == pytest.approx(0.0)


def test_log_ratio_in_nearby_region_still_has_gradient():
    pack, current = _padded()
    loss, stats = opd_loss_fn(pack, current, max_log_ratio=2.0)
    loss.backward()
    assert stats["opd_ratio_clamped_fraction"].item() == 0.0
    assert current.grad[0, 1].item() < 0


def test_metrics_are_scalar_detached_tensors():
    pack, current = _padded()
    _, stats = opd_loss_fn(pack, current)
    for name in (
        "opd_loss", "policy_loss", "total_loss", "opd_response_tokens",
        "opd_advantage_mean", "ratio_mean", "ratio_std",
        "opd_ratio_clamped_fraction", "teacher_logprobs_mean",
        "rollout_logprobs_mean", "train_logprobs_mean", "logp_diff_mean",
    ):
        assert name in stats
        assert stats[name].ndim == 0
        assert not stats[name].requires_grad


def test_registry_binds_opd_coefficient_to_the_real_loss():
    cfg = OPDTrainerConfig(algo="opd", ckpt="student", teacher_ckpt="teacher", dataset_path="prompts", opd_coef=2.5)
    fn = get_algorithm("opd").make_loss_fn(cfg)
    pack, current = _padded()
    direct, _ = opd_loss_fn(pack, current, opd_coef=2.5)
    registered, _ = fn(pack, current)
    assert registered.item() == pytest.approx(direct.item())


def _sequences():
    # Second sequence is shorter; padding must not contribute to any metric.
    return [
        TrainSequence(
            tokens=[11, 12, 31, 32],
            prompt_mask=[True, True, False, False],
            logprobs=[0.0, 0.0, -1.0, -1.2],
            ref_logprobs=[-9.0, -9.0, -0.5, -1.9],
            eos_token_id=999,
        ),
        TrainSequence(
            tokens=[21, 41, 42],
            prompt_mask=[True, False, False],
            logprobs=[0.0, -1.1, -0.9],
            ref_logprobs=[-9.0, -0.2, -1.3],
            eos_token_id=999,
        ),
    ]


def test_real_backend_packing_preserves_teacher_without_critic():
    from areno.api.backend.cuda.training import make_train_pack as _make_train_pack
    from areno.engine.runtime.train_step import _pack_train_data

    padded = _make_train_pack(_sequences())
    assert "ref_logprobs" in padded
    assert "values" not in padded and "returns" not in padded
    packed = _pack_train_data(padded)
    assert "packed_ref_logprobs" in packed
    assert "packed_values" not in packed and "packed_returns" not in packed
    torch.testing.assert_close(
        packed["packed_ref_logprobs"],
        torch.tensor([-9.0, -0.5, -1.9, -0.2, -1.3]),
    )
    torch.testing.assert_close(
        packed["packed_logprobs"],
        torch.tensor([0.0, -1.0, -1.2, -1.1, -0.9]),
    )
    assert packed["packed_response_mask"].tolist() == [False, True, True, True, True]
    assert packed["packed_seq_ids"].tolist() == [0, 0, 0, 1, 1]


def test_real_backend_pack_loss_and_gradients_match_padded():
    from areno.api.backend.cuda.training import make_train_pack as _make_train_pack
    from areno.engine.runtime.train_step import _pack_train_data

    padded_pack = _make_train_pack(_sequences())
    packed_pack = _pack_train_data(padded_pack)
    # Predicted action logprobs (not token-zero) in rectangular and packed order.
    padded_current = torch.tensor(
        [[-8.0, -0.85, -1.35], [-0.9, -1.25, -99.0]],
        requires_grad=True,
    )
    packed_current = torch.tensor(
        [-8.0, -0.85, -1.35, -0.9, -1.25], requires_grad=True,
    )
    loss_pad, stats_pad = opd_loss_fn(padded_pack, padded_current)
    loss_pack, stats_pack = opd_loss_fn(packed_pack, packed_current)
    assert loss_pad.item() == pytest.approx(loss_pack.item(), rel=1e-6)
    assert stats_pad["opd_response_tokens"].item() == stats_pack["opd_response_tokens"].item() == 4
    loss_pad.backward()
    loss_pack.backward()
    expected_padded_flat = torch.cat((padded_current.grad[0], padded_current.grad[1, :2]))
    torch.testing.assert_close(expected_padded_flat, packed_current.grad)
    assert padded_current.grad[1, 2].item() == 0


def test_packing_independently_keeps_optional_fields():
    from areno.api.backend.cuda.training import make_train_pack as _make_train_pack
    from areno.engine.runtime.train_step import _pack_train_data
    seqs = _sequences()
    # PPO-like fields should still work unchanged.
    for seq in seqs:
        seq.values = [0.0] * len(seq.tokens)
        seq.returns = [1.0] * len(seq.tokens)
    packed = _pack_train_data(_make_train_pack(seqs))
    for key in ("packed_ref_logprobs", "packed_values", "packed_returns"):
        assert key in packed
    # One optional field alone should not disappear.
    for seq in seqs:
        seq.ref_logprobs = []
        seq.returns = []
    only_values = _pack_train_data(_make_train_pack(seqs))
    assert "packed_values" in only_values
    assert "packed_ref_logprobs" not in only_values
    assert "packed_returns" not in only_values
