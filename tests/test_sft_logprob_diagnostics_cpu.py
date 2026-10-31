"""Optional native SFT diagnostics preserve loss and gradient semantics."""

from __future__ import annotations

import pytest
import torch

from areno.api.backend.common import LOGP_METRIC_WEIGHT, TrainMetric, reduce_microbatch_metrics
from areno.api.backend.cuda.losses import sft_loss_fn
from areno.api.backend.cuda.training import make_train_pack
from areno.api.models import TrainSequence
from areno.engine.runtime.train_step import _pack_train_data

MARKER = "_sft_logprob_diagnostics"
LEGACY = {"sft_loss", "sft_target_tokens", "sft_logprob_mean"}


@pytest.mark.parametrize(
    "old_rows,enabled",
    [
        ([[], []], False),
        ([[0.0, -0.2, -0.3], []], False),
        ([[0.0, -0.2], [0.0, -0.4]], False),
        ([[0.0, -0.2, -0.3], [0.0, -0.4]], True),
        ([[0.0, 0.0, 0.0], [0.0, 0.0]], True),
    ],
)
def test_pack_enables_only_complete_supplied_rows(old_rows, enabled):
    seqs = [
        TrainSequence(tokens=[1, 2, 3], prompt_len=1, logprobs=old_rows[0]),
        TrainSequence(tokens=[4, 5], prompt_len=1, logprobs=old_rows[1]),
    ]
    pack = make_train_pack(seqs)
    assert (MARKER in pack) == enabled
    packed = _pack_train_data(pack)
    assert (packed.get(MARKER, False) is True) == enabled
    _, metrics = sft_loss_fn(packed, torch.zeros_like(packed["packed_logprobs"]))
    assert (TrainMetric.LOGP_ABS_DIFF_MEAN in metrics) == enabled
    if not enabled:
        assert set(metrics) == LEGACY


def test_metrics_ignore_prompt_and_loss_mask_and_shift_old_values():
    seqs = [
        TrainSequence(
            tokens=[1, 2, 3, 4, 5],
            prompt_len=2,
            loss_mask=[False, False, True, False, True],
            logprobs=[0.0, 100.0, -0.3, 200.0, -0.5],
        ),
        TrainSequence(tokens=[6, 7, 8], prompt_len=1, logprobs=[0.0, -0.8, -0.6]),
    ]
    packed = _pack_train_data(make_train_pack(seqs))
    # Packed action axis excludes each first token; prompt/loss-masked values
    # are deliberately huge so accidentally including them is detectable.
    torch.testing.assert_close(packed["packed_logprobs"], torch.tensor([100.0, -0.3, 200.0, -0.5, -0.8, -0.6]))
    assert packed["packed_response_mask"].tolist() == [False, True, False, True, True, True]
    current = torch.tensor([-99.0, -0.1, -88.0, -0.7, -0.9, -0.2], requires_grad=True)
    _, metrics = sft_loss_fn(packed, current)
    assert float(metrics[LOGP_METRIC_WEIGHT]) == 4
    torch.testing.assert_close(metrics[TrainMetric.ROLLOUT_LOGPROBS_MEAN], torch.tensor(-0.55))
    torch.testing.assert_close(metrics[TrainMetric.TRAIN_LOGPROBS_MEAN], torch.tensor(-0.475))
    torch.testing.assert_close(metrics[TrainMetric.LOGP_DIFF_MEAN], torch.tensor(-0.075))
    torch.testing.assert_close(metrics[TrainMetric.LOGP_ABS_DIFF_MEAN], torch.tensor(0.225))
    assert all(not value.requires_grad for value in metrics.values())


@pytest.mark.parametrize("normalization", [{}, {"_sft_total_target_tokens": 7, "_sft_grad_scale": 3}])
def test_diagnostics_leave_loss_and_gradients_identical(normalization):
    mask = torch.tensor([False, True, False, True])
    before = torch.tensor([-99.0, -0.2, -88.0, -0.8], requires_grad=True)
    after = before.detach().clone().requires_grad_()
    old = torch.tensor([-100.0, -0.5, -200.0, -0.2], requires_grad=True)
    legacy = {"packed_response_mask": mask, **normalization}
    diagnostic = {**legacy, MARKER: True, "packed_logprobs": old}
    loss_before, metrics_before = sft_loss_fn(legacy, before)
    loss_after, metrics_after = sft_loss_fn(diagnostic, after)
    loss_before.backward()
    loss_after.backward()
    assert torch.equal(loss_before, loss_after)
    assert torch.equal(before.grad, after.grad)
    assert old.grad is None
    for key in LEGACY:
        assert torch.equal(metrics_before[key], metrics_after[key])
    assert float(metrics_after[LOGP_METRIC_WEIGHT]) == 2
    for value in metrics_after.values():
        assert not value.requires_grad


def test_token_weighted_reduction_matches_response_token_mean():
    metrics = []
    for count, difference in [(1, 2.0), (3, 0.0)]:
        current = torch.full((count,), -0.5)
        pack = {
            MARKER: True,
            "packed_response_mask": torch.ones(count, dtype=torch.bool),
            "packed_logprobs": current + difference,
        }
        _, row = sft_loss_fn(pack, current)
        metrics.append(row)
    reduced = reduce_microbatch_metrics(metrics)
    assert reduced[str(TrainMetric.LOGP_ABS_DIFF_MEAN)] == pytest.approx(0.5)
    assert reduced[str(TrainMetric.LOGP_DIFF_MEAN)] == pytest.approx(0.5)
    assert LOGP_METRIC_WEIGHT not in reduced


def test_false_marker_retains_exact_legacy_metric_set_without_old_logprobs():
    _, metrics = sft_loss_fn({MARKER: False, "packed_response_mask": torch.tensor([True])}, torch.tensor([-0.7]))
    assert set(metrics) == LEGACY
