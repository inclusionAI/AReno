"""CPU-only checks for MLX training batch semantics."""

import numpy as np
import pytest

from areno.api.backend.common import (
    MetricReduction,
    TrainMetric,
    accumulation_group_size,
    accumulation_steps,
    metric_reduction,
)
from areno.api.backend.mlx.provider import parameter_group
from areno.api.backend.mlx.training import sft_target_token_count
from areno.api.models import TrainSequence
from areno.api.multimodal import image_token_counts_from_features, mrope_position_ids_from_image_grid


def test_sft_target_count_matches_shifted_prompt_and_loss_masks():
    rows = [
        TrainSequence(tokens=[1, 2, 3, 4], prompt_mask=[True, True, False, False]),
        TrainSequence(
            tokens=[5, 6, 7, 8, 9],
            prompt_len=1,
            loss_mask=[False, False, True, False, True],
        ),
    ]

    assert sft_target_token_count(rows) == 4


def test_sft_target_count_is_additive_across_mini_batches():
    rows = [
        TrainSequence(tokens=[1, 2, 3], prompt_len=1),
        TrainSequence(tokens=[4, 5, 6, 7], prompt_len=2),
        TrainSequence(tokens=[8, 9], prompt_len=1),
    ]

    whole = sft_target_token_count(rows)
    split = sum(sft_target_token_count(rows[start : start + 1]) for start in range(0, len(rows), 1))

    assert whole == split == 5


def test_accumulation_windows_match_cuda_mini_batch_semantics():
    assert accumulation_steps(3, None) == 3
    assert accumulation_steps(3, 0) == 1
    assert accumulation_steps(3, 2) == 2
    assert [accumulation_group_size(index, 3, 2) for index in range(3)] == [2, 2, 1]


def test_policy_metric_reductions_use_typed_names():
    assert str(TrainMetric.LOGP_ABS_DIFF_MEAN) == "logp_abs_diff_mean"
    assert str(MetricReduction.FIRST) == "first"
    assert metric_reduction(TrainMetric.LOGP_ABS_DIFF_MEAN) is MetricReduction.FIRST
    assert metric_reduction(str(TrainMetric.LOGP_ABS_DIFF_MEAN)) is MetricReduction.FIRST
    assert metric_reduction("policy_loss") is MetricReduction.MEAN


def test_multimodal_projector_group_takes_precedence_over_parent_tower():
    assert parameter_group("vision_tower.blocks.0.attn.qkv.weight") == "tower"
    assert parameter_group("vision_tower.merger.linear_fc1.weight") == "projector"


def test_multimodal_image_grid_helpers_accept_backend_native_arrays():
    features = {
        "image_grid_thw": np.array([[1, 4, 4]], dtype=np.int64),
        "spatial_merge_size": 2,
    }

    counts = image_token_counts_from_features(features)
    positions = mrope_position_ids_from_image_grid(
        [1, 99, 99, 99, 99, 2],
        image_token_id=99,
        features=features,
    )

    assert counts == [4]
    assert isinstance(positions, np.ndarray)
    assert positions.shape == (3, 6)


def test_adam8bit_lazy_state_remains_stable_after_zero_gradient_steps():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten, tree_unflatten

    from areno.api.backend.mlx.optimizer import _quantized_adamw_class, apply_optimizer_update

    model = nn.Linear(256, 1, bias=False)
    optimizer = _quantized_adamw_class()(learning_rate=1e-3, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())
    state_names = {name for name, _ in tree_flatten(optimizer.state)}
    assert not any(name.endswith(("m_q", "v_q", "m_scale", "v_scale")) for name in state_names)

    path = tree_flatten(model.trainable_parameters())[0][0]
    gradient = mx.exp(mx.linspace(-13.8155106, 0.0, 256)).reshape(model.weight.shape)
    max_updates = []
    for step in range(4):
        previous = mx.array(model.weight)
        current = gradient if step == 0 else mx.zeros_like(gradient)
        apply_optimizer_update(model, optimizer, tree_unflatten([(path, current)]))
        delta = mx.max(mx.abs(model.weight.astype(mx.float32) - previous.astype(mx.float32)))
        mx.eval(delta)
        max_updates.append(float(delta.item()))

    assert max(max_updates) < 6e-3
    assert bool(mx.all(mx.isfinite(model.weight)).item())
