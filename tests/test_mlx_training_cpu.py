"""CPU-only checks for MLX training batch semantics."""

import logging
import sys
from collections import deque
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from areno.api.backend.common import (
    LOGP_METRIC_WEIGHT,
    MetricReduction,
    TrainMetric,
    accumulation_group_size,
    accumulation_steps,
    metric_reduction,
    reduce_microbatch_metrics,
)
from areno.api.backend.mlx.generation import ContinuousBatchScheduler, GenerationConfig, _Request
from areno.api.backend.mlx.provider import parameter_group
from areno.api.backend.mlx.training import sft_target_token_count
from areno.api.models import TrainSequence
from areno.api.multimodal import image_token_counts_from_features, mrope_position_ids_from_image_grid


def _require_mlx_device(mx):
    try:
        probe = mx.zeros((1,))
        mx.eval(probe)
    except RuntimeError as exc:
        if "No Metal device available" in str(exc):
            pytest.skip("MLX device is unavailable in this environment")
        raise


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
    assert metric_reduction(TrainMetric.LOGP_ABS_DIFF_MEAN) is MetricReduction.WEIGHTED_MEAN
    assert metric_reduction(str(TrainMetric.LOGP_ABS_DIFF_MEAN)) is MetricReduction.WEIGHTED_MEAN
    assert metric_reduction(TrainMetric.RATIO_MEAN) is MetricReduction.FIRST
    assert metric_reduction("policy_loss") is MetricReduction.MEAN


def test_logprob_metrics_are_weighted_by_active_tokens_across_microbatches():
    reduced = reduce_microbatch_metrics(
        [
            {
                TrainMetric.LOGP_ABS_DIFF_MEAN: 0.1,
                TrainMetric.TRAIN_LOGPROBS_MEAN: -0.2,
                TrainMetric.RATIO_MEAN: 1.0,
                "policy_loss": 2.0,
                LOGP_METRIC_WEIGHT: 2.0,
            },
            {
                TrainMetric.LOGP_ABS_DIFF_MEAN: 0.01,
                TrainMetric.TRAIN_LOGPROBS_MEAN: -0.5,
                TrainMetric.RATIO_MEAN: 1.1,
                "policy_loss": 4.0,
                LOGP_METRIC_WEIGHT: 8.0,
            },
        ]
    )

    assert reduced[TrainMetric.LOGP_ABS_DIFF_MEAN] == pytest.approx(0.028)
    assert reduced[TrainMetric.TRAIN_LOGPROBS_MEAN] == pytest.approx(-0.44)
    assert reduced[TrainMetric.RATIO_MEAN] == pytest.approx(1.0)
    assert reduced["policy_loss"] == pytest.approx(3.0)
    assert LOGP_METRIC_WEIGHT not in reduced


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


def test_mlx_decode_progress_matches_cuda_log_shape(monkeypatch, caplog):
    scheduler = object.__new__(ContinuousBatchScheduler)
    scheduler._config = GenerationConfig(
        max_running_prompts=2,
        completion_batch_size=2,
        prefill_batch_size=2,
        prefill_step_size=128,
        max_kv_size=None,
        decode_progress_interval_s=10.0,
    )
    scheduler._requests_by_handle = {(None, 1): object(), (None, 2): object()}
    scheduler._decode_progress_next_time = 0.0
    scheduler._decode_progress_window_start = 0.0
    scheduler._decode_progress_window_tokens = 0
    timestamps = iter((1.0, 2.0, 12.0))
    monkeypatch.setattr("areno.api.backend.mlx.generation.time.perf_counter", lambda: next(timestamps))

    with caplog.at_level(logging.INFO, logger="areno.api.backend.mlx.generation"):
        scheduler._record_decode_progress(0)
        scheduler._record_decode_progress(2)
        scheduler._record_decode_progress(20)

    assert caplog.messages == ["rollout decode progress: dp=0/1 active=2 cuda_graph=False tokens_per_second=2.0"]


def test_mlx_scheduler_admission_respects_max_running_prompts():
    class FakeGenerator:
        def __init__(self):
            self.next_uid = 0

        def insert(self, prompts, features, sampling):
            del features, sampling
            uids = list(range(self.next_uid, self.next_uid + len(prompts)))
            self.next_uid += len(prompts)
            return uids

    scheduler = object.__new__(ContinuousBatchScheduler)
    scheduler._config = GenerationConfig(
        max_running_prompts=2,
        completion_batch_size=8,
        prefill_batch_size=8,
        prefill_step_size=128,
        max_kv_size=None,
        decode_progress_interval_s=0.0,
    )
    scheduler._provider = SimpleNamespace(is_multimodal=False)
    generator = FakeGenerator()
    scheduler._generators = {None: generator}
    scheduler._requests_by_handle = {}
    scheduler._pending_requests = deque()
    sampling = SimpleNamespace()
    request = _Request(prompts=[[1], [2], [3]], n_samples=1, sampling=sampling, features=None)

    scheduler._insert_or_fail(request)

    assert len(scheduler._requests_by_handle) == 2
    assert request.next_insert == 2
    first_handles = list(request.handles)
    for handle in first_handles:
        scheduler._record_response(handle[0], generator, SimpleNamespace(uid=handle[1], finish_reason="stop"))
    scheduler._admit_pending()

    assert len(scheduler._requests_by_handle) == 1
    assert request.next_insert == 3
    final_handle = request.handles[-1]
    scheduler._record_response(final_handle[0], generator, SimpleNamespace(uid=final_handle[1], finish_reason="stop"))
    assert request.future.done()
    assert len(request.future.result()) == 3


def test_mlx_backend_uses_default_config_before_loading_provider(monkeypatch):
    from areno.api.backend.mlx.backend import MlxBackend
    from areno.api.config import MlxConfig

    mlx_module = ModuleType("mlx")
    mlx_core_module = ModuleType("mlx.core")
    mlx_module.core = mlx_core_module
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)

    class ProviderLoadReached(Exception):
        pass

    def load_provider(model_path, *, adapter_path=None):
        assert model_path == "model"
        assert adapter_path is None
        raise ProviderLoadReached

    monkeypatch.setattr("areno.api.backend.mlx.backend.load_provider", load_provider)
    backend = MlxBackend()
    ctx = SimpleNamespace(world_size=1, custom_config=None, model_path="model")

    with pytest.raises(ProviderLoadReached):
        backend.initialize(ctx)

    assert backend.config == MlxConfig()


def test_adam8bit_lazy_state_remains_stable_after_zero_gradient_steps():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten, tree_unflatten

    from areno.api.backend.mlx.optimizer import _quantized_adamw_class, apply_optimizer_update

    _require_mlx_device(mx)
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


def test_mlx_fp_optimizer_enables_bias_correction():
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.optimizers")

    from areno.api.backend.mlx.optimizer import _adamw

    assert _adamw({}).bias_correction is True


def test_adam8bit_matches_bias_corrected_adamw_for_uniform_moments():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    optim = pytest.importorskip("mlx.optimizers")
    from mlx.utils import tree_flatten, tree_unflatten

    from areno.api.backend.mlx.optimizer import _quantized_adamw_class, apply_optimizer_update

    _require_mlx_device(mx)
    reference_model = nn.Linear(256, 1, bias=False)
    quantized_model = nn.Linear(256, 1, bias=False)
    quantized_model.update(reference_model.parameters())
    reference = optim.AdamW(learning_rate=1e-3, weight_decay=0.01, bias_correction=True)
    quantized = _quantized_adamw_class()(learning_rate=1e-3, weight_decay=0.01)
    path = tree_flatten(reference_model.trainable_parameters())[0][0]

    for value in (0.25, -0.5, 0.125):
        gradient = mx.full(reference_model.weight.shape, value)
        reference.update(reference_model, tree_unflatten([(path, gradient)]))
        apply_optimizer_update(quantized_model, quantized, tree_unflatten([(path, gradient)]))
        mx.eval(reference_model.parameters(), quantized_model.parameters())

    assert bool(mx.allclose(reference_model.weight, quantized_model.weight, atol=2e-5, rtol=2e-5).item())
