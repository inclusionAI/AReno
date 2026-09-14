"""Vision autograd and policy-sync coverage for both Qwen3.5 backbones."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from areno.engine.config import OptimizerConfig
from areno.engine.modeling import configure_multimodal_training
from areno.engine.optim import AdamWFP32Master


@pytest.fixture(scope="module")
def qwen35_module():
    # The vision path uses native Torch. On macOS, isolate the unused GPU-only
    # import surface; fail immediately if a test ever calls a stubbed kernel.
    imported_before = set(sys.modules)
    with pytest.MonkeyPatch.context() as patch:
        if importlib.util.find_spec("triton") is None:
            from areno.accel import areno_silu_and_mul
            from areno.accel.utils import log_once

            def unavailable(*args, **kwargs):
                raise AssertionError("This CPU vision test must not execute a GPU-only kernel")

            ops = ModuleType("areno.accel.ops")
            ops.FusedMoeConfig = SimpleNamespace
            ops.SegLaMeta = SimpleNamespace
            ops.areno_fused_experts = unavailable
            ops.rms_norm_gate_fwd = unavailable
            ops.seg_la_fwd = unavailable
            ops.areno_silu_and_mul = areno_silu_and_mul
            ops.log_once = log_once
            patch.setitem(sys.modules, "areno.accel.ops", ops)
        from areno.models.qwen3_5 import model

        yield model
        # Do not leave modules bound to the optional-kernel stubs in other tests.
        if importlib.util.find_spec("triton") is None:
            for name in list(sys.modules):
                if name.startswith("areno.") and name not in imported_before:
                    module = sys.modules.pop(name)
                    parent_name, _, child_name = name.rpartition(".")
                    parent = sys.modules.get(parent_name)
                    if parent is not None and getattr(parent, child_name, None) is module:
                        delattr(parent, child_name)


@pytest.fixture(params=[False, True], ids=["dense", "moe"])
def model_factory(qwen35_module, request):
    def build():
        moe = request.param
        text = {
            "model_type": "qwen3_5_moe" if moe else "qwen3_5",
            "vocab_size": 128,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 4,
            "full_attention_interval": 1,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
        }
        adapter = qwen35_module.Qwen35MoeVLAdapter() if moe else qwen35_module.Qwen35VLAdapter()
        config = adapter.config_from_hf(
            {
                "model_type": "qwen3_5_moe" if moe else "qwen3_5",
                "text_config": text,
                "image_token_id": 99,
                "vision_config": {
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "in_channels": 3,
                    "num_heads": 2,
                    "num_position_embeddings": 16,
                    "out_hidden_size": 16,
                    "patch_size": 2,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                },
            }
        )
        config.dtype = torch.float32
        return adapter, adapter.build(config)

    return build


def _configure(model, tower, projector, *, trainable=True, tower_lr=2e-3, projector_lr=3e-3):
    configure_multimodal_training(
        model,
        OptimizerConfig(
            lr=1e-3,
            unfreeze_multimodal_tower=tower,
            unfreeze_multimodal_projector=projector,
            multimodal_tower_lr=tower_lr,
            multimodal_projector_lr=projector_lr,
        ),
        trainable=trainable,
    )


def test_vision_is_frozen_by_default(model_factory):
    _, model = model_factory()
    assert model.train() is model
    assert model.language_model.training
    assert all(p.requires_grad for p in model.language_model.parameters())
    assert all(not p.requires_grad for p in model.visual.parameters())
    assert all(not module.training for module in model.visual.modules())
    assert all(not p._areno_policy_sync for p in model.visual.parameters())


@pytest.mark.parametrize("tower,projector", [(False, False), (True, False), (False, True), (True, True)])
def test_unfreeze_backward_optimizer_and_modes(model_factory, tower, projector):
    torch.manual_seed(7)
    _, model = model_factory()
    _configure(model, tower, projector)
    model.eval()
    assert all(not module.training for module in model.visual.modules())
    model.train()
    assert model.visual.training == tower
    assert model.visual.blocks[0].training == tower
    assert model.visual.merger.training == projector
    features = {
        "pixel_values": torch.randn(4, 12),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
        "image_token_id": 99,
    }
    projected = model._project_pixel_values(features, torch.device("cpu"), batch=1)
    assert "image_embeds" not in features  # No detached feature cache across training steps.
    hidden = torch.randn(1, 3, 16, requires_grad=True)
    output = model.language_model._apply_multimodal_features(hidden, torch.tensor([[1, 99, 2]]), projected)
    output.square().mean().backward()
    received_grad = {name: parameter.grad is not None for name, parameter in model.visual.named_parameters()}
    before = {name: parameter.detach().clone() for name, parameter in model.visual.named_parameters()}
    trainable = [parameter for parameter in model.visual.parameters() if parameter.requires_grad]
    if trainable:
        optimizer = AdamWFP32Master(trainable, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0, bucket_numel=64)
        optimizer.step()
    changed_groups = set()
    for name, parameter in model.visual.named_parameters():
        is_projector = name.startswith("merger.")
        enabled = projector if is_projector else tower
        assert parameter.requires_grad == enabled
        assert received_grad[name] == enabled
        assert parameter.tensor_model_parallel is False
        assert parameter.sequence_parallel is False
        assert parameter.tp_grad_allreduce is False  # Already reduced at the text TP/SP boundary.
        if enabled:
            group = "projector" if is_projector else "tower"
            assert parameter._areno_lr_group == group
            assert parameter._areno_lr == (3e-3 if is_projector else 2e-3)
            if not torch.equal(parameter, before[name]):
                changed_groups.add(group)
        else:
            assert not hasattr(parameter, "_areno_lr")
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
    assert changed_groups == ({"tower"} if tower else set()) | ({"projector"} if projector else set())


def test_reconfigure_clears_lr_metadata_and_preserves_rollout_sync(model_factory):
    _, model = model_factory()
    _configure(model, True, True, tower_lr=None, projector_lr=0.0)
    assert model.visual.patch_embed.proj.weight._areno_lr == 1e-3
    assert model.visual.merger.linear_fc1.weight._areno_lr == 0.0
    _configure(model, True, True, trainable=False)
    for parameter in model.visual.parameters():
        assert not parameter.requires_grad
        assert parameter._areno_policy_sync
        assert not hasattr(parameter, "_areno_lr_group")
        assert not hasattr(parameter, "_areno_lr")
    assert all(not module.training for module in model.visual.modules())
    _configure(model, False, False)
    assert all(not parameter._areno_policy_sync for parameter in model.visual.parameters())


@pytest.mark.parametrize("tower,projector", [(False, False), (True, False), (False, True), (True, True)])
def test_actor_and_rollout_policy_plans_include_only_unfrozen_vision(model_factory, tower, projector):
    adapter, actor = model_factory()
    _, rollout = model_factory()
    _configure(actor, tower, projector)
    _configure(rollout, tower, projector, trainable=False)
    actor_plan = adapter.build_policy_plan(actor)
    rollout_plan = adapter.build_policy_plan(rollout)
    assert actor_plan.keys() == rollout_plan.keys()
    expected = {
        f"model.visual.{name}"
        for name, _ in actor.visual.named_parameters()
        if (projector if name.startswith("merger.") else tower)
    }
    assert {key for key in actor_plan if key.startswith("model.visual.")} == expected
    assert any("embed_tokens" in key for key in actor_plan)
    for key in expected:
        source = actor_plan[key].policy_layout()
        target = rollout_plan[key].policy_layout()
        assert source.replicated and target.replicated
        assert source.shape == target.shape
        with torch.no_grad():
            source.pieces[0].tensor.add_(1.0)
            target.pieces[0].tensor.copy_(source.pieces[0].tensor)
        name = key.removeprefix("model.visual.")
        torch.testing.assert_close(
            dict(actor.visual.named_parameters())[name], dict(rollout.visual.named_parameters())[name]
        )


def test_checkpoint_roundtrip_preserves_updated_and_frozen_visual_weights(model_factory, tmp_path):
    adapter, model = model_factory()
    _configure(model, False, True)
    with torch.no_grad():
        for parameter in model.visual.merger.parameters():
            parameter.add_(0.25)
    adapter.save_weights(model, tmp_path, source_path=None)
    _, restored = model_factory()
    adapter.load_weights(restored, tmp_path)
    for name, parameter in model.visual.named_parameters():
        torch.testing.assert_close(parameter, dict(restored.visual.named_parameters())[name], atol=0, rtol=0)
    assert all(not parameter.requires_grad for parameter in restored.visual.parameters())
