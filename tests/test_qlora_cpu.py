"""NF4 format, nested scales, frozen-base derivatives and CLI contracts."""

import copy
import json

import pytest
import torch
import torch.nn.functional as F

from areno.accel.nf4 import NF4_VALUES, NF4Weight, nf4_linear
from areno.adapters import LoraConfig


def test_qlora_config_enables_paging_and_preserves_base_layout_cpu():
    from areno.api.config import CudaConfig, MlxConfig
    from areno.engine.config import EngineConfig, ModelConfig, RuntimeConfig

    lora = LoraConfig(qlora=True)
    config = EngineConfig(model=ModelConfig(), lora=lora, devices=[0])
    assert config.optimizer.paged
    with pytest.raises(ValueError, match="matching train and rollout TP"):
        CudaConfig(lora=lora, tp_size=1, rollout_tp_size=2, rollout_devices=[1, 2])
    with pytest.raises(ValueError, match="QLoRA requires CUDA"):
        MlxConfig(lora=lora)
    with pytest.raises(ValueError, match="explicit optimizer state offload"):
        EngineConfig(model=ModelConfig(), lora=lora, runtime=RuntimeConfig(optimizer_state_offload="cpu"), devices=[0])


@pytest.mark.parametrize("shape", [(1, 1), (7, 65), (64, 256), (3, 129, 65)])
def test_nf4_format_roundtrip_and_error_cpu(shape):
    torch.manual_seed(42)
    weight = torch.randn(shape)
    q = NF4Weight(weight)
    actual = q.dequantize()
    assert actual.shape == weight.shape
    assert q.packed.numel() == (weight.numel() + 1) // 2
    assert q.scale_codes.numel() == (weight.numel() + 63) // 64
    assert q.scale_absmax.numel() == (q.scale_codes.numel() + 255) // 256
    # Format error is approximate; operator correctness below is tested separately.
    relative_rmse = (actual - weight).square().mean().sqrt() / weight.square().mean().sqrt()
    assert relative_rmse < 0.12
    restored = NF4Weight(torch.zeros_like(weight))
    restored.load_state_dict(q.state_dict())
    torch.testing.assert_close(restored.dequantize(), actual, atol=0, rtol=0)


def test_nf4_zero_and_codebook_cpu():
    q = NF4Weight(torch.zeros(17, 9))
    assert torch.count_nonzero(q.dequantize()) == 0
    assert NF4_VALUES[7] == 0 and len(NF4_VALUES) == 16
    # Exact codebook values with absmax 1 survive both quantization stages.
    weight = torch.tensor(NF4_VALUES).repeat(4).reshape(8, 8)
    torch.testing.assert_close(NF4Weight(weight).dequantize(), weight, atol=0, rtol=0)


def test_nf4_saves_packed_state_and_correct_input_adapter_gradients_cpu():
    torch.manual_seed(4)
    q = NF4Weight(torch.randn(31, 65) * 0.1)
    x = torch.randn(2, 7, 65, requires_grad=True)
    a = torch.randn(8, 65, requires_grad=True)
    b = torch.randn(31, 8, requires_grad=True)
    bias = torch.randn(31, requires_grad=True)
    reference = F.linear(x, q.dequantize(), bias) + F.linear(F.linear(x, a), b)
    expected = torch.autograd.grad(reference.square().mean(), (x, a, b, bias))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
        base = nf4_linear(x, q, bias)
    assert not saved  # No dense frozen weight or full input is retained by this op.
    out = base + F.linear(F.linear(x, a), b)
    actual = torch.autograd.grad(out.square().mean(), (x, a, b, bias))
    for got, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, wanted, atol=1e-5, rtol=1e-5)


def test_nf4_nested_storage_savings_cpu():
    q = NF4Weight(torch.randn(1024, 1024))
    # ~4.127 bits/weight, plus tiny shared codebooks and mean offset.
    assert q.storage_bytes * 8 / (1024 * 1024) < 4.14


def test_qlora_adapter_metadata_cpu(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        )
    )
    (tmp_path / "areno_quantization_config.json").write_text(
        json.dumps(
            {
                "format": "nf4-dq-v1",
                "block_size": 64,
                "scale_block_size": 256,
            }
        )
    )
    assert LoraConfig(adapter_path=str(tmp_path)).qlora


def test_qlora_native_projection_replacement_cpu():
    from areno.adapters.lora import initialize_lora
    from areno.adapters.qlora import initialize_qlora
    from areno.engine.config import ModelConfig
    from areno.models.qwen3.model import Qwen3ForCausalLM

    model = Qwen3ForCausalLM(
        ModelConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
        )
    )
    registry = initialize_lora(model, LoraConfig(qlora=True), seed=4)
    adapter_ids = {id(p) for p in registry.parameters()}
    memory = initialize_qlora(model)
    assert memory["quantized_weight_bytes"] < memory["original_weight_bytes"]
    assert model.layers[0].self_attn.qkv_proj.weight is None
    assert {id(p) for p in model.parameters() if p.requires_grad} == adapter_ids
    assert not copy.deepcopy(model).layers[0].self_attn.qkv_proj.quantized_weight.packed.requires_grad
