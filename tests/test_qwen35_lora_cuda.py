"""Hybrid attention backward against the same dequantized frozen base."""

import copy

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("qlora", [False, True], ids=["lora", "qlora"])
def test_qwen35_hybrid_adapter_loss_and_gradients(qlora):
    from areno.adapters import LoraConfig
    from areno.adapters.config import QWEN35_TARGETS
    from areno.adapters.lora import initialize_lora
    from areno.adapters.qlora import initialize_qlora
    from areno.engine.config import ModelConfig
    from areno.engine.runtime.metadata import TrainMeta
    from areno.models.qwen3_5.model import Qwen35ForCausalLM

    torch.manual_seed(417)
    model = Qwen35ForCausalLM(
        ModelConfig(
            model_type="qwen3_5",
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            layer_types=("linear_attention", "full_attention"),
            attn_output_gate=True,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            dtype=torch.bfloat16,
            attn_backend="native",
        )
    ).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    registry = initialize_lora(model, LoraConfig(rank=4, alpha=8, target_modules=QWEN35_TARGETS, qlora=qlora), seed=9)
    with torch.no_grad():
        for slot in registry.slots.values():
            slot.lora_B.normal_(0, 0.02)
    reference = copy.deepcopy(model)
    if qlora:
        initialize_qlora(model)
        for name, module in model.named_modules():
            quantized = getattr(module, "quantized_weight", None)
            if quantized is not None:
                reference.get_submodule(name).weight.data.copy_(quantized.dequantize())
    tokens = torch.randint(128, (1, 24), device="cuda")
    positions = torch.cat((torch.arange(8), torch.arange(16))).to("cuda")[None]
    meta = TrainMeta(
        cu_seqlens=torch.tensor([0, 8, 24], device="cuda", dtype=torch.int32),
        max_seqlen=16,
        packed=True,
        activation_checkpointing=True,
    )
    outputs, gradients = [], []
    for candidate in (reference, model):
        logits = candidate(tokens, position_ids=positions, train_meta=meta).logits_shard
        loss = F.cross_entropy(logits[0, :-1].float(), tokens[0, 1:])
        outputs.append(loss.detach())
        parameters = [p for p in candidate.parameters() if p.requires_grad]
        gradients.append(torch.autograd.grad(loss, parameters))
    torch.testing.assert_close(outputs[0], outputs[1], atol=2e-4, rtol=2e-4)
    for expected, actual in zip(*gradients, strict=True):
        assert torch.isfinite(actual).all() and actual.abs().sum() > 0
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-2)
