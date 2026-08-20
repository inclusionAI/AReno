from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from areno.adapters.config import BAILING_V3_TARGETS, LoraConfig
from areno.adapters.lora import initialize_lora
from areno.engine.layers import linear
from areno.engine.layers.linear import ColumnParallelLinear, RowParallelLinear


class _ReplicatedAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = None
        self.q_a_proj = nn.Linear(8, 4, bias=False)
        self.q_b_proj = ColumnParallelLinear(4, 12, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(8, 6, bias=False)
        self.kv_b_proj = ColumnParallelLinear(4, 12, bias=False)
        self.dense = RowParallelLinear(8, 8, bias=False)
        self.lora_slots = nn.ModuleDict()

    def install_lora_component(self, component: str, slot: nn.Module) -> None:
        self.lora_slots[component] = slot


class _KDAAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_conv1d_weight = nn.Parameter(torch.empty(8, 1, 2))
        self.q_proj = ColumnParallelLinear(8, 8, bias=False)
        self.k_proj = ColumnParallelLinear(8, 8, bias=False)
        self.v_proj = ColumnParallelLinear(8, 8, bias=False)
        self.f_proj = ColumnParallelLinear(8, 8, bias=False)
        self.g_proj = ColumnParallelLinear(8, 8, bias=False)
        self.b_proj = ColumnParallelLinear(8, 1, bias=False)
        self.o_proj = RowParallelLinear(8, 8, bias=False)


class _DenseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(8, 12, bias=False)
        self.up_proj = ColumnParallelLinear(8, 12, bias=False)
        self.down_proj = RowParallelLinear(12, 8, bias=False)


class _GroupedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 8
        self.intermediate_size = 4
        self.local_num_experts = 2
        self.local_expert_start = 0
        self.linear_fc1 = nn.Linear(8, 8, bias=False)
        self.linear_fc1.weight = nn.Parameter(torch.empty(2, 8, 8))
        self.linear_fc2 = nn.Linear(4, 8, bias=False)
        self.linear_fc2.weight = nn.Parameter(torch.empty(2, 8, 4))
        self.lora_slots = nn.ModuleDict()

    def install_lora_component(self, component: str, slot: nn.Module) -> None:
        self.lora_slots[component] = slot


class _ActiveLoraSlot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(1, 3))
        self.lora_B = nn.Parameter(torch.randn(4, 1))

    @property
    def enabled(self) -> bool:
        return True


class _SparseMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = _GroupedExperts()
        self.shared_experts = _DenseMLP()


class _Layer(nn.Module):
    def __init__(self, attention: nn.Module, mlp: nn.Module) -> None:
        super().__init__()
        self.attention = attention
        self.mlp = mlp


class _BailingModel(nn.Module):
    def __init__(self, *, no_kda_lora: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(model_type="bailing_moe_v3", no_kda_lora=no_kda_lora)
        self.layers = nn.ModuleList(
            (
                _Layer(_KDAAttention(), _DenseMLP()),
                _Layer(_ReplicatedAttention(), _SparseMLP()),
            )
        )


def _single_tp() -> SimpleNamespace:
    return SimpleNamespace(rank=0, world_size=1, group=None)


def test_bailing_v3_full_profile_attaches_native_slots(monkeypatch) -> None:
    monkeypatch.setattr(linear, "get_tp_context", _single_tp)
    monkeypatch.setattr("areno.adapters.lora.get_tp_context", _single_tp)
    model = _BailingModel()
    profile = (
        "q_proj",
        "k_proj",
        "v_proj",
        "f_proj",
        "g_proj",
        "o_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "dense",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    registry = initialize_lora(model, LoraConfig(rank=4, alpha=4, target_modules=profile), seed=42)

    assert set(profile) <= set(BAILING_V3_TARGETS)
    kda_names = {
        f"layers.0.attention.{component}"
        for component in ("q_proj", "k_proj", "v_proj", "f_proj", "g_proj")
    }
    assert kda_names <= registry.slots.keys()
    assert len({id(registry.slots[name]) for name in kda_names}) == len(kda_names)
    assert "layers.0.attention.b_proj" not in registry.slots
    assert "layers.1.attention.q_a_proj" in registry.slots
    assert "layers.1.mlp.shared_experts.gate_proj" in registry.slots
    assert "layers.1.mlp.experts.{expert}.down_proj" in registry.slots
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name)
    assert all(parameter.requires_grad for parameter in registry.parameters())


def test_bailing_v3_requires_non_factorized_kda(monkeypatch) -> None:
    monkeypatch.setattr(linear, "get_tp_context", _single_tp)
    monkeypatch.setattr("areno.adapters.lora.get_tp_context", _single_tp)

    with pytest.raises(ValueError, match="no_kda_lora=true"):
        initialize_lora(
            _BailingModel(no_kda_lora=False),
            LoraConfig(target_modules=("q_proj",)),
            seed=42,
        )


def test_bailing_v3_empty_route_keeps_expert_router_and_lora_gradients(monkeypatch) -> None:
    from areno.models.bailing_v3 import model as bailing_model

    experts = bailing_model.BailingGroupedExperts.__new__(bailing_model.BailingGroupedExperts)
    nn.Module.__init__(experts)
    experts.linear_fc1 = nn.Linear(3, 4, bias=False)
    experts.linear_fc2 = nn.Linear(2, 3, bias=False)
    experts.lora_slots = nn.ModuleDict({"gate_proj": _ActiveLoraSlot()})
    experts.local_expert_start = 0
    experts.local_num_experts = 1

    flat = torch.randn(2, 3, requires_grad=True)
    router_weight = nn.Parameter(torch.randn(2, 1))
    topk_weight = router_weight.sigmoid()
    topk_idx = torch.ones(2, 1, dtype=torch.long)
    empty = flat.new_empty((0, 3))
    monkeypatch.setattr(
        bailing_model,
        "_areno_moe_topk_permute_no_compile",
        lambda *args: (
            empty,
            flat.new_empty((0,)),
            torch.empty(0, dtype=torch.long),
            torch.zeros(1, dtype=torch.long),
        ),
    )
    monkeypatch.setattr(bailing_model, "all_reduce", lambda value: value)

    experts(flat, topk_idx, topk_weight).sum().backward()

    parameters = (
        experts.linear_fc1.weight,
        experts.linear_fc2.weight,
        router_weight,
        experts.lora_slots["gate_proj"].lora_A,
        experts.lora_slots["gate_proj"].lora_B,
    )
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in parameters)
