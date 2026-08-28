from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


@contextmanager
def _record_region(calls, active):
    calls.append(("region", active))
    yield


def _install_sequence_collectives(monkeypatch, module, calls):
    monkeypatch.setattr(module, "is_sequence_parallel_active", lambda: True)
    monkeypatch.setattr(
        module,
        "gather_from_sequence_parallel_region",
        lambda x: calls.append("gather") or torch.cat((x, x + 10), dim=1),
    )
    monkeypatch.setattr(
        module,
        "scatter_to_sequence_parallel_region",
        lambda x: calls.append("scatter") or x[:, : x.shape[1] // 2],
    )
    monkeypatch.setattr(module, "sequence_parallel_region", lambda active: _record_region(calls, active))


def test_qwen3_moe_gathers_before_routing_and_scatters_complete_expert_output(monkeypatch):
    pytest.importorskip("triton")
    import areno.models.qwen3.model as qwen3

    calls = []
    _install_sequence_collectives(monkeypatch, qwen3, calls)
    monkeypatch.setattr(qwen3, "_areno_linear_no_compile", lambda x, weight: x)
    monkeypatch.setattr(
        qwen3,
        "_areno_topk_softmax_no_compile",
        lambda logits, top_k, norm: (
            torch.zeros((logits.shape[0], top_k), dtype=torch.long),
            torch.ones((logits.shape[0], top_k)),
        ),
    )
    mlp = SimpleNamespace(
        gate=torch.zeros((2, 2)),
        top_k=1,
        norm_topk_prob=True,
        training=True,
        experts=lambda flat, indices, weights: flat * 2,
    )
    hidden = torch.ones((1, 2, 2))

    indices, weights = qwen3.Qwen3MoeMLP.route(mlp, hidden)
    output = qwen3.Qwen3MoeMLP.forward_with_routes(mlp, hidden, indices, weights)

    assert indices.shape == weights.shape == (4, 1)
    assert output.shape == hidden.shape
    assert calls == ["gather", ("region", False), "gather", ("region", False), "scatter"]


def test_qwen35_moe_gathers_once_and_scatters_complete_output(monkeypatch):
    pytest.importorskip("triton")
    import areno.models.qwen3_5.model as qwen3_5

    calls = []
    _install_sequence_collectives(monkeypatch, qwen3_5, calls)
    monkeypatch.setattr(qwen3_5, "_areno_linear_no_compile", lambda x, weight: x)
    monkeypatch.setattr(
        qwen3_5,
        "_areno_topk_softmax_no_compile",
        lambda logits, top_k, norm: (
            torch.zeros((logits.shape[0], top_k), dtype=torch.long),
            torch.ones((logits.shape[0], top_k)),
        ),
    )

    def shared_expert(states):
        calls.append(("shared", states.shape[1]))
        return states * 3

    mlp = SimpleNamespace(
        gate=torch.zeros((2, 2)),
        top_k=1,
        norm_topk_prob=True,
        training=True,
        experts=lambda flat, indices, weights: flat * 2,
        shared_expert=shared_expert,
        shared_expert_gate=None,
    )
    hidden = torch.ones((1, 2, 2))

    output = qwen3_5.Qwen35MoeMLP.forward(mlp, hidden)

    assert output.shape == hidden.shape
    assert calls == ["gather", ("region", False), "scatter", ("shared", 2)]
    torch.testing.assert_close(output, torch.full_like(output, 5))


def test_bailing_moe_scatters_already_reduced_expert_output(monkeypatch):
    pytest.importorskip("triton")
    import areno.models.bailing.model as bailing
    import areno.models.bailing_v3.model as bailing_v3

    class Experts:
        linear_fc1 = SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))

        def __call__(self, flat, indices, weights):
            return flat * 2

    for module in (bailing, bailing_v3):
        calls = []
        _install_sequence_collectives(monkeypatch, module, calls)

        def shared_experts(states):
            calls.append(("shared", states.shape[1]))
            return states * 3

        block = SimpleNamespace(
            training=True,
            gate=lambda hidden, num_padding_tokens=0: (
                torch.zeros((hidden.numel() // hidden.shape[-1], 1), dtype=torch.long),
                torch.ones((hidden.numel() // hidden.shape[-1], 1)),
                None,
            ),
            experts=Experts(),
            shared_experts=shared_experts,
        )

        output = module.BailingSparseMoeBlock.forward(block, torch.ones((1, 2, 2)))

        assert output.shape == (1, 2, 2)
        assert calls == ["gather", ("region", False), "scatter", ("shared", 2)]
        torch.testing.assert_close(output.float(), torch.full_like(output.float(), 5))


def test_bailing_router_load_ignores_alignment_tokens(monkeypatch):
    pytest.importorskip("triton")
    import areno.models.bailing.model as bailing
    import areno.models.bailing_v3.model as bailing_v3

    topk_idx = torch.tensor([[0], [1], [1], [0]])
    topk_weight = torch.ones_like(topk_idx, dtype=torch.float32)
    for module in (bailing, bailing_v3):
        monkeypatch.setattr(module, "_areno_linear_no_compile", lambda x, weight: x)
        gate = SimpleNamespace(
            weight=torch.zeros((2, 2)),
            num_experts=2,
            local_tokens_per_expert=torch.zeros(2),
            _forward_grouped_topk=lambda logits: (topk_idx, topk_weight),
        )

        module.BailingGate.forward(gate, torch.ones((1, 4, 2)), num_padding_tokens=1)

        torch.testing.assert_close(gate.local_tokens_per_expert, torch.tensor([1.0, 2.0]))


def test_gemma4_moe_routes_local_shard_then_gathers_expert_inputs(monkeypatch):
    pytest.importorskip("triton")
    import areno.models.gemma4.model as gemma4

    calls = []
    _install_sequence_collectives(monkeypatch, gemma4, calls)
    monkeypatch.setattr(
        gemma4,
        "checkpoint_layer",
        lambda function, *args, train_meta=None, infer_meta=None: function(*args),
    )

    class Moe:
        def route(self, logits):
            calls.append(("route", logits.shape[1]))
            count = logits.shape[0] * logits.shape[1]
            return torch.zeros((count, 1), dtype=torch.long), torch.ones((count, 1))

        def forward_with_routes(self, states, indices, weights):
            calls.append(("experts", states.shape[1], indices.shape[0], weights.shape[0]))
            return states * 2

    layer = SimpleNamespace(
        mlp=lambda states: states * 3,
        router=lambda states: states,
        moe=Moe(),
        pre_feedforward_layernorm_2=lambda states: states,
        post_feedforward_layernorm_1=lambda states: states,
        post_feedforward_layernorm_2=lambda states: states,
    )
    hidden = torch.ones((1, 2, 2))

    output = gemma4._gemma4_moe_feedforward_no_compile(layer, hidden, hidden, None, None)

    assert output.shape == hidden.shape
    assert calls == [
        ("route", 2),
        "gather",
        "gather",
        "gather",
        ("region", False),
        ("experts", 4, 4, 4),
        "scatter",
    ]
    torch.testing.assert_close(output, torch.full_like(output, 5))
