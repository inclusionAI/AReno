from contextlib import contextmanager
from types import SimpleNamespace

import torch

import areno.models.bailing.model as bailing
import areno.models.bailing_v3.model as bailing_v3
import areno.models.qwen3.model as qwen3
import areno.models.qwen3_5.model as qwen3_5


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
            gate=lambda hidden: (
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
