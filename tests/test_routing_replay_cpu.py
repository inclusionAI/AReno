from __future__ import annotations

import torch

from areno.api.backend.cuda.training import make_train_pack
from areno.api.models import TrainSequence
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.runtime.metadata import InferMeta, TrainMeta
from areno.engine.runtime.routing_replay import (
    captured_routing,
    resolve_sigmoid_routes,
    resolve_softmax_routes,
    routing_replay_context,
)


def test_softmax_routing_capture_and_replay_keeps_current_router_gradients():
    logits = torch.tensor([[1.0, 3.0, 2.0], [3.0, 2.0, 1.0]], requires_grad=True)
    dynamic_idx = torch.tensor([[1, 2], [0, 1]])
    dynamic_weight = torch.softmax(logits, dim=-1).gather(-1, dynamic_idx)
    infer_meta = InferMeta(mode="prefill", capture_routing=True)

    with routing_replay_context(infer_meta):
        resolve_softmax_routes(0, logits, dynamic_idx, dynamic_weight, renormalize=True)

    assert torch.equal(captured_routing(infer_meta), dynamic_idx.unsqueeze(1))

    # The first token is forced to a different route.  -1 marks the final
    # causal input position, which was not forwarded during rollout.
    replay = torch.tensor([[[0, 2]], [[-1, -1]]])
    with routing_replay_context(TrainMeta(routing_replay=replay)):
        replay_idx, replay_weight = resolve_softmax_routes(0, logits, dynamic_idx, dynamic_weight, renormalize=True)

    assert replay_idx.tolist() == [[0, 2], [0, 1]]
    (replay_weight * torch.tensor([[1.0, 2.0], [2.0, 1.0]])).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert bool((logits.grad[0] != 0).any())


def test_sigmoid_routing_replay_recomputes_normalized_weights():
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    dynamic_idx = torch.tensor([[2, 1]])
    dynamic_weight = torch.tensor([[0.6, 0.4]])
    replay = torch.tensor([[[0, 2]]])

    with routing_replay_context(TrainMeta(routing_replay=replay)):
        replay_idx, replay_weight = resolve_sigmoid_routes(0, logits, dynamic_idx, dynamic_weight)

    expected = torch.sigmoid(logits).gather(-1, replay_idx)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    assert replay_idx.tolist() == [[0, 2]]
    assert torch.allclose(replay_weight, expected)


def test_train_pack_pads_last_token_with_dynamic_route_sentinel():
    routes = [
        [[0, 1], [2, 3]],
        [[1, 2], [3, 0]],
    ]
    seq = TrainSequence(
        tokens=[10, 11, 12],
        prompt_mask=[True, False, False],
        logprobs=[0.0, -0.1, -0.2],
        advantages=[0.0, 1.0, 1.0],
        routed_experts=routes,
    )

    pack = make_train_pack([seq])

    assert tuple(pack["routing_replay"].shape) == (1, 3, 2, 2)
    assert torch.equal(pack["routing_replay"][0, :2], torch.tensor(routes))
    assert torch.equal(pack["routing_replay"][0, 2], torch.full((2, 2), -1))


def test_rollout_state_aligns_prefill_decode_routes_and_trims_final_token():
    state = InferenceBatchState([[10, 11]], max_new_tokens=2, max_running_seqs=1)
    payload = state.build_prefill_payload()
    assert payload is not None
    prefill_routes = torch.tensor([[[0, 1]], [[1, 0]]])
    state.record_prefill_routing(payload, prefill_routes)
    state.generated = [[12, 13]]
    state.logprobs = [[-0.1, -0.2]]
    state.finish_reason = ["length"]
    # Decode forwards token 12 before sampling token 13.
    state.record_decode_routing(torch.tensor([0]), torch.tensor([[[2, 3]]]))

    output = state.to_rollout()

    assert output.routed_experts is not None
    assert output.routed_experts[0].tolist() == [[[0, 1]], [[1, 0]], [[2, 3]]]
