from types import SimpleNamespace
from unittest.mock import patch

import torch

from areno.api.backend.cuda.roles import RoleManager, WorkerRole, _gather_packed_hidden
from areno.engine.protocol import ScorePayload
from areno.engine.worker import ArenoWorker


def test_score_logprobs_omits_empty_feature_rows_for_text_model():
    class TextModel:
        def __call__(self, *, input_ids, position_ids, train_meta):
            del position_ids, train_meta
            return SimpleNamespace(logits_shard=torch.zeros((*input_ids.shape, 8)))

    manager = RoleManager(SimpleNamespace(device=torch.device("cpu")))
    payload = ScorePayload(
        role="ref",
        token_rows_by_dp=[[[1, 2, 3]]],
        features_by_dp=[[]],
        pad_token_id=0,
    )

    with patch("areno.api.backend.cuda.roles.packed_next_token_logprobs", return_value=torch.zeros(2)):
        rows = manager._score_logprob_rows(TextModel(), [[1, 2, 3]], payload, features=[None], sequence_parallel=False)

    assert rows == [[0.0, 0.0, 0.0]]


def test_worker_role_onload_for_inference_prepares_derived_weights():
    calls = []

    class Model:
        def to(self, device):
            calls.append(("to", device))

        def onload_train_weights(self, device):
            calls.append(("onload_train_weights", device))

        def prepare_infer_weights(self):
            calls.append(("prepare_infer_weights",))

        def offload_train_weights(self):
            calls.append(("offload_train_weights",))

    device = torch.device("cpu")
    WorkerRole("model", Model(), optimizer=None, value_head=None, sequence_parallel=False).onload_for_inference(device)

    assert calls == [
        ("to", device),
        ("onload_train_weights", device),
        ("prepare_infer_weights",),
        ("offload_train_weights",),
    ]


def test_actor_logprob_scoring_prepares_inference_weights():
    class Model:
        def eval(self):
            pass

    class Worker:
        device = torch.device("cpu")
        model = Model()
        config = SimpleNamespace(effective_sequence_parallel=False)

        def __init__(self):
            self.prepared = False

        def _prepare_actor_for_inference(self):
            self.prepared = True

    worker = Worker()
    manager = RoleManager(worker)
    payload = ScorePayload(
        role="actor",
        token_rows_by_dp=[[[1, 2, 3]]],
        features_by_dp=None,
        pad_token_id=0,
    )
    ctx = SimpleNamespace(dp_rank=0, rank=0)

    with (
        patch("areno.api.backend.cuda.roles.get_tp_context", return_value=ctx),
        patch.object(manager, "_score_logprob_rows", return_value=[[0.0, 0.0, 0.0]]),
    ):
        rows = manager.score_logprobs(payload)

    assert worker.prepared
    assert rows == [[0.0, 0.0, 0.0]]


def test_gather_packed_hidden_averages_replicated_head_backbone_gradient():
    hidden = torch.ones(1, 2, 3, requires_grad=True)
    train_meta = SimpleNamespace(sequence_parallel=True)

    with (
        patch("areno.api.backend.cuda.roles.get_tp_context", return_value=SimpleNamespace(world_size=4)),
        patch("areno.api.backend.cuda.roles.gather_from_sequence_parallel_region", side_effect=lambda x: x),
    ):
        gathered = _gather_packed_hidden(hidden, train_meta)
        gathered.sum().backward()

    assert torch.equal(hidden.grad, torch.full_like(hidden, 0.25))


def test_prepare_actor_for_inference_rebuilds_weights_and_invalidates_train_state():
    calls = []

    class Model:
        def onload_train_weights(self, device):
            calls.append(("onload_train_weights", device))

        def prepare_infer_weights(self):
            calls.append(("prepare_infer_weights",))

        def offload_train_weights(self):
            calls.append(("offload_train_weights",))

    device = torch.device("cpu")
    worker = SimpleNamespace(
        device=device,
        model=Model(),
        _train_state_ready=True,
        _prepare_actor_onloaded=lambda: calls.append(("prepare_actor_onloaded",)),
    )

    ArenoWorker._prepare_actor_for_inference(worker)

    assert calls == [
        ("prepare_actor_onloaded",),
        ("onload_train_weights", device),
        ("prepare_infer_weights",),
        ("offload_train_weights",),
    ]
    assert not worker._train_state_ready
