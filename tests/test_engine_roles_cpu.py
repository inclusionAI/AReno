from types import SimpleNamespace
from unittest.mock import patch

import torch

from areno.api.backend.cuda.roles import (
    RoleManager,
    WorkerRole,
    _gather_packed_hidden,
    accumulate_role_main_gradients,
)
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


def test_actor_logprob_scoring_replays_rollout_routes():
    seen = {}

    class TextModel:
        def __call__(self, *, input_ids, position_ids, train_meta):
            del position_ids
            seen["routes"] = train_meta.routing_replay
            return SimpleNamespace(logits_shard=torch.zeros((*input_ids.shape, 8)))

    manager = RoleManager(SimpleNamespace(device=torch.device("cpu")))
    payload = ScorePayload(
        role="actor",
        token_rows_by_dp=[[[1, 2, 3]]],
        features_by_dp=None,
        pad_token_id=0,
    )
    routes = [torch.tensor([[[2, 3]], [[4, 5]]], dtype=torch.int16)]

    with patch("areno.api.backend.cuda.roles.packed_next_token_logprobs", return_value=torch.zeros(2)):
        rows = manager._score_logprob_rows(
            TextModel(), [[1, 2, 3]], payload, routed_experts=routes, sequence_parallel=False
        )

    assert rows == [[0.0, 0.0, 0.0]]
    assert torch.equal(seen["routes"][:2], routes[0])
    assert torch.equal(seen["routes"][2], torch.full((1, 2), -1, dtype=torch.int16))


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


def test_worker_role_preserves_disk_optimizer_residency_across_swap():
    calls = []

    class Model:
        def to(self, device):
            calls.append(("model_to", device))

        def onload_train_weights(self, device):
            calls.append(("weights_onload", device))

        def clear_infer_weights(self):
            calls.append(("clear_infer",))

        def clear_kv_caches(self):
            calls.append(("clear_kv",))

        def offload_train_weights(self):
            calls.append(("weights_offload",))

    class Optimizer:
        def configure_state_offload(self, *, mode, directory, batch_size):
            calls.append(("configure", mode, directory, batch_size))

        def prefetch_state(self):
            calls.append(("prefetch",))

        def onload_state(self, device):
            calls.append(("optimizer_onload", device))

        def offload_state(self, *, mode, directory, batch_size):
            calls.append(("optimizer_offload", mode, directory, batch_size))

    device = torch.device("cpu")
    role = WorkerRole(
        "model",
        Model(),
        Optimizer(),
        value_head=None,
        optimizer_offload_mode="disk",
        optimizer_offload_dir="/tmp/optimizer",
        optimizer_offload_batch_size=2,
    )

    role.onload(device)
    role.offload()

    assert ("optimizer_onload", device) not in calls
    assert ("configure", "disk", "/tmp/optimizer", 2) in calls
    assert ("prefetch",) in calls
    assert ("optimizer_offload", "disk", "/tmp/optimizer", 2) in calls


def test_actor_preserves_disk_optimizer_residency_across_role_swap():
    calls = []

    class Model:
        def to(self, device):
            calls.append(("model_to", device))

        def onload_train_weights(self, device):
            calls.append(("weights_onload", device))

        def clear_infer_weights(self):
            calls.append(("clear_infer",))

        def clear_kv_caches(self):
            calls.append(("clear_kv",))

        def offload_train_weights(self):
            calls.append(("weights_offload",))

    class Optimizer:
        def configure_state_offload(self, *, mode, directory, batch_size):
            calls.append(("configure", mode, directory, batch_size))

        def onload_state(self, device):
            calls.append(("optimizer_onload", device))

        def offload_state(self, *, mode, directory, batch_size):
            calls.append(("optimizer_offload", mode, directory, batch_size))

    device = torch.device("cpu")
    runtime = SimpleNamespace(
        optimizer_state_offload="disk",
        optimizer_state_offload_dir="/tmp/optimizer",
        optimizer_state_offload_batch_size=2,
    )
    worker = SimpleNamespace(
        model=Model(),
        optimizer=Optimizer(),
        device=device,
        config=SimpleNamespace(runtime=runtime),
        _actor_on_device=False,
        _train_state_ready=True,
    )
    worker._optimizer_offload_options = lambda: ArenoWorker._optimizer_offload_options(worker)
    worker._release_decode_graphs = lambda: calls.append(("release_decode_graphs",))

    ArenoWorker._prepare_actor_onloaded(worker)
    ArenoWorker._prepare_actor_offloaded(worker)

    assert ("optimizer_onload", device) not in calls
    assert ("configure", "disk", "/tmp/optimizer", 2) in calls
    assert ("optimizer_offload", "disk", "/tmp/optimizer", 2) in calls


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


def test_resident_role_gradient_accumulation_stays_fp32():
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
    role = WorkerRole("model", model, optimizer=None, value_head=None)
    parameter = model.weight

    parameter.grad = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)
    accumulate_role_main_gradients(role)
    parameter.grad = torch.tensor([[0.75, 0.25]], dtype=torch.bfloat16)
    accumulate_role_main_gradients(role)

    assert parameter.grad is None
    assert parameter.main_grad.dtype == torch.float32
    torch.testing.assert_close(parameter.main_grad, torch.tensor([[1.0, -0.25]]))


def test_role_gradient_sync_keeps_full_gradient_path():
    calls = []
    optimizer = SimpleNamespace(reduce_scatter_gradients=lambda: calls.append("reduce_scatter"))
    role = WorkerRole("model", torch.nn.Linear(2, 1), optimizer=optimizer, value_head=None)
    ctx = SimpleNamespace(world_size=1, dp_size=1)

    with patch("areno.engine.worker.get_tp_context", return_value=ctx):
        ArenoWorker._sync_role_grads(SimpleNamespace(), role)

    assert calls == []


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
