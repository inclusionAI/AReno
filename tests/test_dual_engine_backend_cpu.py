from __future__ import annotations

from types import SimpleNamespace

import pytest

from areno.api.backend.cuda.backend import CudaBackend
from areno.engine.policy_sync import PolicyTensorMeta
from areno.engine.protocol import Op


class _Handle:
    def __init__(self, result=None, error: Exception | None = None):
        self._result = result
        self._error = error

    def result(self):
        if self._error is not None:
            raise self._error
        return self._result


class _Cluster:
    def __init__(self, plan, *, sync_error: Exception | None = None):
        self.plan = plan
        self.sync_error = sync_error
        self.calls = []

    def call(self, op, payload=None):
        self.calls.append((op, payload))
        assert op is Op.POLICY_SYNC_PLAN
        return [self.plan]

    def submit(self, op, payload=None):
        self.calls.append((op, payload))
        if self.sync_error is not None:
            return _Handle(error=self.sync_error)
        return _Handle(
            [
                {
                    "version": payload.version,
                    "bytes": 16,
                    "tensors": 1,
                    "elapsed_s": 0.01,
                }
            ]
        )


class _Engine:
    def __init__(self, name, plan, *, sync_error: Exception | None = None):
        self.name = name
        self.cluster = _Cluster(plan, sync_error=sync_error)
        self.config = SimpleNamespace(dp_size=1, model=SimpleNamespace(max_position_embeddings=128))
        self.events = []

    def begin_rollout_session(self):
        self.events.append("begin")

    async def begin_rollout_session_async(self):
        self.events.append("begin_async")

    def end_rollout_session(self):
        self.events.append("end")

    async def end_rollout_session_async(self):
        self.events.append("end_async")

    def save_checkpoint(self, path):
        self.events.append(("save", path))
        return path

    def close(self):
        self.events.append("close")


def _backend(*, rollout_error: Exception | None = None) -> tuple[CudaBackend, _Engine, _Engine]:
    plan = (PolicyTensorMeta("weight", (4,), "float32", 16),)
    train = _Engine("train", plan)
    rollout = _Engine("rollout", plan, sync_error=rollout_error)
    backend = CudaBackend()
    backend._train_engine = train
    backend._rollout_engine = rollout
    backend._separate_rollout = True
    backend._policy_sync_bucket_bytes = 1024
    return backend, train, rollout


def test_dual_backend_routes_rollout_lifecycle_and_checkpoint() -> None:
    backend, train, rollout = _backend()

    backend.begin_rollout_session(None)
    backend.end_rollout_session(None)
    assert backend.save_checkpoint(None, "checkpoint") == "checkpoint"

    assert rollout.events == ["begin", "end"]
    assert train.events == [("save", "checkpoint")]


def test_policy_sync_runs_once_for_each_new_train_version() -> None:
    backend, train, rollout = _backend()
    backend._train_policy_version = 2

    backend._sync_policy_if_needed()
    backend._sync_policy_if_needed()

    assert backend._rollout_policy_version == 2
    assert [op for op, _ in train.cluster.calls] == [Op.POLICY_SYNC_PLAN, Op.POLICY_SYNC_PUBLISH]
    assert [op for op, _ in rollout.cluster.calls] == [Op.POLICY_SYNC_PLAN, Op.POLICY_SYNC_RECEIVE]
    assert backend._pending_policy_sync_metrics["policy_sync_transfer_time_s"] == pytest.approx(0.01)
    assert backend._pending_policy_sync_metrics["policy_sync_bytes"] == 16.0
    assert backend._pending_policy_sync_metrics["policy_sync_tensors"] == 1.0
    assert backend._pending_policy_sync_metrics["policy_sync_throughput_gbps"] == pytest.approx(0.0000128)
    assert backend._pending_policy_sync_metrics["policy_sync_time_s"] > 0.0


def test_policy_sync_failure_does_not_advance_rollout_version() -> None:
    backend, _, _ = _backend(rollout_error=RuntimeError("receive failed"))
    backend._train_policy_version = 1

    with pytest.raises(RuntimeError, match="receive failed"):
        backend._sync_policy_if_needed()

    assert backend._rollout_policy_version == 0


def test_policy_plan_mismatch_fails_before_collectives() -> None:
    backend, train, rollout = _backend()
    rollout.cluster.plan = (PolicyTensorMeta("other", (4,), "float32", 16),)
    backend._train_policy_version = 1

    with pytest.raises(RuntimeError, match="layouts do not match"):
        backend._sync_policy_if_needed()

    assert [op for op, _ in train.cluster.calls] == [Op.POLICY_SYNC_PLAN]
    assert [op for op, _ in rollout.cluster.calls] == [Op.POLICY_SYNC_PLAN]


def test_close_releases_both_engines() -> None:
    backend, train, rollout = _backend()

    backend.close()

    assert rollout.events == ["close"]
    assert train.events == ["close"]
