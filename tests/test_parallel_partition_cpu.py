from __future__ import annotations

import multiprocessing as mp

import pytest
import torch

from areno.engine.parallel import context
from areno.engine.parallel.collectives import broadcast_object
from areno.engine.protocol import _create_rendezvous_store


def _run_offset_tp_broadcast(global_rank: int, port: int, output_queue) -> None:
    role = "train" if global_rank < 2 else "rollout"
    local_rank = global_rank if role == "train" else global_rank - 2
    context.init_process_group(
        rank=local_rank,
        world_size=2,
        master_addr="127.0.0.1",
        master_port=port,
        device_id=local_rank,
        tp_size=2,
        global_rank=global_rank,
        global_world_size=4,
        train_world_size=2,
        train_tp_size=2,
        rollout_world_size=2,
        rollout_tp_size=2,
        role=role,
    )
    try:
        value = f"{role}-root" if local_rank == 0 else None
        output_queue.put((global_rank, broadcast_object(value, src=0)))
    finally:
        context.destroy_process_group()


def _mock_distributed(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # These tests exercise group construction only, not the rendezvous, so
    # the client TCPStore and the process-group init are both faked.
    monkeypatch.setattr(context.dist, "TCPStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(context.dist, "init_process_group", lambda **kwargs: calls.append(("init", kwargs)))

    def new_group(*, ranks):
        group = tuple(ranks)
        calls.append(("group", group))
        return group

    monkeypatch.setattr(context.dist, "new_group", new_group)
    return calls


def _init_partition(*, local_rank: int, global_rank: int, role: str):
    return context.init_process_group(
        rank=local_rank,
        world_size=4 if role == "train" else 2,
        master_addr="127.0.0.1",
        master_port=12345,
        device_id=local_rank,
        tp_size=2 if role == "train" else 1,
        global_rank=global_rank,
        global_world_size=6,
        train_world_size=4,
        train_tp_size=2,
        rollout_world_size=2,
        rollout_tp_size=1,
        role=role,
    )


def test_partitioned_group_construction_order_is_identical(monkeypatch) -> None:
    train_calls = _mock_distributed(monkeypatch)
    train_ctx = _init_partition(local_rank=1, global_rank=1, role="train")
    train_groups = [value for kind, value in train_calls if kind == "group"]

    rollout_calls = _mock_distributed(monkeypatch)
    rollout_ctx = _init_partition(local_rank=1, global_rank=5, role="rollout")
    rollout_groups = [value for kind, value in rollout_calls if kind == "group"]

    assert (
        train_groups
        == rollout_groups
        == [
            (0, 1),
            (2, 3),
            (0, 2),
            (1, 3),
            (4,),
            (5,),
            (4, 5),
            (0, 5),
            (2, 4),
        ]
    )
    assert train_ctx.rank == 1
    assert train_ctx.dp_rank == 0
    assert train_ctx.group == (0, 1)
    assert train_ctx.dp_group == (1, 3)
    assert train_ctx.policy_publisher_groups == (None, None)
    assert rollout_ctx.rank == 0
    assert rollout_ctx.dp_rank == 1
    assert rollout_ctx.group == (5,)
    assert rollout_ctx.dp_group == (4, 5)
    assert rollout_ctx.policy_publisher_groups == ((0, 5), None)
    assert rollout_ctx.policy_source_ranks == (0, 2)
    assert rollout_ctx.policy_bridge_ranks == (5, 4)
    assert rollout_ctx.policy_bridge_dp_ranks == (1, 0)
    assert train_ctx.tp_global_rank(0) == 0
    assert train_ctx.tp_global_rank(1) == 1
    assert rollout_ctx.tp_global_rank(0) == 5


def test_single_engine_context_creates_no_policy_publisher_group(monkeypatch) -> None:
    calls = _mock_distributed(monkeypatch)
    ctx = context.init_process_group(
        rank=0,
        world_size=2,
        master_addr="127.0.0.1",
        master_port=12345,
        device_id=0,
        tp_size=2,
    )

    assert ctx.role == "train"
    assert ctx.policy_publisher_groups == ()
    assert [value for kind, value in calls if kind == "group"] == [(0, 1), (0,), (1,)]


def test_real_gloo_tp_broadcast_uses_partition_global_root() -> None:
    spawn = mp.get_context("spawn")
    output_queue = spawn.Queue()
    # The coordinator holds the server store (port=0) so the resolved port is
    # genuinely reserved before the workers join as client stores.
    store = _create_rendezvous_store("127.0.0.1", 4)
    port = int(store.port)
    processes = [
        spawn.Process(target=_run_offset_tp_broadcast, args=(global_rank, port, output_queue))
        for global_rank in range(4)
    ]
    for process in processes:
        process.start()
    results = dict(output_queue.get(timeout=30) for _ in processes)
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    assert results == {
        0: "train-root",
        1: "train-root",
        2: "rollout-root",
        3: "rollout-root",
    }


def test_rendezvous_store_resolves_and_holds_its_port() -> None:
    """The coordinator store resolves a real port and keeps it bound.

    The resolved port must be immediately usable by client stores, and binding
    the same port again must fail while the server store is alive (the socket
    is genuinely reserved, unlike the old bind-close-bind probe, #517).
    """

    import socket

    import torch.distributed as dist

    store = _create_rendezvous_store("127.0.0.1", 2)
    port = int(store.port)
    assert port > 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        with pytest.raises(OSError):
            sock.bind(("127.0.0.1", port))
    # A client store connects to the retained server.
    client = dist.TCPStore("127.0.0.1", port, world_size=2, is_master=False)
    store.set("k", "v")
    assert client.get("k") == b"v"


def test_start_partitioned_clusters_resolves_frozen_world_spec() -> None:
    """`start_partitioned_clusters` must fill in the resolved rendezvous port
    without mutating the frozen `world_spec`.

    `DistributedWorldSpec` is ``dataclass(frozen=True)``, so assigning
    ``world_spec.master_port`` raises ``FrozenInstanceError``. The
    coordinator-held store's resolved port is instead carried into a rebuilt
    spec before any worker spawns (see #517).
    """

    from areno.engine.protocol import (
        ClusterPartition,
        DistributedWorldSpec,
        start_partitioned_clusters,
    )

    class FakeCluster:
        """Stand-in for TPCluster exposing exactly what the helper touches."""

        def __init__(self, partition):
            self.world_spec = None
            self.partition = partition
            self._rendezvous_store = None
            self.started = False

        def _spawn_workers(self) -> None:
            pass

        def _wait_for_worker_ready(self, ranks) -> None:
            pass

        def _abort_start(self) -> None:
            pass

        def _start_result_pump(self) -> None:
            pass

    # Mirror backend.py: a placeholder port that the helper must resolve.
    world_spec = DistributedWorldSpec(
        master_addr="127.0.0.1",
        master_port=0,
        global_world_size=4,
        train=ClusterPartition(
            role="train",
            global_rank_offset=0,
            local_world_size=2,
            tp_size=2,
            devices=(0, 1),
        ),
        rollout=ClusterPartition(
            role="rollout",
            global_rank_offset=2,
            local_world_size=2,
            tp_size=2,
            devices=(2, 3),
        ),
    )
    train = FakeCluster(world_spec.train)
    rollout = FakeCluster(world_spec.rollout)
    train.world_spec = world_spec
    rollout.world_spec = world_spec

    start_partitioned_clusters(train, rollout, world_spec)

    # The shared world_spec's port is resolved and both partitions agree.
    assert train.started and rollout.started
    assert train.world_spec is rollout.world_spec
    assert train.world_spec.master_port > 0
    assert train.world_spec.master_port == rollout.world_spec.master_port
    # The coordinator store is retained on every cluster.
    assert train._rendezvous_store is rollout._rendezvous_store
    # The frozen input spec must not have been mutated in place.
    assert world_spec.master_port == 0
