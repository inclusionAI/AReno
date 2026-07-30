from __future__ import annotations

import torch

from areno.engine.parallel import context


def _mock_distributed(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
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
            (0, 1, 4, 5),
            (2, 3, 4, 5),
        ]
    )
    assert train_ctx.rank == 1
    assert train_ctx.dp_rank == 0
    assert train_ctx.group == (0, 1)
    assert train_ctx.dp_group == (1, 3)
    assert train_ctx.policy_publisher_groups == ((0, 1, 4, 5), None)
    assert rollout_ctx.rank == 0
    assert rollout_ctx.dp_rank == 1
    assert rollout_ctx.group == (5,)
    assert rollout_ctx.dp_group == (4, 5)
    assert rollout_ctx.policy_publisher_groups == ((0, 1, 4, 5), (2, 3, 4, 5))


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
