"""Whole-trajectory loss grouping with exact, independent turn contexts."""

from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from areno.api.backend.common import train_microbatches, train_sequence_ids
from areno.api.backend.cuda.losses import gspo_loss_fn
from areno.api.backend.cuda.training import make_train_pack
from areno.api.models import TrainSequence
from areno.api.rewards import RewardRecord
from areno.api.trainers.policy_only import PolicyOnlyTrainer
from areno.engine.runtime import train_step
from areno.engine.runtime.common import split_data_pack_by_dp


def _agent_batch():
    return SimpleNamespace(
        # The second turn rewrites the prefix. It must be scored under that
        # actual context while sharing the first turn's sequence loss.
        token_rows=[[1, 2, 10], [9, 8, 11, 12], [7, 13]],
        response_masks=[[False, False, True], [False, False, True, True], [False, True]],
        loss_masks=[[False, False, True], [False, False, True, True], [False, True]],
        rollout_logprobs=[[0, 0, -0.1], [0, 0, -0.2, -0.3], [0, -0.4]],
        features=[None] * 3,
        routed_experts=[torch.full((length - 1, 1, 1), i, dtype=torch.int16) for i, length in enumerate([3, 4, 2])],
        rewards=[0.0, 1.0],
        reward_records=[RewardRecord(prompt="p", completion="", metadata={"prompt_index": 0}) for _ in range(2)],
        row_reward_indices=[0, 0, 1],
    )


def _training_rows():
    policy = object.__new__(PolicyOnlyTrainer)
    return policy._materialize_agentic_train_batch(SimpleNamespace(eos_token_id=0), None, _agent_batch())[0]


def test_trajectory_ids_follow_reward_records_and_preserve_turn_contexts():
    rows = _training_rows()

    assert [row.trajectory_id for row in rows] == [0, 0, 1]
    assert [row.scalar_advantage for row in rows] == [-1.0, -1.0, 1.0]
    assert [row.tokens for row in rows] == _agent_batch().token_rows
    assert [row.logprobs for row in rows] == _agent_batch().rollout_logprobs
    assert all(
        torch.equal(row.routed_experts, route) for row, route in zip(rows, _agent_batch().routed_experts, strict=True)
    )


def test_sequence_ids_do_not_merge_unassigned_rows_or_distinct_trajectories():
    rows = [TrainSequence(trajectory_id=value) for value in [7, 7, None, 12, None, 12]]
    assert train_sequence_ids(rows) == [0, 0, 1, 2, 3, 2]


def test_microbatches_keep_interleaved_trajectories_whole():
    rows = [TrainSequence(tokens=[index], trajectory_id=value) for index, value in enumerate([7, 12, 7, None, 12])]

    assert [[row.tokens[0] for row in batch] for batch in train_microbatches(rows, 1)] == [[0, 2], [1, 4], [3]]
    assert [[row.tokens[0] for row in batch] for batch in train_microbatches(rows, 2)] == [[0, 2, 1, 4], [3]]
    assert train_microbatches([], 1) == []
    with pytest.raises(ValueError, match="mini_bs must be positive"):
        train_microbatches(rows, 0)


def test_cuda_backend_passes_complete_trajectories_to_each_microbatch():
    from areno.api.backend.cuda.backend import CudaBackend

    received = []

    def step(packs, *, gradient_accumulation_steps):
        received.extend(packs)
        assert gradient_accumulation_steps is None
        return [SimpleNamespace(loss=0.0, metrics={}, stepped=True, adapter_version=None) for _ in packs]

    backend = CudaBackend()
    backend._train_engine = SimpleNamespace(step=step)
    backend.train(SimpleNamespace(), _training_rows(), gspo_loss_fn, mini_bs=1)

    assert [pack["loss_sequence_ids"].tolist() for pack in received] == [[0, 0], [0]]
    assert [pack["lengths"].tolist() for pack in received] == [[3, 4], [2]]


@pytest.mark.parametrize("tp_size", [1, 2])
def test_packed_attention_contexts_are_independent_but_gspo_uses_trajectory_ids(monkeypatch, tp_size):
    monkeypatch.setattr(train_step, "get_tp_context", lambda: SimpleNamespace(world_size=tp_size))
    rows = _training_rows()
    pack = make_train_pack(rows)
    packed = train_step._pack_train_data(pack)

    expected_cu = [0, 3, 7, 9] + ([10] if tp_size == 2 else [])
    assert packed["train_cu_seqlens"].tolist() == expected_cu
    assert packed["packed_num_sequences"] == 2
    assert packed["packed_seq_ids"].tolist() == [0, 0, 0, 0, 0, 1]
    assert packed["packed_response_mask"].tolist() == [False, True, False, True, True, True]
    assert packed["packed_logprobs"].tolist() == pytest.approx([0, -0.1, 0, -0.2, -0.3, -0.4])
    assert packed["position_ids"].flatten().tolist()[:9] == [0, 1, 2, 0, 1, 2, 3, 0, 1]
    routes = packed["packed_routing_replay"].flatten().tolist()
    assert routes[:9] == [0, 0, -1, 1, 1, 1, -1, 2, -1]

    logprobs = torch.full((6,), -1.0, requires_grad=True)
    loss, stats = gspo_loss_fn(packed, logprobs)
    loss.backward()

    # Three actions over two turns and one action in one turn each receive
    # half the trajectory-mean gradient mass, with zero context gradients.
    assert loss.item() == pytest.approx(0.0)
    assert logprobs.grad.tolist() == pytest.approx([0, 1 / 6, 0, 1 / 6, 1 / 6, -1 / 2])
    assert stats["response_len"].item() == pytest.approx(2.0)


def test_dp_shards_keep_every_turn_and_sidecar_of_a_trajectory_together():
    rows = _training_rows()
    pack = make_train_pack(rows)
    pack["features"] = [{"turn": index} for index in range(3)]
    shards = split_data_pack_by_dp(pack, 2)

    assert [shard["loss_sequence_ids"].tolist() for shard in shards] == [[0, 0], [1]]
    assert [shard["features"] for shard in shards] == [[{"turn": 0}, {"turn": 1}], [{"turn": 2}]]
    assert shards[0]["lengths"].tolist() == [3, 4]
    assert shards[0]["routing_replay"][:, 0, 0, 0].tolist() == [0, 1]
    assert shards[1]["routing_replay"][0, 0, 0, 0].item() == 2
    # One complete trajectory is replicated if it cannot fill all DP ranks.
    one = make_train_pack(rows[:2])
    assert all(shard is one for shard in split_data_pack_by_dp(one, 2))


def test_ordinary_training_rows_keep_independent_sequence_ids():
    rows = _training_rows()
    for row in rows:
        row.trajectory_id = None
    pack = make_train_pack(rows)
    assert "loss_sequence_ids" not in pack
    assert train_step._pack_train_data(pack)["packed_num_sequences"] == 3
    assert [len(batch) for batch in train_microbatches(rows, 2)] == [2, 1]


def test_mlx_loss_view_joins_only_trajectory_loss_signals(monkeypatch):
    import sys

    from areno.api.backend.mlx.training import make_train_batch, trajectory_loss_inputs

    # These array operations have identical NumPy semantics, so the packing
    # contract can also be exercised on CPU-only Linux without Metal.
    mlx = ModuleType("mlx")
    mlx.core = np
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", np)
    rows = _training_rows()
    batch = make_train_batch(rows)
    selected = np.arange(9, dtype=np.float32).reshape(3, 3)
    loss_batch, loss_logprobs = trajectory_loss_inputs(batch, selected)

    assert batch["input_ids"].shape == (3, 4)
    assert loss_batch["response_mask"].shape == loss_logprobs.shape == (2, 6)
    assert loss_logprobs[loss_batch["response_mask"].astype(bool)].tolist() == [1, 4, 5, 6]
    assert loss_batch["old_logprobs"][loss_batch["response_mask"].astype(bool)].tolist() == pytest.approx(
        [-0.1, -0.2, -0.3, -0.4]
    )
    assert loss_batch["advantages"][loss_batch["response_mask"].astype(bool)].tolist() == [-1, -1, -1, 1]
    assert not np.any(loss_batch["response_mask"][1, 3:])
    assert trajectory_loss_inputs({}, selected)[1] is selected
