from types import SimpleNamespace

import torch

import areno.engine.inference as inference_mod
from areno.engine.data import SamplingParams
from areno.engine.data.rollout_state import InferenceBatchState
from areno.engine.inference import InferenceManager
from areno.engine.runtime.decode_metadata import DecodeMetadata
from tests.helpers import PatchedContext
from tests.test_inference_scheduler_cpu import _FakeInferenceManager


def test_tables_reuse_storage_and_grow_only_at_page_boundaries():
    state = InferenceBatchState(
        [[1] * 3, [2]], 6, max_running_seqs=2, max_cache_len=12, kv_block_size=4, num_cache_blocks=6
    )
    state.build_prefill_payload()
    metadata = DecodeMetadata(state, torch.device("cpu"))
    rows = torch.tensor([0, 1])
    metadata.sync_rows(state, rows)
    table, slots = metadata.prepare(state)
    ptrs = table.data_ptr(), slots.data_ptr()
    initial = table.clone()
    metadata.sync_rows(state, rows)
    assert metadata.prepare(state)[0] is table
    assert torch.equal(table, initial)
    metadata.finish_step(None)
    metadata.prepare(state)
    assert len(state._seq_to_blocks[0]) == 2
    assert len(state._seq_to_blocks[1]) == 1
    assert table[0, 1] == state._seq_to_blocks[0][1]
    assert (table.data_ptr(), slots.data_ptr()) == ptrs
    assert metadata.lengths == [4, 2]


def test_compaction_refill_preserves_lengths_and_reuses_recurrent_slot():
    state = InferenceBatchState(
        [[1] * 3, [2], [3] * 2], 6, max_running_seqs=2, max_cache_len=12, kv_block_size=4, num_cache_blocks=6
    )
    state.build_prefill_payload()
    resets = []
    manager = InferenceManager(
        SimpleNamespace(
            device=torch.device("cpu"),
            model=SimpleNamespace(reset_recurrent_cache_slots=lambda slots: resets.append(slots.tolist())),
        )
    )
    metadata = DecodeMetadata(state, torch.device("cpu"))
    metadata.sync_rows(state, torch.tensor([0, 1]))
    metadata.prepare(state)
    metadata.finish_step(torch.tensor([True, False]))
    manager._free_rollout_rows(state, torch.tensor([0]))
    metadata.sync_rows(state, torch.tensor([1]))
    metadata.prepare(state)
    payload = state.build_prefill_payload()
    assert payload["recurrent_slots"].tolist() == [0]
    metadata.sync_rows(state, torch.tensor([1, 2]))
    table, slots = metadata.prepare(state)
    assert metadata.lengths == [2, 2]
    assert slots.tolist() == [1, 0]
    assert table[:, 0].tolist() == [state._seq_to_blocks[1][0], state._seq_to_blocks[2][0]]
    assert resets == [[0]]
    for _ in range(3):
        metadata.finish_step(None)
    assert metadata.full_length_mask(state) is None
    metadata.finish_step(None)
    assert metadata.full_length_mask(state).tolist() == [True, False]


def test_decode_scheduler_does_not_rebuild_metadata_each_token():
    class Manager(_FakeInferenceManager):
        def _ensure_decode_kv_blocks(self, *args):
            raise AssertionError("ordinary decode read lengths from device")

        def _block_table_for_active_rows(self, *args):
            raise AssertionError("ordinary decode reconstructed block table")

        def _recurrent_slots_for_active_rows(self, *args):
            raise AssertionError("ordinary decode reconstructed recurrent slots")

    manager = Manager()
    state = InferenceBatchState(
        [[1] * 3, [2]], 7, max_running_seqs=2, max_cache_len=12, kv_block_size=4, num_cache_blocks=6
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)
    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(state, SamplingParams(), None)
    assert state.generated == [list(range(1, 8))] * 2
    assert state.finish_reason == ["length", "length"]


def test_eos_compaction_refill_and_cpu_cancellation_keep_request_identity():
    flags = torch.tensor([False, False, False])
    counts = {}

    class Manager(_FakeInferenceManager):
        def _infer_decode_next_token_tensor(self, tokens, positions, lengths, table, slots, *args, **kwargs):
            # The recurrent slot uniquely identifies each live request, and
            # admission reuses slot 0 after the first row finishes.
            outputs = []
            for slot, token in zip(slots.tolist(), tokens.tolist(), strict=True):
                key = (slot, token)
                counts[key] = counts.get(key, 0) + 1
                outputs.append(7 if slot == 0 else token + 1)
            flags[1] = True
            return torch.tensor(outputs), torch.zeros(len(outputs))

    manager = Manager()
    state = InferenceBatchState(
        [[1] * 3, [2], [3] * 2], 5, max_running_seqs=2, max_cache_len=12, kv_block_size=4, num_cache_blocks=6
    )
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)
    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(
            state, SamplingParams(), 7, cancel_flags=flags, cancel_indices=[0, 1, 2]
        )
    assert state.generated[0] == [1, 7]
    assert state.generated[2] == [1, 7]
    assert state.generated[1][-1] == 7
    assert state._seq_to_blocks == {}
    assert sorted(state._free_recurrent_slots) == [0, 1]


def test_shrinking_batch_changes_graph_buckets_without_stale_metadata():
    buckets = []
    pointers = []
    state = InferenceBatchState(
        [[row] * 3 for row in range(5)],
        8,
        max_running_seqs=5,
        max_cache_len=12,
        kv_block_size=4,
        num_cache_blocks=15,
    )

    class Manager(_FakeInferenceManager):
        def _infer_decode_next_token_tensor(self, tokens, positions, lengths, table, slots, count, *args, **kwargs):
            graph = self._decode_graph_for_active_count(count)
            buckets.append(graph.bucket)
            pointers.append((table.data_ptr(), slots.data_ptr()))
            for index, row in enumerate(slots.tolist()):
                assert lengths[index] == len(state.prompts[row]) + tokens[index] - 1
                blocks = state._seq_to_blocks[row]
                assert table[index].tolist() == blocks + [blocks[-1]] * (state.max_blocks_per_seq - len(blocks))
            return torch.where(tokens >= slots + 1, 7, tokens + 1), torch.zeros(count)

    manager = Manager()
    manager.config.runtime.eager_decode = False
    manager.worker._decode_graphs = {size: SimpleNamespace(bucket=size) for size in [1, 2, 4, 8]}
    ctx = SimpleNamespace(is_rank0=True, dp_rank=0, dp_size=1)
    with PatchedContext(inference_mod, get_tp_context=lambda: ctx, broadcast_object=lambda value, src=0: value):
        manager._generate_rollout_tokens_no_sync(state, SamplingParams(), 7)
    assert buckets == [8, 4, 4, 2, 1]
    assert len(set(pointers)) == 1
    assert state.generated == [list(range(1, row + 2)) + [7] for row in range(5)]
    assert state.finish_reason == ["stop"] * 5
