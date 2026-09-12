from __future__ import annotations

import multiprocessing as mp
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import areno.engine.runtime.common as runtime_common
from areno.engine.api import _merge_dp_rank0_strided_results
from areno.engine.data import RolloutOutput
from areno.engine.protocol import _create_rendezvous_store
from areno.engine.runtime.common import (
    _check_token_ids,
    _device_long,
    dp_rank0_results,
    merge_metric_dicts,
    merge_train_stats,
    split_data_pack_by_dp,
    split_list_by_dp,
)
from areno.engine.runtime.decode_graph import (
    DecodeGraph,
    agree_across_ranks,
    bucket_for,
    ceil_div,
    graph_capture_headroom,
)
from areno.engine.runtime.rollout import _empty_rollout, _merge_dp_rollouts_in_input_order, _merge_rollouts
from areno.engine.runtime.train_step import _grad_norms


def _rollout(prompt_ids, response_ids, logprobs, finish_reason=None, metrics=None):
    """Build a minimal RolloutOutput for merge-helper tests."""

    rows = len(prompt_ids)
    max_len = max((len(p) + len(r) for p, r in zip(prompt_ids, response_ids, strict=True)), default=0)
    max_resp = max((len(r) for r in response_ids), default=0)
    return RolloutOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=torch.zeros(rows, max_len, dtype=torch.long),
        attention_mask=torch.zeros(rows, max_len, dtype=torch.long),
        response_mask=torch.zeros(rows, max_len, dtype=torch.long),
        logprobs=torch.tensor(logprobs, dtype=torch.float32).reshape(rows, max_resp)
        if rows and max_resp
        else torch.empty(rows, 0),
        finish_reason=finish_reason or ["length"] * rows,
        metrics=metrics,
    )


class RuntimeCommonTest(unittest.TestCase):
    """Runtime utility tests cover DP slicing and scalar merge behavior."""

    def test_split_list_by_dp_uses_round_robin_order(self):
        """Round-robin split must match the DP rollout ordering contract."""
        self.assertEqual(split_list_by_dp([0, 1, 2, 3, 4], 2), [[0, 2, 4], [1, 3]])

    def test_split_data_pack_by_dp_slices_batch_major_values(self):
        """Batch-leading tensors and lists should be strided by DP rank."""
        pack = {
            "input_ids": torch.arange(12).view(4, 3),
            "labels": torch.arange(4),
            "meta": {"rows": ["a", "b", "c", "d"], "constant": torch.tensor(9)},
        }

        shards = split_data_pack_by_dp(pack, 2)

        self.assertEqual(shards[0]["input_ids"].tolist(), [[0, 1, 2], [6, 7, 8]])
        self.assertEqual(shards[1]["labels"].tolist(), [1, 3])
        self.assertEqual(shards[0]["meta"]["rows"], ["a", "c"])
        self.assertEqual(int(shards[1]["meta"]["constant"]), 9)

    def test_split_data_pack_replicates_tiny_batches(self):
        """Tiny batches are replicated because they cannot fill every DP rank."""
        pack = {"input_ids": torch.tensor([[1, 2]])}

        shards = split_data_pack_by_dp(pack, 2)

        self.assertIs(shards[0], pack)
        self.assertIs(shards[1], pack)

    def test_dp_rank0_results_drops_tensor_parallel_duplicates(self):
        """Coordinator should keep only TP rank 0 from each DP group."""
        self.assertEqual(
            dp_rank0_results(["dp0tp0", "dp0tp1", "dp1tp0", "dp1tp1"], tp_size=2, dp_size=2), ["dp0tp0", "dp1tp0"]
        )

    def test_score_result_merge_restores_dp_strided_order(self):
        """Score ops should merge local DP shards without worker-side gather."""
        results = [[0, 2, 4], None, [1, 3], None]

        merged = _merge_dp_rank0_strided_results(results, tp_size=2, dp_size=2)

        self.assertEqual(merged, [0, 1, 2, 3, 4])

    def test_merge_train_stats_averages_loss_and_metrics(self):
        """Train stats from DP ranks should average numeric metrics."""
        stats = merge_train_stats(
            [
                {"loss": 1.0, "stepped": True, "metrics": {"a": 2.0}},
                {"loss": 3.0, "stepped": False, "metrics": {"a": 4.0, "b": 6.0}},
            ]
        )

        self.assertEqual(stats.loss, 2.0)
        self.assertFalse(stats.stepped)
        self.assertEqual(stats.metrics, {"a": 3.0, "b": 6.0})
        self.assertIsNone(merge_metric_dicts([None, {}]))

    def test_device_long_and_token_id_guard(self):
        """Token id validation should accept valid ids and describe invalid ones."""
        tensor = torch.tensor([1, 2], dtype=torch.int32)

        converted = _device_long(tensor, torch.device("cpu"))

        self.assertEqual(converted.dtype, torch.long)
        with patch.object(runtime_common, "_CHECK_TOKEN_IDS", True):
            _check_token_ids(converted, vocab_size=3, name="sample")
            with self.assertRaisesRegex(RuntimeError, "sample out of vocab range"):
                _check_token_ids(torch.tensor([0, 3]), vocab_size=3, name="sample")

    def test_grad_norms_fuses_global_and_parameter_groups(self):
        """One gradient pass should produce exact global and group norms."""

        text = torch.nn.Parameter(torch.zeros(2))
        tower = torch.nn.Parameter(torch.zeros(2))
        projector = torch.nn.Parameter(torch.zeros(1))
        text.main_grad = torch.tensor([3.0, 4.0])
        tower.main_grad = torch.tensor([5.0, 12.0])
        projector.main_grad = torch.tensor([84.0])
        tower._areno_lr_group = "tower"
        projector._areno_lr_group = "projector"

        with patch(
            "areno.engine.runtime.train_step.get_tp_context",
            return_value=SimpleNamespace(world_size=1, rank=0, group=None),
        ):
            norms = _grad_norms([text, tower, projector], ("tower", "projector"))

        self.assertAlmostEqual(norms["global"], 85.14693450927734)
        self.assertEqual(norms["tower"], 13.0)
        self.assertEqual(norms["projector"], 84.0)


class DecodeGraphUtilityTest(unittest.TestCase):
    """Decode graph pure helpers can be tested without CUDA graph capture."""

    def test_bucket_for_uses_smallest_covering_bucket(self):
        """Bucket selection should not overgrow unless no bucket fits."""
        self.assertEqual(bucket_for(5, [1, 4, 8]), 8)
        self.assertEqual(bucket_for(16, [1, 4, 8]), 16)

    def test_ceil_div_rounds_up(self):
        """Ceil division is used for block counts and should round up."""
        self.assertEqual(ceil_div(9, 4), 3)
        self.assertEqual(ceil_div(8, 4), 2)

    def test_recurrent_padding_uses_dedicated_scratch_slot(self):
        """Graph capture and padded rows must not mutate a live request slot."""

        fake_graph = SimpleNamespace(replay=lambda: None)
        with patch("torch.cuda.CUDAGraph", return_value=fake_graph):
            graph = DecodeGraph(
                SimpleNamespace(),
                bucket=4,
                max_blocks_per_seq=2,
                scratch_block=9,
                scratch_recurrent_slot=4,
                device=torch.device("cpu"),
            )
        graph.logits_shard = torch.zeros(1, 4, 1)

        self.assertIsNone(graph.routing_capture)

        graph.replay_tensors(
            input_ids=torch.tensor([11, 12]),
            position_ids=torch.tensor([3, 7]),
            cache_seqlens=torch.tensor([3, 7], dtype=torch.int32),
            block_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            recurrent_slots=torch.tensor([0, 2]),
        )

        self.assertEqual(graph.recurrent_slots.tolist(), [0, 2, 4, 4])

    def test_agree_across_ranks_returns_local_verdict_without_distributed(self):
        """Single-process rollout keeps its own verdict; no collective is issued."""

        device = torch.device("cpu")
        self.assertTrue(agree_across_ranks(device, None, True))
        self.assertFalse(agree_across_ranks(device, None, False))

    def test_graph_capture_headroom_is_permissive_off_cuda(self):
        """CPU workers never capture, so the headroom check must not query CUDA."""

        self.assertEqual(graph_capture_headroom(torch.device("cpu"), 1 << 30), (True, 0))


def _run_capture_vote(rank: int, port: int, output_queue) -> None:
    """One gloo rank voting on capture; rank 1 reports a local failure."""

    import torch.distributed as dist

    # The parent process owns the server store, so both ranks join as clients.
    store = dist.TCPStore("127.0.0.1", port, 2, is_master=False)
    dist.init_process_group(backend="gloo", store=store, rank=rank, world_size=2)
    try:
        output_queue.put((rank, agree_across_ranks(torch.device("cpu"), None, rank == 0)))
    finally:
        dist.destroy_process_group()


class GraphCaptureVoteTest(unittest.TestCase):
    """A capture verdict must be unanimous or the group desynchronises."""

    def test_real_gloo_capture_vote_is_unanimous(self):
        spawn = mp.get_context("spawn")
        output_queue = spawn.Queue()
        store = _create_rendezvous_store("127.0.0.1", 2)
        port = int(store.port)
        processes = [spawn.Process(target=_run_capture_vote, args=(rank, port, output_queue)) for rank in range(2)]
        for process in processes:
            process.start()
        results = dict(output_queue.get(timeout=60) for _ in processes)
        for process in processes:
            process.join(timeout=60)
            self.assertEqual(process.exitcode, 0)

        # Rank 0 could have captured, but rank 1 could not: both must skip, or
        # rank 0 would replay a collective that rank 1 never joins.
        self.assertEqual(results, {0: False, 1: False})


class CaptureGraphOrderingTest(unittest.TestCase):
    """The cross-rank vote has to precede warmup, which runs TP collectives."""

    def _capture(self, *, headroom: bool, unanimous: bool) -> tuple[bool, list[str]]:
        """Drive `_capture_graph` with stubs and record the order of the steps."""

        from areno.engine import inference as inference_mod

        calls: list[str] = []
        # InferenceManager delegates attribute access to the worker it wraps.
        worker = SimpleNamespace(device=torch.device("cpu"), _decode_graph_warmup_peak=1 << 20)
        manager = inference_mod.InferenceManager(worker)

        graph = SimpleNamespace(
            warmup=lambda: calls.append("warmup") or (1 << 21),
            capture=lambda: calls.append("capture"),
        )

        def fake_agree(device, group, local_ok):
            del device, group
            calls.append(f"vote({local_ok})")
            return unanimous

        with (
            patch.object(inference_mod, "get_tp_context", lambda: SimpleNamespace(group=None, rank=1, is_rank0=False)),
            patch.object(inference_mod, "graph_capture_headroom", lambda device, want: (headroom, 1 << 30)),
            patch.object(inference_mod, "sync_before_graph_capture", lambda device, group: calls.append("barrier")),
            patch.object(inference_mod, "agree_across_ranks", fake_agree),
            # Capture only ever runs on CUDA; stub the accounting so this test
            # does not depend on whether an earlier test initialized CUDA.
            patch.object(torch.cuda, "memory_reserved", lambda device: 0),
        ):
            captured = manager._capture_graph(graph, "bucket=8")
        return captured, calls

    def test_vote_happens_before_warmup(self):
        """Warmup all-reduces, so a rank must not enter it before the group agrees."""

        captured, calls = self._capture(headroom=True, unanimous=True)

        self.assertTrue(captured)
        self.assertEqual(calls, ["barrier", "vote(True)", "warmup", "capture"])

    def test_a_peer_veto_skips_warmup_entirely(self):
        """A rank whose peer cannot capture must not run the model at all."""

        captured, calls = self._capture(headroom=True, unanimous=False)

        self.assertFalse(captured)
        self.assertEqual(calls, ["barrier", "vote(True)"])

    def test_local_veto_still_votes_so_peers_see_it(self):
        """The local verdict is voted on, never acted on alone."""

        captured, calls = self._capture(headroom=False, unanimous=False)

        self.assertFalse(captured)
        self.assertEqual(calls, ["barrier", "vote(False)"])

    def test_capture_oom_after_the_vote_is_fatal_with_guidance(self):
        """Past the vote there is no safe exit, so fail loudly instead of hanging."""

        from areno.engine import inference as inference_mod

        worker = SimpleNamespace(device=torch.device("cpu"), _decode_graph_warmup_peak=0)
        manager = inference_mod.InferenceManager(worker)

        def boom():
            raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 12.00 GiB")

        graph = SimpleNamespace(warmup=boom, capture=lambda: None)

        with (
            patch.object(inference_mod, "get_tp_context", lambda: SimpleNamespace(group=None, rank=3, is_rank0=False)),
            patch.object(inference_mod, "graph_capture_headroom", lambda device, want: (True, 1 << 30)),
            patch.object(inference_mod, "sync_before_graph_capture", lambda device, group: None),
            patch.object(inference_mod, "agree_across_ranks", lambda device, group, local_ok: True),
            patch.object(torch.cuda, "empty_cache", lambda: None),
            patch.object(torch.cuda, "memory_reserved", lambda device: 0),
        ):
            with self.assertRaises(RuntimeError) as raised:
                manager._capture_graph(graph, "verify bucket=8 tokens_per_seq=3")

        message = str(raised.exception)
        self.assertIn("rank 3", message)
        self.assertIn("--eager-decode", message)


class RolloutMergeTest(unittest.TestCase):
    """Rollout merge helpers rebuild padded tensors from variable rows."""

    def test_merge_rollouts_concatenates_chunks_and_sums_metrics(self):
        """Chunk merge should preserve row order and build response masks."""
        first = _rollout([[1]], [[2, 3]], [[-0.1, -0.2]], metrics={"tokens": 2})
        second = _rollout([[4, 5]], [[6]], [[-0.3]], metrics={"tokens": 1})

        merged = _merge_rollouts([first, second])

        self.assertEqual(merged.prompt_ids, [[1], [4, 5]])
        self.assertEqual(merged.input_ids.tolist(), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(merged.response_mask.tolist(), [[0, 1, 1], [0, 0, 1]])
        self.assertEqual(merged.metrics, {"tokens": 3.0})

    def test_merge_dp_rollouts_restores_original_prompt_order(self):
        """DP inverse merge should undo prompts[rank::dp_size] splitting."""
        dp0 = _rollout([[0], [2]], [[10], [12]], [[-0.1], [-0.3]], finish_reason=["stop", "length"])
        dp1 = _rollout([[1]], [[11]], [[-0.2]], finish_reason=["stop"])

        merged = _merge_dp_rollouts_in_input_order([dp0, dp1], total_count=3)

        self.assertEqual(merged.prompt_ids, [[0], [1], [2]])
        self.assertEqual(merged.response_ids, [[10], [11], [12]])
        self.assertEqual(merged.finish_reason, ["stop", "stop", "length"])

    def test_empty_rollout_has_empty_tensors(self):
        """No-prompt paths should return shape-safe empty tensors."""
        output = _empty_rollout()

        self.assertEqual(output.input_ids.shape, (0, 0))
        self.assertEqual(output.logprobs.shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
