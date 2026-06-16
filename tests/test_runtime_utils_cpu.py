from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

import areno.engine.runtime.common as runtime_common
from areno.engine.api import _merge_dp_rank0_strided_results
from areno.engine.data import RolloutOutput
from areno.engine.runtime.common import (
    _check_token_ids,
    _device_long,
    dp_rank0_results,
    merge_metric_dicts,
    merge_train_stats,
    split_data_pack_by_dp,
    split_list_by_dp,
)
from areno.engine.runtime.decode_graph import bucket_for, ceil_div
from areno.engine.runtime.rollout import _empty_rollout, _merge_dp_rollouts_in_input_order, _merge_rollouts


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
