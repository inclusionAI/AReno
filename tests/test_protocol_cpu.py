from __future__ import annotations

import unittest
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_protocol_module():
    """Load protocol.py without importing areno.engine package side effects."""

    path = Path(__file__).resolve().parents[1] / "areno" / "engine" / "protocol.py"
    spec = importlib.util.spec_from_file_location("_areno_protocol_for_tests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol = _load_protocol_module()
TPCluster = protocol.TPCluster


class FakeQueue:
    """Small queue double that records close/join_thread calls."""

    def __init__(self):
        self.closed = False
        self.joined = False
        self.items = []

    def put(self, item):
        self.items.append(item)

    def close(self):
        self.closed = True

    def join_thread(self):
        self.joined = True


class FakeProcess:
    """Small process double for TPCluster.close resource cleanup tests."""

    def __init__(self, alive: bool):
        self._alive = alive
        self.join_calls = []
        self.terminated = False

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def is_alive(self):
        return self._alive

    def terminate(self):
        self.terminated = True
        self._alive = False


class TPClusterResourceTest(unittest.TestCase):
    """Protocol resource tests avoid spawning real multiprocessing workers."""

    def test_close_closes_command_and_result_queues(self):
        """TPCluster.close should release queue semaphores after worker shutdown."""
        cluster = object.__new__(TPCluster)
        cluster.config = SimpleNamespace(tp_size=1, dp_size=2)
        cluster.started = True
        cluster.cmd_queues = [FakeQueue(), FakeQueue()]
        cluster.result_queue = FakeQueue()
        cluster.processes = [FakeProcess(alive=False), FakeProcess(alive=True)]

        cluster.close()

        self.assertFalse(cluster.started)
        self.assertFalse(cluster.processes[1].is_alive())
        self.assertTrue(cluster.processes[1].terminated)
        self.assertEqual(cluster.processes[0].join_calls, [5, 0])
        self.assertEqual(cluster.processes[1].join_calls, [5, 0])
        for queue in [*cluster.cmd_queues, cluster.result_queue]:
            self.assertTrue(queue.closed)
            self.assertTrue(queue.joined)

    def test_coalesced_rollout_pending_ranks_follow_owner_dp(self):
        """A coalesced request should only wait for ranks in the DP lane that owns rows."""

        counts_by_dp = [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]

        self.assertEqual(protocol._coalesced_pending_ranks(counts_by_dp, 0, tp_size=2, world_size=4), {0, 1})
        self.assertEqual(protocol._coalesced_pending_ranks(counts_by_dp, 1, tp_size=2, world_size=4), {2, 3})

    def test_coalesced_rollout_merges_requests_with_different_capacity_limits(self):
        """Serve requests with different prompt lengths should still share one rollout batch."""

        def payload(prompt, *, max_cache_len, max_blocks_per_seq, max_prefill_tokens):
            return protocol.RolloutPayload(
                prompts_by_dp=[[prompt], []],
                prompt_indices_by_dp=[[0], []],
                max_new_tokens=16,
                eos_token_id=1,
                sampling_params=protocol.SamplingParams(temperature=0.7),
                max_running_seqs=1,
                max_cache_len=max_cache_len,
                max_blocks_per_seq=max_blocks_per_seq,
                max_prefill_tokens=max_prefill_tokens,
                num_blocks=max_blocks_per_seq,
                block_size=16,
                coalesce_max_running_seqs=2,
                coalesce_timeout_s=5.0,
                partial_tail_cooldown_until_s_by_dp=[[float(max_cache_len)], []],
            )

        short = payload([1, 2], max_cache_len=18, max_blocks_per_seq=2, max_prefill_tokens=4)
        long = payload([3, 4, 5, 6], max_cache_len=20, max_blocks_per_seq=3, max_prefill_tokens=8)

        self.assertTrue(protocol._rollout_payloads_compatible(short, long))
        merged = protocol._merge_rollout_payloads_for_cluster([short, long], [10, 11])

        self.assertEqual(merged.max_cache_len, 20)
        self.assertEqual(merged.max_blocks_per_seq, 3)
        self.assertEqual(merged.max_prefill_tokens, 8)
        self.assertEqual(merged.num_blocks, 6)
        self.assertEqual(merged.prompts_by_dp, [[[1, 2]], [[3, 4, 5, 6]]])
        self.assertEqual(merged.partial_tail_cooldown_until_s_by_dp, [[18.0], [20.0]])
        self.assertEqual(merged.coalesced_request_ids, [10, 11])
        self.assertEqual(merged.coalesced_counts_by_dp, [[1, 0], [0, 1]])

    def test_coalesced_rollout_recomputes_partial_tail_threshold_from_merged_batch(self):
        """Single-request payloads should get a real tail cutoff after queue batching."""

        payloads = [
            protocol.RolloutPayload(
                prompts_by_dp=[[[idx]]],
                prompt_indices_by_dp=[[0]],
                max_new_tokens=16,
                eos_token_id=1,
                sampling_params=protocol.SamplingParams(),
                max_running_seqs=1,
                max_cache_len=32,
                max_blocks_per_seq=2,
                max_prefill_tokens=16,
                num_blocks=2,
                block_size=16,
                coalesce_max_running_seqs=16,
                coalesce_timeout_s=5.0,
                partial_tail_threshold=0,
                partial_tail_min_tokens=256,
            )
            for idx in range(16)
        ]

        merged = protocol._merge_rollout_payloads_for_cluster(payloads, list(range(16)))

        self.assertEqual(merged.max_running_seqs, 16)
        self.assertEqual(merged.partial_tail_threshold, 2)
        self.assertEqual(merged.partial_tail_min_tokens, 256)


if __name__ == "__main__":
    unittest.main()
