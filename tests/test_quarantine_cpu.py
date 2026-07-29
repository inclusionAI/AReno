"""CPU tests for the quarantine module.

These tests exercise the full QuarantineWriter lifecycle without requiring
GPU hardware or any AReno backend.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

from areno.engine.quarantine import (
    QuarantineConfig,
    QuarantineThresholdExceeded,
    QuarantineWriter,
    _truncated_hash,
)


class TestQuarantineDisabled(unittest.TestCase):
    """When disabled, the writer should be a complete no-op."""

    def test_disabled_does_not_create_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(enabled=False, output_dir=tmp)
            writer = QuarantineWriter(cfg)
            writer.record(phase="reward", reason="boom", sample_meta={})
            writer.record_success()
            writer.close()
            self.assertEqual(writer.entry_count, 0)
            self.assertFalse(any(Path(tmp).iterdir()))

    def test_disabled_default_config(self):
        cfg = QuarantineConfig()
        self.assertFalse(cfg.enabled)


class TestQuarantineWriting(unittest.TestCase):
    """Verify that valid entries are written correctly."""

    def test_writes_jsonl_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(enabled=True, output_dir=tmp, max_entries=10)
            writer = QuarantineWriter(cfg)
            writer.record(
                phase="reward",
                reason="ValueError: bad parse",
                sample_meta={
                    "step": 5,
                    "epoch": 0,
                    "prompt_index": 1,
                    "sample_index": 2,
                    "prompt_len": 10,
                    "completion_len": 5,
                    "prompt_text": "secret prompt",
                    "completion_text": "secret answer",
                },
            )
            writer.close()
            files = list(Path(tmp).glob("quarantine.*.jsonl"))
            self.assertEqual(len(files), 1)
            lines = files[0].read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["phase"], "reward")
            self.assertEqual(entry["reason"], "ValueError: bad parse")
            self.assertEqual(entry["step"], 5)
            self.assertEqual(entry["prompt_index"], 1)
            self.assertEqual(entry["sample_index"], 2)

    def test_redacts_sensitive_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(enabled=True, output_dir=tmp)
            writer = QuarantineWriter(cfg)
            writer.record(
                phase="generation",
                reason="empty",
                sample_meta={
                    "prompt_text": "very secret prompt",
                    "completion_text": "very secret completion",
                },
            )
            writer.close()
            file = next(Path(tmp).glob("quarantine.*.jsonl"))
            raw = file.read_text(encoding="utf-8")
            self.assertNotIn("very secret prompt", raw)
            self.assertNotIn("very secret completion", raw)
            entry = json.loads(raw.strip())
            self.assertIn("prompt_hash", entry)
            self.assertIn("completion_hash", entry)
            self.assertIsNone(entry.get("prompt_text"))
            self.assertIsNone(entry.get("completion_text"))

    def test_truncated_hash(self):
        h = _truncated_hash("hello")
        self.assertIsNotNone(h)
        self.assertEqual(len(h), 16)
        self.assertIsNone(_truncated_hash(None))


class TestQuarantineBounds(unittest.TestCase):
    """max_entries and max_file_bytes should be enforced."""

    def test_max_entries_freezes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=3,
                failure_rate_threshold=1.0, failure_rate_window=100,
            )
            writer = QuarantineWriter(cfg)
            for i in range(5):
                writer.record(phase="reward", reason=f"err {i}", sample_meta={})
            writer.close()
            self.assertEqual(writer.entry_count, 3)
            self.assertTrue(writer.frozen)

    def test_max_file_bytes_freezes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=10000,
                max_file_bytes=512,
                failure_rate_threshold=1.0, failure_rate_window=100,
            )
            writer = QuarantineWriter(cfg)
            for i in range(200):
                writer.record(
                    phase="reward",
                    reason="x" * 100,
                    sample_meta={"prompt_text": "p" * 50},
                )
            writer.close()
            self.assertTrue(writer.frozen)
            file = next(Path(tmp).glob("quarantine.*.jsonl"))
            # File should be frozen shortly after exceeding max_file_bytes.
            # The last entry may push the file past the limit, so allow generous headroom.
            self.assertLessEqual(file.stat().st_size, 512 + 512)


class TestQuarantineThreshold(unittest.TestCase):
    """The failure-rate threshold should protect against systemic failures."""

    def test_threshold_exceeded_raises(self):
        cfg = QuarantineConfig(
            enabled=True, output_dir=None,
            failure_rate_threshold=0.5, failure_rate_window=4,
        )
        writer = QuarantineWriter(cfg)
        # 4 consecutive failures with a 0.5 threshold and window 4
        with self.assertRaises(QuarantineThresholdExceeded):
            writer.record(phase="reward", reason="e1", sample_meta={})
            writer.record(phase="reward", reason="e2", sample_meta={})
            writer.record(phase="reward", reason="e3", sample_meta={})
            writer.record(phase="reward", reason="e4", sample_meta={})

    def test_threshold_not_triggered_by_isolated_failures(self):
        cfg = QuarantineConfig(
            enabled=True, output_dir=None,
            failure_rate_threshold=0.5, failure_rate_window=10,
        )
        writer = QuarantineWriter(cfg)
        # 1 failure among 10 samples -> 10% < 50%
        for _ in range(9):
            writer.record_success()
        writer.record(phase="reward", reason="isolated", sample_meta={})
        self.assertFalse(writer.frozen)

    def test_threshold_carries_original_error(self):
        exc = QuarantineThresholdExceeded("rate too high", original_error=ValueError("root cause"))
        self.assertIsInstance(exc.original_error, ValueError)
        self.assertEqual(str(exc.original_error), "root cause")


class TestQuarantineConcurrency(unittest.TestCase):
    """Concurrent writes from multiple threads should not corrupt the file."""

    def test_concurrent_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=500,
                failure_rate_threshold=1.0, failure_rate_window=1000,
            )
            writer = QuarantineWriter(cfg)
            errors = []

            def worker(tid):
                for i in range(50):
                    try:
                        writer.record(
                            phase="reward",
                            reason=f"thread-{tid}-err-{i}",
                            sample_meta={"step": i},
                        )
                    except Exception as exc:
                        errors.append(exc)

            threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            writer.close()
            self.assertEqual(len(errors), 0)
            self.assertGreater(writer.entry_count, 0)
            # Verify all lines are valid JSON
            file = next(Path(tmp).glob("quarantine.*.jsonl"))
            for line in file.read_text(encoding="utf-8").strip().splitlines():
                json.loads(line)  # should not raise


class TestQuarantineSerialization(unittest.TestCase):
    """Non-serializable values in sample_meta should not crash the writer."""

    def test_non_serializable_meta_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=5,
                failure_rate_threshold=1.0, failure_rate_window=100,
            )
            writer = QuarantineWriter(cfg)

            class Unserializable:
                pass

            # _build_entry only pulls known keys from sample_meta, so
            # unknown non-serializable values are simply ignored.
            writer.record(
                phase="agent",
                reason="bad",
                sample_meta={"step": 0, "junk": Unserializable()},
            )
            writer.close()
            self.assertEqual(writer.entry_count, 1)

    def test_none_values_handled(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=5,
                failure_rate_threshold=1.0, failure_rate_window=100,
            )
            writer = QuarantineWriter(cfg)
            writer.record(phase="reward", reason="e", sample_meta={})
            writer.close()
            file = next(Path(tmp).glob("quarantine.*.jsonl"))
            entry = json.loads(file.read_text(encoding="utf-8").strip())
            self.assertIsNone(entry["step"])
            self.assertIsNone(entry["prompt_index"])


class TestQuarantineClose(unittest.TestCase):
    """close() should flush and close the file handle."""

    def test_close_flushes_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=10,
                failure_rate_threshold=1.0, failure_rate_window=100,
            )
            writer = QuarantineWriter(cfg)
            writer.record(phase="reward", reason="e1", sample_meta={"step": 1})
            writer.record(phase="reward", reason="e2", sample_meta={"step": 2})
            writer.close()
            file = next(Path(tmp).glob("quarantine.*.jsonl"))
            lines = file.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)

    def test_close_is_idempotent(self):
        cfg = QuarantineConfig(enabled=True, output_dir="/tmp/areno_test_quarantine")
        writer = QuarantineWriter(cfg)
        writer.close()
        writer.close()  # should not raise


class TestQuarantineRecordSuccess(unittest.TestCase):
    """record_success should track successes without writing entries."""

    def test_success_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=10,
                failure_rate_threshold=0.5, failure_rate_window=5,
            )
            writer = QuarantineWriter(cfg)
            for _ in range(3):
                writer.record_success()
            # 3 successes, no failures -> 0 entries, not frozen
            self.assertEqual(writer.entry_count, 0)
            self.assertFalse(writer.frozen)

    def test_mixed_success_failure_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=10,
                failure_rate_threshold=0.6, failure_rate_window=5,
            )
            writer = QuarantineWriter(cfg)
            # 2 failures out of 5 -> 40% < 60%
            writer.record(phase="reward", reason="e1", sample_meta={})
            writer.record_success()
            writer.record(phase="reward", reason="e2", sample_meta={})
            writer.record_success()
            writer.record_success()
            self.assertFalse(writer.frozen)
            self.assertEqual(writer.entry_count, 2)


class TestQuarantineIntegrationScenario(unittest.TestCase):
    """Integration-style tests simulating reward_fn failures during training."""

    def test_partial_reward_failure_continues_and_quarantines(self):
        """A reward_fn that fails on 1 of 4 samples: quarantine gets 1 entry,
        training continues, failed sample gets reward 0.0."""

        with tempfile.TemporaryDirectory() as tmp:
            cfg = QuarantineConfig(
                enabled=True, output_dir=tmp, max_entries=100,
                failure_rate_threshold=0.5, failure_rate_window=10,
            )
            writer = QuarantineWriter(cfg)

            # Simulate 4 reward calls: 3 succeed, 1 fails.
            rewards_all = []
            for i in range(4):
                try:
                    if i == 2:
                        raise ValueError(f"bad sample {i}")
                    rewards_all.append(float(i))
                    writer.record_success()
                except QuarantineThresholdExceeded:
                    raise
                except Exception as exc:
                    writer.record(
                        phase="reward",
                        reason=f"{type(exc).__name__}: {exc}",
                        sample_meta={"prompt_index": 0, "sample_index": i, "prompt_text": f"prompt_{i}"},
                    )
                    rewards_all.append(0.0)
            writer.close()

            # Training continued — 4 rewards produced.
            self.assertEqual(len(rewards_all), 4)
            self.assertEqual(rewards_all, [0.0, 1.0, 0.0, 3.0])

            # Quarantine has 1 entry for the failed sample.
            self.assertEqual(writer.entry_count, 1)
            file = next(Path(tmp).glob("quarantine.*.jsonl"))
            entry = json.loads(file.read_text(encoding="utf-8").strip())
            self.assertEqual(entry["phase"], "reward")
            self.assertIn("ValueError", entry["reason"])
            self.assertEqual(entry["sample_index"], 2)
            self.assertNotIn("prompt_2", file.read_text(encoding="utf-8"))  # redacted

    def test_all_reward_failures_propagate_via_threshold(self):
        """A reward_fn that always fails: quarantine records the first few,
        then QuarantineThresholdExceeded propagates the original error."""

        cfg = QuarantineConfig(
            enabled=True, output_dir=None,
            failure_rate_threshold=0.5, failure_rate_window=4,
        )
        writer = QuarantineWriter(cfg)
        propagated = None
        try:
            for i in range(10):
                try:
                    raise RuntimeError(f"always fails {i}")
                except QuarantineThresholdExceeded:
                    raise
                except Exception as exc:
                    writer.record(
                        phase="reward",
                        reason=f"{type(exc).__name__}: {exc}",
                        sample_meta={"prompt_index": 0, "sample_index": i},
                    )
        except QuarantineThresholdExceeded as exc:
            propagated = exc

        self.assertIsNotNone(propagated)
        self.assertTrue(writer.frozen)
        # The threshold fired after the window was full (4 consecutive failures).
        self.assertLessEqual(writer.entry_count, 4)

    def test_disabled_quarantine_preserves_original_behavior(self):
        """When disabled, a reward_fn failure should propagate as-is
        (no quarantine interception, no reward=0.0 substitution)."""

        cfg = QuarantineConfig(enabled=False, output_dir="/tmp/unused")
        writer = QuarantineWriter(cfg)

        propagated_exc = None
        try:
            try:
                raise ValueError("original error")
            except QuarantineThresholdExceeded:
                raise
            except Exception as exc:
                writer.record(
                    phase="reward",
                    reason=f"{type(exc).__name__}: {exc}",
                    sample_meta={},
                )
                raise  # In disabled mode, caller still raises
        except ValueError as exc:
            propagated_exc = exc

        self.assertIsNotNone(propagated_exc)
        self.assertEqual(str(propagated_exc), "original error")
        self.assertEqual(writer.entry_count, 0)


if __name__ == "__main__":
    unittest.main()
