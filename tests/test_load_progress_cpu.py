from __future__ import annotations

import logging
import unittest

from areno.engine.runtime.load_progress import (
    ModelLoadTracker,
    STAGE_CONFIG_TOKENIZER,
    STAGE_REFERENCE_RESOLUTION,
)


class ModelLoadTrackerTest(unittest.TestCase):
    """CPU-only tests for the model-loading progress tracker."""

    def _capture_logs(self, logger_name: str) -> list[str]:
        """Attach a list-collecting handler to ``logger_name`` and return it."""

        records: list[str] = []

        class _ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record.getMessage())

        handler = _ListHandler(level=logging.DEBUG)
        logger = logging.getLogger(logger_name)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        self.addCleanup(logger.removeHandler, handler)
        return records

    def test_rank0_emits_start_and_done_lines(self):
        tracker = ModelLoadTracker(rank0=True)
        logs = self._capture_logs("areno.engine.load_progress")

        with tracker.stage(STAGE_REFERENCE_RESOLUTION, detail="Qwen/Qwen3.5-0.8B"):
            pass

        self.assertTrue(any("status=start" in m and "reference_resolution" in m for m in logs))
        self.assertTrue(any("status=done" in m and "reference_resolution" in m for m in logs))
        self.assertEqual(tracker.last_completed_stage, STAGE_REFERENCE_RESOLUTION)

    def test_non_rank0_is_silent_but_still_tracks(self):
        tracker = ModelLoadTracker(rank0=False)
        logs = self._capture_logs("areno.engine.load_progress")

        with tracker.stage(STAGE_CONFIG_TOKENIZER):
            pass

        self.assertEqual(logs, [])
        # Stage is still recorded for structured consumers even when silent.
        self.assertEqual(tracker.last_completed_stage, STAGE_CONFIG_TOKENIZER)

    def test_failure_emits_failed_line_and_reraises(self):
        tracker = ModelLoadTracker(rank0=True)
        logs = self._capture_logs("areno.engine.load_progress")

        with self.assertRaises(RuntimeError):
            with tracker.stage(STAGE_CONFIG_TOKENIZER):
                raise RuntimeError("boom")

        self.assertTrue(any("status=failed" in m and "config_tokenizer_load" in m for m in logs))
        # The failing stage is NOT marked completed.
        self.assertIsNone(tracker.last_completed_stage)

    def test_last_completed_stage_updates_across_stages(self):
        tracker = ModelLoadTracker(rank0=True)

        with tracker.stage(STAGE_REFERENCE_RESOLUTION):
            pass
        self.assertEqual(tracker.last_completed_stage, STAGE_REFERENCE_RESOLUTION)

        with tracker.stage(STAGE_CONFIG_TOKENIZER):
            pass
        self.assertEqual(tracker.last_completed_stage, STAGE_CONFIG_TOKENIZER)

    def test_summary_returns_structured_snapshot(self):
        tracker = ModelLoadTracker(rank0=True)
        with tracker.stage(STAGE_REFERENCE_RESOLUTION):
            pass

        snapshot = tracker.summary()
        self.assertEqual(snapshot["last_completed_stage"], STAGE_REFERENCE_RESOLUTION)
        self.assertIsInstance(snapshot["total_elapsed_s"], float)
        self.assertGreaterEqual(snapshot["total_elapsed_s"], 0.0)


if __name__ == "__main__":
    unittest.main()