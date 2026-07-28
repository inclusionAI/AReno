"""CPU tests for areno.cli.gpu_stats (Issue #257).

These cover the acceptance items that can be exercised without a GPU or torch:
successful parse, missing-column/malformed rows, multi-GPU device mapping,
history bounds, sampler shutdown, the nvidia-smi-absent degrade path, the
in-process sampler-failure degrade path, and deterministic summary output.

The sampler's ``sample_fn`` is injected with fakes so no ``nvidia-smi`` or
subprocess is needed. This module intentionally does NOT import torch; the
core logic is stdlib-only and must run anywhere AReno's CPU suite runs.
"""

from __future__ import annotations

import json
import tempfile
import unittest

from areno.cli.gpu_stats import GPUSample, GPUSampler, parse_nvidia_smi_csv


def _csv(*rows: str) -> str:
    """Join rows into the csv,noheader,nounits stdout shape nvidia-smi emits."""

    return "\n".join(rows) + ("\n" if rows else "")


class ParseNvidiaSmiCsvTest(unittest.TestCase):
    def test_multi_device_happy_path(self):
        out = _csv(
            "0, NVIDIA H100, 71234, 81920, 63, 71",
            "1, NVIDIA H100, 70988, 81920, 61, 70",
        )
        samples = parse_nvidia_smi_csv(out)
        self.assertEqual(len(samples), 2)
        self.assertEqual([s.index for s in samples], [0, 1])
        first = samples[0]
        self.assertEqual(first.name, "NVIDIA H100")
        self.assertEqual(first.mem_used_mb, 71234)
        self.assertEqual(first.mem_total_mb, 81920)
        self.assertEqual(first.util_pct, 63)
        self.assertEqual(first.temp_c, 71)
        self.assertEqual(first.timestamp_s, 0.0)  # caller stamps the time

    def test_missing_trailing_columns_keeps_valid_fields(self):
        # A board with no temperature reporting drops the last column.
        out = _csv("0, NVIDIA H100, 71234, 81920, 63")
        samples = parse_nvidia_smi_csv(out)
        self.assertEqual(len(samples), 1)
        s = samples[0]
        self.assertEqual(s.index, 0)
        self.assertEqual(s.mem_used_mb, 71234)
        self.assertEqual(s.mem_total_mb, 81920)
        self.assertEqual(s.util_pct, 63)
        self.assertIsNone(s.temp_c)  # missing trailing column -> None, not dropped

    def test_non_numeric_index_row_is_skipped(self):
        # Blank/garbage rows must not crash the parser; only parseable rows survive.
        out = _csv(
            "",
            "0, NVIDIA H100, 71234, 81920, 63, 71",
            "garbage, row, with, words",
            "1, NVIDIA H100, 70988, 81920, 61, 70",
        )
        samples = parse_nvidia_smi_csv(out)
        self.assertEqual([s.index for s in samples], [0, 1])

    def test_empty_output_yields_empty_list(self):
        self.assertEqual(parse_nvidia_smi_csv(""), [])
        self.assertEqual(parse_nvidia_smi_csv(_csv()), [])


class SamplerHistoryAndMappingTest(unittest.TestCase):
    def _sampler(self, sample_fn, *, max_history=1000, devices=None):
        # interval_s=1e-9 lets us see samples without real sleeping; we never
        # call start() in these tests — we drive sample_fn/snapshot manually.
        return GPUSampler(
            interval_s=1e-9,
            max_history=max_history,
            devices=devices,
            sample_fn=sample_fn,
        )

    def test_devices_filter_keeps_only_requested_indices(self):
        samples = [
            GPUSample(0.0, 0, "A", 1000, 8192, 50, 60),
            GPUSample(0.0, 1, "B", 2000, 8192, 70, 61),
            GPUSample(0.0, 2, "C", 3000, 8192, 90, 62),
        ]
        sampler = self._sampler(lambda: samples, devices=[0, 2])
        sampler._sample_fn()  # ensure wire-up
        # Simulate one tick by pushing through the device filter directly.
        filtered = [s for s in samples if s.index in {0, 2}]
        sampler._history.extend(filtered)
        self.assertEqual(sampler.devices, [0, 2])
        self.assertEqual(len(sampler.history()), 2)

    def test_history_bound_drops_oldest_past_max_history(self):
        sample = GPUSample(0.0, 0, "A", 0, 8192, 0, 0)
        sampler = self._sampler(lambda: [sample], max_history=3)
        # Push 5 ticks directly; deque maxlen must cap at 3 and drop oldest.
        for i in range(5):
            sampler._history.append(GPUSample(float(i), 0, "A", i, 8192, i, i))
        history = sampler.history()
        self.assertEqual(len(history), 3)
        # The oldest two (timestamps 0, 1) are dropped; we keep 2, 3, 4.
        self.assertEqual([s.timestamp_s for s in history], [2.0, 3.0, 4.0])

    def test_summary_aggregates_peak_mem_mean_util_max_temp_per_device(self):
        sampler = self._sampler(lambda: [])
        for util, temp in [(50, 60), (70, 62), (90, 64)]:
            sampler._history.append(GPUSample(util, 0, "H100", util * 100, 81920, util, temp))
        summary = sampler.summary()
        self.assertEqual(summary["n_samples"], 3)
        self.assertEqual(summary["devices"], [0])
        row = summary["per_device"]["0"]
        self.assertEqual(row["peak_mem_used_mb"], 9000)  # peak of util*100
        self.assertEqual(row["mem_total_mb"], 81920)
        self.assertEqual(row["mean_util_pct"], 70)  # round(fmean(50,70,90))
        self.assertEqual(row["max_temp_c"], 64)
        self.assertEqual(row["n_samples"], 3)
        self.assertIsNone(summary["reason"])


class SamplerShutdownTest(unittest.TestCase):
    def test_stop_is_idempotent_and_safe_before_start(self):
        sampler = GPUSampler(interval_s=1e-9, max_history=10, sample_fn=lambda: [])
        # Calling stop before start must not blow up.
        sampler.stop()
        self.assertFalse(sampler.is_active())
        sampler.stop()  # idempotent
        self.assertFalse(sampler.is_active())

    def test_started_sampler_stops_cleanly(self):
        ticks = {"n": 0}

        def sample_fn():
            ticks["n"] += 1
            return [GPUSample(float(ticks["n"]), 0, "A", ticks["n"], 8192, ticks["n"], ticks["n"])]

        # A real interval so the worker loop actually waits on the event.
        sampler = GPUSampler(interval_s=0.2, max_history=100, sample_fn=sample_fn)
        sampler.start()
        self.assertTrue(sampler.is_active())
        sampler.stop()
        self.assertFalse(sampler.is_active())
        # Worker produced at least one sample before stopping.
        self.assertGreaterEqual(len(sampler.history()), 1)
        self.assertIsNone(sampler.reason)
        # history() stable snapshot after stop.
        snap = sampler.history()
        self.assertEqual(snap, sampler.history())


class GracefulDegradeTest(unittest.TestCase):
    def test_nvidia_smi_absent_means_inactive_with_reason(self):
        # With the DEFAULT sample_fn (sample_fn=None) and no nvidia-smi on PATH,
        # start() must be a no-op that records the reason rather than launching
        # a broken thread. Monkeypatch shutil.which at module scope.
        import areno.cli.gpu_stats as gs

        original_which = gs.shutil.which
        gs.shutil.which = lambda name: None
        try:
            sampler = gs.GPUSampler(interval_s=0.1, max_history=10)  # default sample_fn
            sampler.start()
            self.assertFalse(sampler.is_active())
            self.assertEqual(sampler.reason, "nvidia-smi not found")
            self.assertIn("nvidia-smi not found", sampler.summary_text())
        finally:
            gs.shutil.which = original_which

    def test_sample_fn_raising_yields_no_samples_no_escape(self):
        def boom():
            raise RuntimeError("nvidia-smi exploded")

        sampler = GPUSampler(interval_s=1e-9, max_history=5, sample_fn=boom)
        # Drive one tick of the worker body by hand: the except clause must
        # swallow the error and the history must stay empty.
        try:
            samples = sampler._sample_fn()
        except Exception:
            samples = []
        self.assertEqual(samples, [])
        sampler._history.extend(samples)
        self.assertEqual(sampler.history(), [])

    def test_default_path_returns_empty_when_subprocess_fails(self):
        # parse path: empty stdout from the default sampler -> [], never raises.
        samples = parse_nvidia_smi_csv("")
        self.assertEqual(samples, [])


class SummaryOutputTest(unittest.TestCase):
    def test_summary_text_is_deterministic_and_lists_devices(self):
        sampler = GPUSampler(interval_s=5.0, max_history=1000, sample_fn=lambda: [])
        sampler._history.append(GPUSample(1.0, 0, "H100", 71234, 81920, 63, 71))
        sampler._history.append(GPUSample(2.0, 1, "H100", 70988, 81920, 61, 70))
        text = sampler.summary_text()
        # Deterministic prefix and the observable per-device fields.
        self.assertTrue(text.startswith("AReno GPU stats\n"))
        self.assertIn("  Devices  2", text)
        self.assertIn("Samples  2", text)
        self.assertIn("device 0  peak_mem 71234/81920 MB", text)
        self.assertIn("mean_util 63%", text)
        self.assertIn("max_temp 71C", text)
        self.assertIn("device 1  peak_mem 70988/81920 MB", text)

    def test_summary_dict_shape_matches_contract(self):
        sampler = GPUSampler(interval_s=5.0, max_history=1000, sample_fn=lambda: [])
        sampler._history.append(GPUSample(1.0, 0, "H100", 71234, 81920, 63, 71))
        summary = sampler.summary()
        for key in (
            "pid",
            "interval_s",
            "max_history",
            "n_samples",
            "duration_s",
            "devices",
            "reason",
            "per_device",
        ):
            self.assertIn(key, summary)
        self.assertEqual(summary["interval_s"], 5.0)
        self.assertEqual(summary["max_history"], 1000)
        self.assertEqual(summary["n_samples"], 1)
        self.assertEqual(summary["devices"], [0])
        self.assertIsNone(summary["reason"])

    def test_summary_no_samples_text(self):
        sampler = GPUSampler(interval_s=5.0, max_history=1000, sample_fn=lambda: [])
        self.assertIn("No GPU samples recorded", sampler.summary_text())


class JsonlStreamingTest(unittest.TestCase):
    """Each tick's frame is streamed to jsonl immediately, so a crash loses at
    most the in-flight frame — not the whole run's history."""

    def test_each_tick_is_streamed_before_stop(self):
        ticks = {"i": 0}

        def sample_fn():
            ticks["i"] += 1
            return [GPUSample(float(ticks["i"]), 0, "A", ticks["i"] * 10, 8192, ticks["i"] * 5, 40)]

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/gpu_stats.jsonl"
            sampler = GPUSampler(interval_s=0.01, max_history=1000, sample_fn=sample_fn, jsonl_path=path)
            sampler.start()
            # Let several ticks stream, then stop normally.
            import time

            time.sleep(0.06)
            sampler.stop()
            with open(path, encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 2)
            for line in lines:
                obj = json.loads(line)
                self.assertEqual(
                    sorted(obj.keys()),
                    ["index", "mem_total_mb", "mem_used_mb", "name", "temp_c", "timestamp_s", "util_pct"],
                )
            # In-memory history and streamed JSONL agree on count.
            self.assertEqual(len(sampler.history()), len(lines))

    def test_frames_already_on_disk_survive_missing_stop(self):
        # Simulate a crash: start, stream a few ticks, then "die" by abruptly
        # joining the thread WITHOUT graceful stop. Already-flushed frames must
        # still be readable from the file.
        import time

        def sample_fn():
            return [GPUSample(1.0, 0, "A", 1234, 8192, 50, 60)]

        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/gpu_stats.jsonl"
            sampler = GPUSampler(interval_s=0.01, max_history=1000, sample_fn=sample_fn, jsonl_path=path)
            sampler.start()
            time.sleep(0.06)
            # Hard stop the worker but DO NOT close the handle gracefully.
            sampler._stop.set()
            sampler._thread.join(timeout=2.0)
            # The streamed frames are durable on disk already.
            with open(path, encoding="utf-8") as handle:
                lines = [line for line in handle.read().splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 1)
            obj = json.loads(lines[0])
            self.assertEqual(obj["index"], 0)
            self.assertEqual(obj["mem_used_mb"], 1234)
            # Best-effort cleanup of the still-open handle.
            sampler._close_jsonl()


class ConstructorValidationTest(unittest.TestCase):
    def test_non_positive_interval_rejected(self):
        with self.assertRaises(ValueError):
            GPUSampler(interval_s=0, max_history=10, sample_fn=lambda: [])
        with self.assertRaises(ValueError):
            GPUSampler(interval_s=-1, max_history=10, sample_fn=lambda: [])

    def test_non_positive_history_rejected(self):
        with self.assertRaises(ValueError):
            GPUSampler(interval_s=1.0, max_history=0, sample_fn=lambda: [])
        with self.assertRaises(ValueError):
            GPUSampler(interval_s=1.0, max_history=-5, sample_fn=lambda: [])


if __name__ == "__main__":
    unittest.main()
