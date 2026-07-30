from __future__ import annotations

import unittest

from areno.dashboard.server import Job, _parse_time, _duration_seconds, _launch_value


class ParseTimeTest(unittest.TestCase):
    """_parse_time handles ISO strings, epoch floats, and bad inputs."""

    def test_parse_iso_timestamp(self):
        """An ISO-8601 string with timezone produces a valid epoch float."""
        result = _parse_time("2026-01-01T00:00:00+00:00")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 1767225600.0, places=1)

    def test_parse_epoch_float(self):
        """A raw float is returned as-is."""
        self.assertEqual(_parse_time(1700000000.0), 1700000000.0)

    def test_parse_epoch_int(self):
        """An int is also accepted."""
        self.assertEqual(_parse_time(1700000000), 1700000000.0)

    def test_parse_none(self):
        self.assertIsNone(_parse_time(None))

    def test_parse_garbage(self):
        self.assertIsNone(_parse_time("not-a-date"))


class DurationSecondsTest(unittest.TestCase):
    """_duration_seconds computes elapsed time with a non-negative floor."""

    def test_iso_pair(self):
        start = "2026-01-01T00:00:00+00:00"
        end = "2026-01-01T00:10:00+00:00"
        self.assertAlmostEqual(_duration_seconds(start, end), 600.0, places=1)

    def test_epoch_pair(self):
        self.assertAlmostEqual(_duration_seconds(100.0, 125.5), 25.5)

    def test_start_unparseable(self):
        self.assertIsNone(_duration_seconds("bad", "2026-01-01T00:00:00+00:00"))

    def test_end_unparseable(self):
        self.assertIsNone(_duration_seconds(100.0, None))

    def test_negative_floored_to_zero(self):
        """If updated_at predates created_at the duration is 0, not negative."""
        self.assertEqual(_duration_seconds(200.0, 100.0), 0.0)

    def test_running_uses_now(self):
        """Running jobs compute duration to now(), not updated_at."""
        start = "2020-01-01T00:00:00+00:00"
        stale_updated = "2020-01-01T00:01:00+00:00"
        result = _duration_seconds(start, stale_updated, status="running")
        self.assertIsNotNone(result)
        self.assertGreater(result, 60.0)

    def test_terminal_uses_updated_at(self):
        """Terminal jobs use updated_at, not now()."""
        start = "2026-01-01T00:00:00+00:00"
        end = "2026-01-01T00:05:00+00:00"
        result = _duration_seconds(start, end, status="exited")
        self.assertAlmostEqual(result, 300.0, places=1)


def _make_job(**overrides):
    """Create a minimal Job with sensible defaults for summary tests."""
    defaults = dict(
        kind="train",
        name="test job",
        command=["areno", "train"],
        config={"algo": "gspo", "ckpt": "Qwen/Qwen3-0.6B", "dataset_path": "gsm8k:main"},
        metrics_dir="/tmp/areno/metrics",
    )
    defaults.update(overrides)
    return Job(**defaults)


class SummaryJsonTest(unittest.TestCase):
    """to_summary_json surfaces fields needed for dashboard search/filter/sort."""

    def test_new_fields_present(self):
        job = _make_job()
        summary = job.to_summary_json()
        self.assertEqual(summary["algo"], "gspo")
        self.assertEqual(summary["ckpt"], "Qwen/Qwen3-0.6B")
        self.assertEqual(summary["dataset_path"], "gsm8k:main")
        self.assertIn("duration_s", summary)

    def test_existing_fields_unchanged(self):
        """All fields that existed before the change are still present."""
        job = _make_job()
        summary = job.to_summary_json()
        for key in (
            "id", "kind", "name", "metrics_dir", "status", "stage", "role",
            "step", "created_at", "updated_at", "returncode", "pid", "perf",
        ):
            self.assertIn(key, summary, f"missing pre-existing field: {key}")

    def test_duration_calculated_from_timestamps(self):
        job = _make_job()
        job.created_at = "2026-01-01T00:00:00+00:00"
        job.updated_at = "2026-01-01T00:05:00+00:00"
        summary = job.to_summary_json()
        self.assertAlmostEqual(summary["duration_s"], 300.0, places=1)

    def test_duration_none_when_created_at_invalid(self):
        """A job with a corrupted created_at yields None, not an exception."""
        job = _make_job()
        job.created_at = "garbage"
        job.updated_at = "2026-01-01T00:00:00+00:00"
        summary = job.to_summary_json()
        self.assertIsNone(summary["duration_s"])

    def test_duration_running_uses_now(self):
        """Running jobs compute duration against now(), not stale updated_at."""
        job = _make_job()
        job.status = "running"
        job.created_at = "2020-01-01T00:00:00+00:00"
        job.updated_at = "2020-01-01T00:01:00+00:00"
        summary = job.to_summary_json()
        self.assertIsNotNone(summary["duration_s"])
        self.assertGreater(summary["duration_s"], 60.0)

    def test_serve_job_uses_model_path(self):
        """Serve jobs store the model under model_path, not ckpt."""
        job = _make_job(
            kind="serve",
            config={"model_path": "Qwen/Qwen3-0.6B"},
        )
        summary = job.to_summary_json()
        self.assertEqual(summary["ckpt"], "Qwen/Qwen3-0.6B")
        self.assertEqual(summary["algo"], "")

    def test_empty_launch_config(self):
        """A job whose launch_config was cleared still returns safe defaults."""
        job = _make_job()
        job.launch_config = {}
        summary = job.to_summary_json()
        self.assertEqual(summary["algo"], "")
        self.assertEqual(summary["ckpt"], "")
        self.assertEqual(summary["dataset_path"], "")

    def test_none_launch_config(self):
        """A job whose launch_config is None returns safe defaults."""
        job = _make_job()
        job.launch_config = None  # type: ignore[assignment]
        summary = job.to_summary_json()
        self.assertEqual(summary["algo"], "")
        self.assertEqual(summary["ckpt"], "")

    def test_from_json_preserves_launch_fields(self):
        """A job restored from JSON retains launch_config for summary fields."""
        original = _make_job()
        payload = original.to_json()
        restored = Job.from_json(payload)
        summary = restored.to_summary_json()
        self.assertEqual(summary["algo"], "gspo")
        self.assertEqual(summary["ckpt"], "Qwen/Qwen3-0.6B")
        self.assertEqual(summary["dataset_path"], "gsm8k:main")


class LaunchValueTest(unittest.TestCase):
    """_launch_value reads from both flat dict and sections format."""

    def test_flat_dict(self):
        launch = {"algo": "gspo", "ckpt": "Qwen/Qwen3-0.6B"}
        self.assertEqual(_launch_value(launch, "algo"), "gspo")
        self.assertEqual(_launch_value(launch, "ckpt"), "Qwen/Qwen3-0.6B")

    def test_sections_format(self):
        """CLI-registered jobs store launch config as sections arrays."""
        launch = {
            "sections": [
                {"title": "Basic", "items": [
                    {"key": "algo", "value": "gspo"},
                    {"key": "ckpt", "value": "Qwen/Qwen3-0.6B"},
                    {"key": "dataset_path", "value": "tictactoe.jsonl"},
                ]},
                {"title": "Runtime", "items": [
                    {"key": "tp_size", "value": 2},
                ]},
            ]
        }
        self.assertEqual(_launch_value(launch, "algo"), "gspo")
        self.assertEqual(_launch_value(launch, "ckpt"), "Qwen/Qwen3-0.6B")
        self.assertEqual(_launch_value(launch, "dataset_path"), "tictactoe.jsonl")
        self.assertEqual(_launch_value(launch, "tp_size"), 2)

    def test_missing_key_returns_default(self):
        self.assertEqual(_launch_value({"sections": []}, "algo"), "")
        self.assertEqual(_launch_value({}, "algo"), "")

    def test_none_launch(self):
        self.assertEqual(_launch_value(None, "algo"), "")

    def test_serve_sections_with_model_path(self):
        launch = {
            "sections": [
                {"title": "Basic", "items": [
                    {"key": "model_path", "value": "Qwen/Qwen3-0.6B"},
                ]},
            ]
        }
        self.assertEqual(_launch_value(launch, "model_path"), "Qwen/Qwen3-0.6B")


class SummarySectionsFormatTest(unittest.TestCase):
    """to_summary_json works with sections-format launch_config (CLI jobs)."""

    def test_sections_format_summary(self):
        job = _make_job()
        job.launch_config = {
            "sections": [
                {"title": "Basic", "items": [
                    {"key": "algo", "value": "gspo"},
                    {"key": "ckpt", "value": "Qwen/Qwen3-0.6B"},
                    {"key": "dataset_path", "value": "tictactoe.jsonl"},
                ]},
            ]
        }
        summary = job.to_summary_json()
        self.assertEqual(summary["algo"], "gspo")
        self.assertEqual(summary["ckpt"], "Qwen/Qwen3-0.6B")
        self.assertEqual(summary["dataset_path"], "tictactoe.jsonl")

    def test_serve_sections_format_summary(self):
        job = _make_job(kind="serve")
        job.launch_config = {
            "sections": [
                {"title": "Basic", "items": [
                    {"key": "model_path", "value": "Qwen/Qwen3-0.6B"},
                ]},
            ]
        }
        summary = job.to_summary_json()
        self.assertEqual(summary["ckpt"], "Qwen/Qwen3-0.6B")
        self.assertEqual(summary["algo"], "")

    def test_from_json_with_sections_launch(self):
        """A job restored from JSON with sections launch retains summary fields."""
        job = _make_job()
        job.launch_config = {
            "sections": [
                {"title": "Basic", "items": [
                    {"key": "algo", "value": "dpo"},
                    {"key": "ckpt", "value": "Qwen/Qwen2.5-0.5B"},
                    {"key": "dataset_path", "value": "alpaca.jsonl"},
                ]},
            ]
        }
        payload = job.to_json()
        restored = Job.from_json(payload)
        summary = restored.to_summary_json()
        self.assertEqual(summary["algo"], "dpo")
        self.assertEqual(summary["ckpt"], "Qwen/Qwen2.5-0.5B")
        self.assertEqual(summary["dataset_path"], "alpaca.jsonl")


if __name__ == "__main__":
    unittest.main()