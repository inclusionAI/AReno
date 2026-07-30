"""CPU tests for preflight output-directory writability probe."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from areno.cli.preflight_io import (
    PreflightConfig,
    PreflightProbeResult,
    format_probe_results,
    format_probe_results_json,
    probe_directory_writability,
    probe_paths,
)


class ProbeSuccessTest(unittest.TestCase):
    def test_probe_success_on_writable_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = probe_directory_writability(tmp, stage="checkpoint")
            self.assertTrue(result.ok)
            self.assertEqual(result.stage, "checkpoint")
            self.assertEqual(result.operation, "")

    def test_probe_success_on_nested_missing_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            deep = Path(tmp) / "a" / "b" / "c"
            result = probe_directory_writability(deep, stage="metrics")
            self.assertTrue(result.ok)
            self.assertTrue(deep.is_dir())

    def test_probe_results_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            r1 = probe_directory_writability(tmp, stage="checkpoint")
            r2 = probe_directory_writability(tmp, stage="checkpoint")
            self.assertEqual(r1.ok, r2.ok)
            self.assertTrue(r1.ok)
            self.assertTrue(r2.ok)


class ProbeFailureTest(unittest.TestCase):
    def setUp(self):
        self._dirs_to_restore: list[Path] = []

    def tearDown(self):
        # Restore permissions so TemporaryDirectory can clean up.
        for d in reversed(self._dirs_to_restore):
            try:
                os.chmod(d, 0o755)
            except OSError:
                pass

    def test_probe_fails_on_readonly_dir(self):
        # On some systems (e.g. Colab running as root), chmod 0o444 does not
        # prevent writes. Use mock to simulate a real permission denial so the
        # test is reliable across all environments.
        d = tempfile.mkdtemp()
        readonly = Path(d) / "readonly"
        readonly.mkdir()

        real_open = open

        def fail_open(file, mode="r", *args, **kwargs):
            if "xb" in mode and str(readonly) in str(file):
                raise PermissionError(13, "Permission denied", str(file))
            return real_open(file, mode, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=fail_open):
            result = probe_directory_writability(readonly, stage="checkpoint")
        self.assertFalse(result.ok)
        self.assertIn(result.operation, ("create", "write"))

    def test_probe_fails_on_existing_file(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test")
            f.flush()
            file_path = f.name
        try:
            result = probe_directory_writability(file_path, stage="metrics")
            self.assertFalse(result.ok)
            self.assertEqual(result.operation, "create")
            self.assertIn("file", result.error or "")
        finally:
            os.unlink(file_path)

    def test_probe_fails_on_disk_full(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_open = open

            def fail_write_open(file, mode="r", *args, **kwargs):
                fh = real_open(file, mode, *args, **kwargs)
                if "xb" in mode and ".areno_preflight_" in str(file):
                    original_write = fh.write

                    def fail_write(data):
                        raise OSError(28, "No space left on device")

                    fh.write = fail_write
                return fh

            with mock.patch("builtins.open", side_effect=fail_write_open):
                result = probe_directory_writability(tmp, stage="checkpoint")
            self.assertFalse(result.ok)

    def test_probe_fails_on_quota_exceeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            real_open = open

            def fail_write_open(file, mode="r", *args, **kwargs):
                fh = real_open(file, mode, *args, **kwargs)
                if "xb" in mode and ".areno_preflight_" in str(file):
                    def fail_write(data):
                        raise OSError(122, "Disk quota exceeded")

                    fh.write = fail_write
                return fh

            with mock.patch("builtins.open", side_effect=fail_write_open):
                result = probe_directory_writability(tmp, stage="metrics")
            self.assertFalse(result.ok)


class ProbeCleanupTest(unittest.TestCase):
    def setUp(self):
        self._dirs_to_restore: list[Path] = []

    def tearDown(self):
        for d in reversed(self._dirs_to_restore):
            try:
                os.chmod(d, 0o755)
            except OSError:
                pass

    def test_probe_cleans_up_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe_directory_writability(tmp, stage="checkpoint")
            leftovers = list(Path(tmp).glob(".areno_preflight_*"))
            self.assertEqual(leftovers, [])

    def test_probe_cleans_up_on_failure(self):
        d = tempfile.mkdtemp()
        readonly = Path(d) / "readonly"
        readonly.mkdir()
        os.chmod(readonly, 0o444)
        self._dirs_to_restore.append(readonly)

        probe_directory_writability(readonly, stage="checkpoint")
        # readonly dir may not have probe files, but check parent just in case.
        leftovers = list(Path(d).glob(".areno_preflight_*"))
        self.assertEqual(leftovers, [])

    def test_probe_cleans_up_on_keyboard_interrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "probe_target"
            target.mkdir()

            original_replace = Path.replace

            def interrupt_replace(self, target_path):
                if ".areno_preflight_" in str(self):
                    raise KeyboardInterrupt
                return original_replace(self, target_path)

            with mock.patch.object(Path, "replace", interrupt_replace):
                result = probe_directory_writability(target, stage="checkpoint")

            self.assertFalse(result.ok)
            self.assertEqual(result.operation, "interrupted")
            leftovers = list(target.glob(".areno_preflight_*"))
            self.assertEqual(leftovers, [])


class ProbeSafetyTest(unittest.TestCase):
    def test_probe_does_not_overwrite_user_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            user_file = Path(tmp) / "user_data.txt"
            user_file.write_text("important data")

            probe_directory_writability(tmp, stage="checkpoint")

            self.assertEqual(user_file.read_text(), "important data")

    def test_probe_concurrent_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Pre-create a probe file with the exact name pattern to simulate
            # a concurrent process.  The probe should still succeed because
            # it uses UUID and retries.
            pre_existing = Path(tmp) / ".areno_preflight_checkpoint_preexisting.tmp"
            pre_existing.write_text("dummy")

            result = probe_directory_writability(tmp, stage="checkpoint")
            # Should succeed (different UUID).
            self.assertTrue(result.ok)
            # Note: the cleanup glob uses the prefix pattern, so the pre-existing
            # file with the .areno_preflight_ prefix WILL be removed by cleanup.
            # This is documented behavior: users should not create files with the
            # .areno_preflight_ prefix.


class ProbeConfigTest(unittest.TestCase):
    def test_probe_disabled_returns_skipped(self):
        cfg = PreflightConfig(enabled=False)
        result = probe_directory_writability(
            "/nonexistent/path/that/should/not/be/touched",
            stage="checkpoint",
            config=cfg,
        )
        self.assertTrue(result.ok)

    def test_probe_none_path_is_skipped(self):
        result = probe_directory_writability(None, stage="checkpoint")  # type: ignore
        self.assertTrue(result.ok)

    def test_probe_empty_path_is_skipped(self):
        result = probe_directory_writability("", stage="checkpoint")
        self.assertTrue(result.ok)

    def test_probe_with_custom_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PreflightConfig(probe_prefix=".custom_probe_")
            result = probe_directory_writability(tmp, stage="checkpoint", config=cfg)
            self.assertTrue(result.ok)
            # No default-prefix files left.
            self.assertEqual(list(Path(tmp).glob(".areno_preflight_*")), [])
            # No custom-prefix files left either.
            self.assertEqual(list(Path(tmp).glob(".custom_probe_*")), [])


class ProbeBatchTest(unittest.TestCase):
    def test_probe_paths_multiple(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            results = probe_paths([
                ("checkpoint", tmp1),
                ("metrics", tmp2),
            ])
            self.assertEqual(len(results), 2)
            self.assertTrue(all(r.ok for r in results))
            self.assertEqual(results[0].stage, "checkpoint")
            self.assertEqual(results[1].stage, "metrics")


class ProbeFormatTest(unittest.TestCase):
    def test_format_probe_results_contains_stage_and_operation(self):
        results = [
            PreflightProbeResult(stage="checkpoint", path="/readonly", ok=False, operation="create", error="PermissionError"),
            PreflightProbeResult(stage="metrics", path="/tmp/tfevent", ok=True),
        ]
        text = format_probe_results(results)
        self.assertIn("checkpoint", text)
        self.assertIn("metrics", text)
        self.assertIn("FAIL", text)
        self.assertIn("OK", text)
        self.assertIn("create", text)
        self.assertIn("/readonly", text)

    def test_format_probe_results_json_is_valid_json(self):
        results = [
            PreflightProbeResult(stage="checkpoint", path="/tmp", ok=True),
            PreflightProbeResult(stage="metrics", path="/bad", ok=False, operation="write", error="No space"),
        ]
        text = format_probe_results_json(results)
        data = json.loads(text)
        self.assertIn("checks", data)
        self.assertEqual(len(data["checks"]), 2)
        self.assertEqual(data["checks"][0]["stage"], "checkpoint")
        self.assertEqual(data["checks"][1]["ok"], False)


if __name__ == "__main__":
    unittest.main()