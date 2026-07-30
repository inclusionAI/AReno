"""CPU tests for atomic file-writing helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from areno.cli.atomic_io import atomic_write_bytes, atomic_write_json, atomic_write_text


class AtomicWriteTextTest(unittest.TestCase):
    def test_atomic_write_text_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.txt"
            atomic_write_text(p, "hello")
            self.assertEqual(p.read_text(), "hello")
            # No temp file left behind.
            self.assertFalse((Path(str(p) + ".tmp")).exists())

    def test_atomic_write_text_overwrites_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.txt"
            p.write_text("old")
            atomic_write_text(p, "new")
            self.assertEqual(p.read_text(), "new")

    def test_atomic_write_text_cleans_temp_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.txt"
            tmp_path = Path(str(p) + ".tmp")

            original_write_text = Path.write_text

            def fail_write_text(self, data, **kwargs):
                if self == tmp_path:
                    raise OSError("simulated write failure")
                return original_write_text(self, data, **kwargs)

            with mock.patch.object(Path, "write_text", fail_write_text):
                with self.assertRaises(OSError):
                    atomic_write_text(p, "hello")

            self.assertFalse(tmp_path.exists())

    def test_atomic_write_text_cleans_temp_on_replace_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.txt"
            tmp_path = Path(str(p) + ".tmp")

            original_replace = Path.replace

            def fail_replace(self, target):
                if self == tmp_path:
                    raise OSError("simulated rename failure")
                return original_replace(self, target)

            with mock.patch.object(Path, "replace", fail_replace):
                with self.assertRaises(OSError):
                    atomic_write_text(p, "hello")

            # Temp file should be cleaned up.
            self.assertFalse(tmp_path.exists())


class AtomicWriteBytesTest(unittest.TestCase):
    def test_atomic_write_bytes_creates_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.bin"
            payload = b"\x00\x01\x02\xff"
            atomic_write_bytes(p, payload)
            self.assertEqual(p.read_bytes(), payload)
            self.assertFalse((Path(str(p) + ".tmp")).exists())


class AtomicWriteJsonTest(unittest.TestCase):
    def test_atomic_write_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            data = {"step": 10, "loss": 0.5, "items": [1, 2, 3]}
            atomic_write_json(p, data, sort_keys=True)
            loaded = json.loads(p.read_text())
            self.assertEqual(loaded, data)

    def test_atomic_write_json_ensure_ascii(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "data.json"
            atomic_write_json(p, {"name": "test"}, ensure_ascii=True)
            text = p.read_text()
            self.assertIn("test", text)


class AtomicWriteNoPartialFileTest(unittest.TestCase):
    def test_atomic_write_no_partial_file_on_failure(self):
        """If the write fails, the target file must not be created or modified."""

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.txt"
            p.write_text("original")

            tmp_path = Path(str(p) + ".tmp")
            original_write_text = Path.write_text

            def fail_write_text(self, data, **kwargs):
                if self == tmp_path:
                    raise OSError("boom")
                return original_write_text(self, data, **kwargs)

            with mock.patch.object(Path, "write_text", fail_write_text):
                with self.assertRaises(OSError):
                    atomic_write_text(p, "new")

            # Original content preserved.
            self.assertEqual(p.read_text(), "original")
            self.assertFalse(tmp_path.exists())


if __name__ == "__main__":
    unittest.main()