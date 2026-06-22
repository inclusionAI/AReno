from __future__ import annotations

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_setup_module() -> dict:
    setup_path = Path(__file__).resolve().parents[1] / "setup.py"
    with (
        patch.object(sys, "argv", ["setup.py", "egg_info"]),
        patch("setuptools.setup"),
    ):
        return runpy.run_path(str(setup_path))


class SetupGuardrailsTest(unittest.TestCase):
    def test_missing_torch_error_is_actionable(self):
        setup_mod = _load_setup_module()

        with (
            patch.dict(sys.modules, {"torch": None}),
            self.assertRaises(RuntimeError) as exc,
        ):
            setup_mod["_require_torch"]()
        message = str(exc.exception)
        self.assertIn("PyTorch is not installed", message)
        self.assertIn("CUDA-enabled PyTorch", message)

    def test_cpu_only_torch_error_is_actionable(self):
        setup_mod = _load_setup_module()
        fake_torch = types.SimpleNamespace(version=types.SimpleNamespace(cuda=None))

        with self.assertRaises(RuntimeError) as exc:
            setup_mod["_check_cuda_torch"](fake_torch)
        message = str(exc.exception)
        self.assertIn("CPU-only", message)
        self.assertIn("CUDA-enabled PyTorch", message)

    def test_unsupported_platform_error_mentions_metadata_install(self):
        setup_mod = _load_setup_module()

        with (
            patch.object(setup_mod["platform"], "system", return_value="Darwin"),
            patch.object(setup_mod["platform"], "machine", return_value="arm64"),
            self.assertRaises(RuntimeError) as exc,
        ):
            setup_mod["_check_supported_build_platform"]()
        message = str(exc.exception)
        self.assertIn("not supported", message)
        self.assertIn("ARENO_BUILD_EXT=0", message)


if __name__ == "__main__":
    unittest.main()
