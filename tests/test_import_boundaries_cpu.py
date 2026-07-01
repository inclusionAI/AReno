from __future__ import annotations

import subprocess
import sys
import textwrap


def test_public_api_imports_do_not_load_engine_heavy_modules():
    """Public API imports stay on the lazy side of the engine/backend boundary."""

    script = textwrap.dedent(
        """
        import importlib
        import sys

        for module_name in [
            "areno",
            "areno.api",
            "areno.api.trainer",
            "areno.api.backend",
            "areno.api.backend.base",
        ]:
            importlib.import_module(module_name)

        heavy_modules = [
            "areno.api.backend.areno",
            "areno.engine.api",
            "areno.engine.inference",
            "areno.engine.worker",
        ]
        for name in heavy_modules:
            assert name not in sys.modules, f"{name} was unexpectedly loaded"
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
    )
