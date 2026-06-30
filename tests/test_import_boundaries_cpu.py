from __future__ import annotations

import json
import subprocess
import sys
import textwrap


def test_public_api_imports_do_not_load_engine_heavy_modules():
    """Public API imports stay on the lazy side of the engine/backend boundary."""

    script = textwrap.dedent(
        """
        import importlib
        import json
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
        print(json.dumps({name: name in sys.modules for name in heavy_modules}, sort_keys=True))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    imported = json.loads(result.stdout.strip().splitlines()[-1])
    assert imported == {
        "areno.api.backend.areno": False,
        "areno.engine.api": False,
        "areno.engine.inference": False,
        "areno.engine.worker": False,
    }
