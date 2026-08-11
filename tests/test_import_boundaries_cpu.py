from __future__ import annotations

import subprocess
import sys
import textwrap


def test_public_api_imports_do_not_load_engine_heavy_modules():
    """Public API imports stay on the lazy side of the engine/backend boundary.

    Importing the SDK surface -- ``areno``, ``areno.api``, ``areno.api.trainer``,
    and the lazy seam modules (``areno.api.backend.base`` resolves concrete
    backends via ``__import__``; ``areno.api.tokenizer`` wraps the engine
    tokenizer loader) -- must not eagerly construct ``ArenoEngine`` or pull in a
    concrete backend / engine implementation module.

    Only the heavy modules that matter are asserted, so the test does not break
    when an unrelated dependency adds a new transitive import.
    """

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
            "areno.api.tokenizer",
        ]:
            importlib.import_module(module_name)

        heavy_modules = [
            "areno.api.backend.areno",
            "areno.engine.api",
            "areno.engine.inference",
            "areno.engine.worker",
            "areno.engine.data.tokenizer",
        ]
        for name in heavy_modules:
            assert name not in sys.modules, f"{name} was unexpectedly loaded"
        """
    )

    subprocess.run(
        [sys.executable, "-c", script],
        check=True,
    )
