from __future__ import annotations

import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest


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
            "areno.api.backend.cuda",
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


def test_cuda_worker_configures_torch_runtime_before_model_build(monkeypatch):
    """Spawned workers must restore Dynamo limits before model construction or compilation."""

    from areno.engine import worker as worker_mod

    events = []

    class ModelBuildReached(Exception):
        pass

    def build_model(*args, **kwargs):
        del args, kwargs
        events.append("build_model")
        raise ModelBuildReached

    monkeypatch.setattr(worker_mod, "_configure_torch_runtime", lambda: events.append("configure_torch"))
    monkeypatch.setattr(worker_mod, "get_tp_context", lambda: SimpleNamespace(device="cuda:0"))
    monkeypatch.setattr(worker_mod, "build_model_on_device", build_model)

    with pytest.raises(ModelBuildReached):
        worker_mod.ArenoWorker(SimpleNamespace())

    assert events == ["configure_torch", "build_model"]
