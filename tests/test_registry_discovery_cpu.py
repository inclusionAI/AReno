from __future__ import annotations

import sys
import types
from unittest.mock import patch

import areno
from areno.api.algorithms import list_algorithms
from areno.models import registry


class SmokeAdapter:
    """Minimal adapter for registry-discovery smoke coverage."""

    name = "smoke"

    def match_hf_config(self, hf_config):
        return hf_config.get("model_type") == self.name


def test_algorithm_and_model_registry_discovery_smoke():
    """Current registries expose algorithms and lazily loaded model adapters."""

    algorithms = list_algorithms(include_experimental=False)

    assert {"sft", "gspo", "ppo"}.issubset(algorithms)
    assert algorithms["sft"].requires_rollout is False
    assert algorithms["gspo"].requires_rollout is True

    # Use a tiny plugin pack instead of importing bundled model modules here:
    # some adapters depend on optional CUDA/FLA packages that CPU smoke tests
    # should not require.
    fake_module = types.ModuleType("areno.models")
    fake_module.register_models = lambda: registry.register_adapter(SmokeAdapter())

    with (
        patch.dict(sys.modules, {"areno.models": fake_module}),
        patch.object(areno, "models", fake_module, create=True),
        patch.dict(registry._ADAPTERS, {}, clear=True),
        patch.object(registry, "_PLUGINS_LOADED", False),
    ):
        registry.load_model_plugins()

        assert sorted(registry._ADAPTERS) == ["smoke"]
