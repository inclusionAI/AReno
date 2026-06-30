from __future__ import annotations

import sys
import types

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
    old_module = sys.modules.get("areno.models")
    old_attr = getattr(areno, "models", None)
    had_attr = hasattr(areno, "models")
    old_adapters = dict(registry._ADAPTERS)
    old_plugins_loaded = registry._PLUGINS_LOADED
    fake_module = types.ModuleType("areno.models")
    fake_module.register_models = lambda: registry.register_adapter(SmokeAdapter())

    try:
        sys.modules["areno.models"] = fake_module
        areno.models = fake_module
        registry._ADAPTERS.clear()
        registry._PLUGINS_LOADED = False

        registry.load_model_plugins()

        assert sorted(registry._ADAPTERS) == ["smoke"]
    finally:
        if old_module is None:
            sys.modules.pop("areno.models", None)
        else:
            sys.modules["areno.models"] = old_module
        if had_attr:
            areno.models = old_attr
        else:
            delattr(areno, "models")
        registry._ADAPTERS.clear()
        registry._ADAPTERS.update(old_adapters)
        registry._PLUGINS_LOADED = old_plugins_loaded
