from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

from torch import nn

from areno.engine.config import ModelConfig
from areno.models import registry


class FakeAdapter:
    """Minimal adapter used to test registry dispatch without real models."""

    name = "fake"

    def __init__(self):
        self.loaded = []
        self.saved = []

    def match_hf_config(self, hf_config):
        return hf_config.get("model_type") == "fake"

    def config_from_hf(self, hf_config):
        return ModelConfig(
            model_type=self.name,
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=8,
        )

    def build(self, config):
        return nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def load_weights(self, model, model_path):
        self.loaded.append((model, str(model_path)))

    def save_weights(self, model, output_path, source_path):
        self.saved.append((model, str(output_path), None if source_path is None else str(source_path)))
        return str(output_path)


class RegistryTest(unittest.TestCase):
    """Registry tests stay model-free by replacing plugin loading with fakes."""

    def setUp(self):
        # Isolate global adapter state so these tests do not import model plugins.
        self.old_adapters = dict(registry._ADAPTERS)
        self.old_plugins_loaded = registry._PLUGINS_LOADED
        registry._ADAPTERS.clear()
        registry._PLUGINS_LOADED = True

    def tearDown(self):
        registry._ADAPTERS.clear()
        registry._ADAPTERS.update(self.old_adapters)
        registry._PLUGINS_LOADED = self.old_plugins_loaded

    def test_register_adapter_rejects_duplicate_names(self):
        """Duplicate adapter names would make model_type dispatch ambiguous."""
        adapter = FakeAdapter()
        registry.register_adapter(adapter)

        with self.assertRaisesRegex(ValueError, "duplicate model adapter"):
            registry.register_adapter(FakeAdapter())

    def test_config_from_hf_uses_matching_adapter(self):
        """HF config matching should be delegated to the registered adapter."""
        adapter = FakeAdapter()
        registry.register_adapter(adapter)

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"model_type": "fake"}), encoding="utf-8")
            config = registry.config_from_hf(tmp)

        self.assertEqual(config.model_type, "fake")
        self.assertEqual(config.hidden_size, 8)

    def test_unknown_hf_config_lists_registered_adapters(self):
        """Unknown HF configs should produce a useful error for debugging."""
        registry.register_adapter(FakeAdapter())

        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"model_type": "other"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "registered: fake"):
                registry.adapter_from_hf(tmp)

    def test_build_load_and_save_dispatch_to_adapter(self):
        """Build/load/save must route through the adapter selected by model_type."""
        adapter = FakeAdapter()
        registry.register_adapter(adapter)
        config = ModelConfig(
            model_type="fake",
            hidden_size=8,
            intermediate_size=8,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=8,
        )
        model = registry.build_model(config)

        registry.load_model_weights(model, config, "/ckpt")
        saved_path = registry.save_model_weights(model, config, "/out", "/src")

        self.assertIs(adapter.loaded[0][0], model)
        self.assertEqual(adapter.loaded[0][1], "/ckpt")
        self.assertIs(adapter.saved[0][0], model)
        self.assertEqual(adapter.saved[0][1:], ("/out", "/src"))
        self.assertEqual(saved_path, "/out")

    def test_save_unwraps_nested_compiled_modules(self):
        """Saving should strip torch.compile-style wrappers before serialization."""
        adapter = FakeAdapter()
        registry.register_adapter(adapter)
        config = ModelConfig(
            model_type="fake",
            hidden_size=8,
            intermediate_size=8,
            num_attention_heads=1,
            num_key_value_heads=1,
            vocab_size=8,
        )
        raw = nn.Linear(2, 2)
        wrapped_once = type("Wrapped", (), {"_orig_mod": raw})()
        wrapped_twice = type("WrappedAgain", (), {"_orig_mod": wrapped_once})()

        registry.save_model_weights(wrapped_twice, config, "/out", None)

        self.assertIs(adapter.saved[0][0], raw)

    def test_load_model_plugins_is_thread_safe(self):
        """Concurrent first-use plugin loading should register adapters once."""
        import areno

        old_module = sys.modules.get("areno.models")
        old_attr = getattr(areno, "models", None)
        had_attr = hasattr(areno, "models")
        fake_module = types.ModuleType("areno.models")
        register_calls = []

        def register_models():
            time.sleep(0.01)
            register_calls.append(1)
            registry.register_adapter(FakeAdapter())

        fake_module.register_models = register_models
        sys.modules["areno.models"] = fake_module
        areno.models = fake_module
        registry._ADAPTERS.clear()
        registry._PLUGINS_LOADED = False
        errors = []

        def load_once():
            try:
                registry.load_model_plugins()
            except Exception as exc:  # pragma: no cover - assertion reports it.
                errors.append(exc)

        threads = [threading.Thread(target=load_once), threading.Thread(target=load_once)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        if old_module is None:
            sys.modules.pop("areno.models", None)
        else:
            sys.modules["areno.models"] = old_module
        if had_attr:
            areno.models = old_attr
        else:
            delattr(areno, "models")

        self.assertEqual(errors, [])
        self.assertEqual(len(register_calls), 1)
        self.assertEqual(sorted(registry._ADAPTERS), ["fake"])

    def test_bundled_model_registration_is_lazy_and_idempotent(self):
        """Selecting one bundled group must not import unrelated runtimes."""
        code = """
import sys
import types

from areno.models import register_models
from areno.models.registry import _adapter

# Keep this registry unit test independent of CUDA/Triton/FLA. The real Qwen
# implementation is exercised by the GPU serving smoke; here the fake module
# makes the selected import boundary directly observable.
qwen = types.ModuleType("areno.models.qwen3")
qwen.Qwen3Adapter = type("Qwen3Adapter", (), {"name": "qwen3"})
qwen.Qwen3MoeAdapter = type("Qwen3MoeAdapter", (), {"name": "qwen3_moe"})
sys.modules["areno.models.qwen3"] = qwen

assert register_models("qwen3")
assert register_models("qwen3")
assert _adapter("qwen3").name == "qwen3"
assert "areno.models.qwen3" in sys.modules
assert "areno.models.bailing" not in sys.modules
assert "areno.models.bailing_v3" not in sys.modules
assert not any(name == "fla" or name.startswith("fla.") for name in sys.modules)

bailing_v3 = types.ModuleType("areno.models.bailing_v3")
bailing_v3.BailingMoeV3Adapter = type(
    "BailingMoeV3Adapter", (), {"name": "bailing_moe_v3"}
)
sys.modules["areno.models.bailing_v3"] = bailing_v3
assert register_models("bailing_moe_v3")
assert _adapter("bailing_moe_v3").name == "bailing_moe_v3"
assert "areno.models.bailing_v3" in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
