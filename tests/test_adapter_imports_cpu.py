"""Import-boundary checks for the backend-neutral SDK and adapter configuration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_sdk_imports_do_not_load_torch_or_native_lora() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script = r"""
import importlib.abc
import sys


class BlockHeavyImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if (
            fullname == "torch"
            or fullname.startswith("torch.")
            or fullname == "mlx"
            or fullname.startswith("mlx.")
            or fullname == "areno.adapters.lora"
        ):
            raise AssertionError(f"unexpected heavy import: {fullname}")
        return None


sys.meta_path.insert(0, BlockHeavyImports())

from areno import Trainer
from areno.adapters import LoraConfig as PublicLoraConfig
from areno.adapters.config import LoraConfig
from areno.api import MLX, Trainer as ApiTrainer
from areno.api.agentic import RolloutSession, _routing_to_cpu_tensor
from areno.api.backend.mlx.lora import MlxLoraState
from areno.api.config import MlxConfig
from areno.api.multimodal import image_token_counts_from_features
from areno.api.trainer_config import TrainerConfig

assert PublicLoraConfig is LoraConfig
assert MlxLoraState.__module__ == "areno.api.backend.mlx.lora"
assert MlxConfig(lora=LoraConfig()).lora is not None
assert TrainerConfig(algo="sft", backend="mlx", ckpt="model", dataset_path="data").backend == "mlx"
assert Trainer is ApiTrainer
trainer = Trainer(world_size=1, model_path="unused", backend_type=MLX)
session = RolloutSession(None, sampling_params=None)
assert session.max_running_prompts == 1
assert _routing_to_cpu_tensor(None) is None
assert image_token_counts_from_features(None) == []
assert image_token_counts_from_features({"processor_expanded_image_tokens": True}) == []
trainer.close()
assert "torch" not in sys.modules
assert "mlx" not in sys.modules
assert "areno.adapters.lora" not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_native_adapter_public_exports_remain_available() -> None:
    from areno.adapters import AdapterRegistry, LoraSlot, initialize_lora

    assert AdapterRegistry.__module__ == "areno.adapters.lora"
    assert LoraSlot.__module__ == "areno.adapters.lora"
    assert initialize_lora.__module__ == "areno.adapters.lora"
