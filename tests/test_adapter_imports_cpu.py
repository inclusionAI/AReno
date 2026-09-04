"""Import-boundary checks for backend-neutral adapter configuration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_config_imports_do_not_load_torch_or_native_lora() -> None:
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

from areno.adapters import LoraConfig as PublicLoraConfig
from areno.adapters.config import LoraConfig
from areno.api.backend.mlx.lora import MlxLoraState
from areno.api.config import MlxConfig
from areno.api.trainer_config import TrainerConfig

assert PublicLoraConfig is LoraConfig
assert MlxLoraState.__module__ == "areno.api.backend.mlx.lora"
assert MlxConfig(lora=LoraConfig()).lora is not None
assert TrainerConfig(algo="sft", backend="mlx", ckpt="model", dataset_path="data").backend == "mlx"
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
