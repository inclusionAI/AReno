"""QLoRA policy and checkpoint identity contracts, without an MLX runtime."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from areno.adapters import LoraConfig
from areno.api.backend.mlx.quantization import logical_weight_shape, validate_adapter_base, validate_quantized_base
from areno.api.config import CudaConfig, MlxConfig


def _base(tmp_path, **parameters):
    params = {"bits": 4, "group_size": 64, "mode": "affine", **parameters}
    module = SimpleNamespace(**params, weight=np.zeros((3, 64, 8), dtype=np.uint32), scales=np.ones((3, 64, 1)))
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0.mlp.switch_mlp.up_proj", module)])
    config = {"model_type": "bailing_hybrid", "quantization": params}
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors").write_bytes(b"base weights")
    return model, config, module


def _adapter(path, metadata):
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj"],
            }
        )
    )
    (path / "areno_quantization_config.json").write_text(json.dumps(metadata))


def test_mlx_quantized_base_identity_and_backend_dispatch(tmp_path):
    model, config, module = _base(tmp_path)
    assert logical_weight_shape(module) == (3, 64, 64)
    metadata = validate_quantized_base(model, config, LoraConfig(qlora=True), str(tmp_path))
    adapter = tmp_path / "adapter"
    _adapter(adapter, metadata)
    lora = LoraConfig(adapter_path=str(adapter))
    assert lora.qlora and MlxConfig(lora=lora).lora is lora
    assert validate_quantized_base(model, config, lora, str(tmp_path)) == metadata
    with pytest.raises(ValueError, match="CUDA requires NF4"):
        CudaConfig(lora=lora)
    (tmp_path / "model.safetensors").write_bytes(b"different weights, same model dimensions")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_quantized_base(model, config, lora, str(tmp_path))


@pytest.mark.parametrize("params", [{"bits": 8}, {"group_size": 32}, {"mode": "mxfp4"}])
def test_mlx_qlora_rejects_other_quantization_formats(tmp_path, params):
    model, config, _ = _base(tmp_path, **params)
    with pytest.raises(ValueError, match="4-bit affine"):
        validate_quantized_base(model, config, LoraConfig(qlora=True), str(tmp_path))


def test_mlx_qlora_rejects_protected_layers_and_false_metadata(tmp_path):
    model, config, module = _base(tmp_path)
    model.quant_predicate = lambda path, module: False
    with pytest.raises(ValueError, match="model-protected"):
        validate_quantized_base(model, config, LoraConfig(qlora=True), str(tmp_path))
    del model.quant_predicate
    module.bits = 8
    with pytest.raises(ValueError, match="4-bit affine"):
        validate_quantized_base(model, config, LoraConfig(qlora=True), str(tmp_path))


def test_mlx_qlora_requires_explicit_quantized_training(tmp_path):
    model, config, _ = _base(tmp_path)
    for lora in (None, LoraConfig()):
        with pytest.raises(ValueError, match="requires LoRA"):
            validate_quantized_base(model, config, lora, str(tmp_path))
    with pytest.raises(ValueError, match="preconverted"):
        validate_quantized_base(model, {"model_type": "qwen3"}, LoraConfig(qlora=True), str(tmp_path))
    with pytest.raises(ValueError, match="fingerprint"):
        validate_adapter_base(tmp_path, {"format": "mlx-affine-v1"})


def test_mlx_rejects_cuda_nf4_adapter(tmp_path):
    adapter = tmp_path / "adapter"
    _adapter(adapter, {"format": "nf4-dq-v1", "block_size": 64, "scale_block_size": 256})
    lora = LoraConfig(adapter_path=str(adapter))
    assert CudaConfig(lora=lora).lora.qlora
    with pytest.raises(ValueError, match="CUDA NF4"):
        MlxConfig(lora=lora)
