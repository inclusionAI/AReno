"""Public configuration for the native LoRA runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

QWEN3_DENSE_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

BAILING_V3_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "f_proj",
    "g_proj",
    "o_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "dense",
    "gate_proj",
    "up_proj",
    "down_proj",
)

QWEN35_TARGETS = (*QWEN3_DENSE_TARGETS, "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")

NATIVE_LORA_TARGETS = tuple(dict.fromkeys((*QWEN3_DENSE_TARGETS, *BAILING_V3_TARGETS, *QWEN35_TARGETS, "b_proj")))


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Supported PEFT-compatible subset for native LoRA model families.

    When ``adapter_path`` is set, its standard PEFT metadata is authoritative
    for rank, alpha, dropout, and targets.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = QWEN3_DENSE_TARGETS
    adapter_path: str | None = None
    qlora: bool = False

    def __post_init__(self) -> None:
        if self.adapter_path is not None:
            adapter_config = _read_adapter_config(self.adapter_path)
            object.__setattr__(self, "rank", int(adapter_config["r"]))
            object.__setattr__(self, "alpha", float(adapter_config["lora_alpha"]))
            object.__setattr__(self, "dropout", float(adapter_config.get("lora_dropout", 0.0)))
            object.__setattr__(self, "target_modules", tuple(adapter_config["target_modules"]))
            if read_adapter_quantization(self.adapter_path) is not None:
                object.__setattr__(self, "qlora", True)
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if self.rank < 1:
            raise ValueError("lora rank must be >= 1")
        if self.alpha <= 0:
            raise ValueError("lora alpha must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        requested = set(self.target_modules)
        supported = set(NATIVE_LORA_TARGETS)
        if not requested or not requested <= supported:
            raise ValueError(f"target_modules must be a non-empty subset of {NATIVE_LORA_TARGETS}")

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)


def _read_adapter_config(path: str) -> dict:
    adapter_config = json.loads((Path(path) / "adapter_config.json").read_text(encoding="utf-8"))
    if str(adapter_config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter_path must contain a PEFT LoRA artifact")
    unsupported = []
    if adapter_config.get("bias", "none") != "none" or bool(adapter_config.get("lora_bias", False)):
        unsupported.append("bias")
    if bool(adapter_config.get("fan_in_fan_out", False)):
        unsupported.append("fan_in_fan_out")
    for option in (
        "use_rslora",
        "use_dora",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "alora_invocation_tokens",
        "layer_replication",
        "trainable_token_indices",
        "target_parameters",
        "use_qalora",
    ):
        if adapter_config.get(option):
            unsupported.append(option)
    if unsupported:
        raise ValueError(f"unsupported PEFT LoRA options: {', '.join(unsupported)}")
    return adapter_config


def read_adapter_quantization(path: str | Path | None) -> dict | None:
    """Read backend-specific QLoRA metadata without importing a tensor runtime."""
    if path is None:
        return None
    metadata = Path(path) / "areno_quantization_config.json"
    if not metadata.exists():
        return None
    value = json.loads(metadata.read_text(encoding="utf-8"))
    if value == {"format": "nf4-dq-v1", "block_size": 64, "scale_block_size": 256}:
        return value
    if isinstance(value, dict) and value.get("format") == "mlx-affine-v1":
        if all(
            value.get(key)
            for key in ("model_type", "quantization", "layers", "base_config_sha256", "base_weights_sha256")
        ):
            return value
    raise ValueError("unsupported AReno QLoRA checkpoint quantization format")
