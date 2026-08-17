"""Public configuration for the Qwen3 dense and MoE LoRA runtime."""

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


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Supported PEFT-compatible LoRA subset for Qwen3 dense and MoE models.

    When ``adapter_path`` is set, its standard PEFT metadata is authoritative
    for rank, alpha, dropout, and targets.
    """

    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = QWEN3_DENSE_TARGETS
    adapter_path: str | None = None

    def __post_init__(self) -> None:
        if self.adapter_path is not None:
            adapter_config = _read_adapter_config(self.adapter_path)
            object.__setattr__(self, "rank", int(adapter_config["r"]))
            object.__setattr__(self, "alpha", float(adapter_config["lora_alpha"]))
            object.__setattr__(self, "dropout", float(adapter_config.get("lora_dropout", 0.0)))
            object.__setattr__(self, "target_modules", tuple(adapter_config["target_modules"]))
        object.__setattr__(self, "target_modules", tuple(self.target_modules))
        if self.rank < 1:
            raise ValueError("lora rank must be >= 1")
        if self.alpha <= 0:
            raise ValueError("lora alpha must be > 0")
        if self.dropout != 0.0:
            raise ValueError("native Qwen3 LoRA currently requires dropout=0")
        requested = set(self.target_modules)
        supported = set(QWEN3_DENSE_TARGETS)
        if not requested or not requested <= supported:
            raise ValueError(f"target_modules must be a non-empty subset of {QWEN3_DENSE_TARGETS}")

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
    for option in ("use_rslora", "use_dora", "rank_pattern", "alpha_pattern", "modules_to_save"):
        if adapter_config.get(option):
            unsupported.append(option)
    if unsupported:
        raise ValueError(f"unsupported PEFT LoRA options: {', '.join(unsupported)}")
    return adapter_config
