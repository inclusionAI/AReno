"""Model adapter protocol and shared output dataclass.

Each supported architecture is plugged into the runtime via a `ModelAdapter`
that knows how to (a) recognize its HuggingFace config, (b) translate that
config into the framework's `ModelConfig`, (c) instantiate the nn.Module,
and (d) load and save checkpoint weights in HF-compatible layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from areno.engine.config import ModelConfig


@dataclass(slots=True)
class CausalLMOutput:
    """Common forward output of causal LM models.

    ``logits_shard`` is the per-rank vocab-sharded logits tensor (the
    softmax across all shards is computed by the loss code); ``hidden_states``
    is the last decoder layer output; ``values`` is an optional scalar head
    used by RL training.
    """

    logits_shard: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None
    values: torch.Tensor | None = None


class ModelAdapter(ABC):
    """Glue between HuggingFace checkpoints and areno's nn.Modules."""

    name: str

    @abstractmethod
    def match_hf_config(self, hf_config: dict[str, Any]) -> bool:
        """Return True if this adapter handles ``hf_config`` (by model_type, etc.)."""

        raise NotImplementedError

    @abstractmethod
    def config_from_hf(self, hf_config: dict[str, Any]) -> ModelConfig:
        """Translate a raw HF config dict into the internal ``ModelConfig``."""

        raise NotImplementedError

    @abstractmethod
    def build(self, config: ModelConfig) -> nn.Module:
        """Instantiate the nn.Module described by ``config`` (no weights yet)."""

        raise NotImplementedError

    @abstractmethod
    def load_weights(self, model: nn.Module, model_path: str | Path) -> None:
        """Load HF-format weights from ``model_path`` into ``model`` in place."""

        raise NotImplementedError

    @abstractmethod
    def save_weights(self, model: nn.Module, output_path: str | Path, source_path: str | Path | None) -> str | None:
        """Save weights back to an HF-compatible checkpoint at ``output_path``."""

        raise NotImplementedError
