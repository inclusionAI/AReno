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
from typing import Any, Protocol, runtime_checkable

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
    # Vocab-sharded logits of the model's MTP layer(s), predicting token t+2
    # at position t. Only populated during training when the model has MTP
    # layers and `TrainMeta.mtp_enabled` is set.
    mtp_logits_shard: torch.Tensor | None = None


@runtime_checkable
class SpeculativeDraftModel(Protocol):
    """Model-side contract for MTP speculative decoding in the rollout engine.

    The engine verifies ``k`` drafts per sequence with one decode forward at
    ``InferMeta.tokens_per_seq = k + 1`` and then calls these hooks. Models
    without an MTP head simply do not implement the protocol.
    """

    def enable_mtp_draft(self, *, max_rows: int, tokens_per_seq: int) -> None:
        """Give the MTP layers inference caches and size the shared verify-state buffers.

        Call before ``allocate_kv_caches``. ``max_rows`` bounds the active rows
        of one verify forward and ``tokens_per_seq`` is ``k + 1``.
        """

    def mtp_draft_forward(
        self,
        *,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        infer_meta: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the MTP layer as the draft model; return (logits_shard, hidden_states)."""

    def commit_speculative_state(self, committed: torch.Tensor, *, infer_meta: Any) -> None:
        """Keep recurrent state after the first ``committed[row]`` fed tokens of the verify run with ``infer_meta``."""


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

    def build_policy_plan(self, model: nn.Module):
        """Return live canonical weight-layout tasks for direct policy sync."""

        raise NotImplementedError(f"{type(self).__name__} does not provide a policy synchronization plan")
