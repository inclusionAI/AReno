"""Public surface of the areno training SDK.

This module re-exports the user-facing types so callers can write
``import areno.api`` and reach `Trainer`, the typed backend configs, the
sampling/rollout/training schemas, and the bundled loss functions without
having to know the internal package layout.
"""

from areno.api.trainer import Trainer
from areno.api.algorithms import AlgorithmSpec, get_algorithm, list_algorithms, register_algorithm
from areno.api.config import ArenoConfig
from areno.api.data import PromptBatch, PromptItem
from areno.api.agentic import (
    AgentBatch,
    AgentItem,
    AgentTrainBatch,
    AgentTrajectory,
    AgentTrajectoryTurn,
    LossMaskPolicy,
    RewardEvent,
    RewardRecord,
    RolloutSession,
)
from areno.api.models import (
    SamplingParams,
    RolloutSequence,
    RolloutResult,
    TrainSequence,
    BackendType,
)
from areno.api.loss_fns import dpo_loss_fn, gspo_loss_fn, grpo_loss_fn, ppo_loss_fn, sft_loss_fn

# Friendly aliases mirroring the BackendType enum members; `DefaultBackend`
# documents the fallback used when callers do not pass `backend_type=`.
Areno = BackendType.Areno
DefaultBackend = BackendType.Areno

__all__ = [
    "Trainer",
    "AlgorithmSpec",
    "ArenoConfig",
    "PromptBatch",
    "PromptItem",
    "AgentBatch",
    "AgentItem",
    "AgentTrainBatch",
    "AgentTrajectory",
    "AgentTrajectoryTurn",
    "LossMaskPolicy",
    "RewardEvent",
    "RewardRecord",
    "RolloutSession",
    "SamplingParams",
    "RolloutResult",
    "RolloutSequence",
    "TrainSequence",
    "Areno",
    "DefaultBackend",
    "get_algorithm",
    "list_algorithms",
    "register_algorithm",
    "dpo_loss_fn",
    "gspo_loss_fn",
    "grpo_loss_fn",
    "ppo_loss_fn",
    "sft_loss_fn",
]
