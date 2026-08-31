"""Public surface of the areno training SDK.

This module re-exports the user-facing types so callers can write
``import areno.api`` and reach `Trainer`, the typed backend configs, the
sampling/rollout/training schemas, and the bundled loss functions without
having to know the internal package layout.
"""

from areno.api.agentic import (
    AgentBatch,
    AgentItem,
    AgentTrainBatch,
    AgentTrajectory,
    AgentTrajectoryTurn,
    LossMaskPolicy,
    RolloutSession,
)
from areno.api.algorithms import (
    AlgorithmSpec,
    dpo_loss_fn,
    get_algorithm,
    grpo_loss_fn,
    gspo_loss_fn,
    list_algorithms,
    ppo_loss_fn,
    register_algorithm,
    sft_loss_fn,
)
from areno.api.config import CudaConfig, LoraConfig, MlxConfig, default_backend_type
from areno.api.data import PromptBatch, PromptItem
from areno.api.models import (
    BackendType,
    RolloutResult,
    RolloutSequence,
    SamplingParams,
    TrainSequence,
)
from areno.api.rewards import RewardEvent, RewardRecord
from areno.api.trainer import Trainer

# Friendly aliases mirroring the BackendType enum members. The default is
# selected from the host platform without importing either backend.
CUDA = BackendType.CUDA
MLX = BackendType.MLX
DefaultBackend = default_backend_type()

__all__ = [
    "Trainer",
    "AlgorithmSpec",
    "CudaConfig",
    "MlxConfig",
    "LoraConfig",
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
    "CUDA",
    "MLX",
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
