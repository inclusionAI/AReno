"""Public surface of the areno training SDK.

This module re-exports the user-facing types so callers can write
``import areno.api`` and reach `Trainer`, the typed backend configs, the
sampling/rollout/training schemas, and the bundled loss functions without
having to know the internal package layout.
"""

from importlib import import_module
from typing import Any

from areno.api.config import CudaConfig, LoraConfig, MlxConfig, default_backend_type
from areno.api.models import BackendType

# Friendly aliases mirroring the BackendType enum members. The default is
# selected from the host platform without importing either backend.
CUDA = BackendType.CUDA
MLX = BackendType.MLX
DefaultBackend = default_backend_type()

_LAZY_EXPORTS = {
    "AgentBatch": ("areno.api.agentic", "AgentBatch"),
    "AgentItem": ("areno.api.agentic", "AgentItem"),
    "AgentTrainBatch": ("areno.api.agentic", "AgentTrainBatch"),
    "AgentTrajectory": ("areno.api.agentic", "AgentTrajectory"),
    "AgentTrajectoryTurn": ("areno.api.agentic", "AgentTrajectoryTurn"),
    "AlgorithmSpec": ("areno.api.algorithms", "AlgorithmSpec"),
    "LossMaskPolicy": ("areno.api.agentic", "LossMaskPolicy"),
    "PromptBatch": ("areno.api.data", "PromptBatch"),
    "PromptItem": ("areno.api.data", "PromptItem"),
    "RewardEvent": ("areno.api.rewards", "RewardEvent"),
    "RewardRecord": ("areno.api.rewards", "RewardRecord"),
    "RolloutResult": ("areno.api.models", "RolloutResult"),
    "RolloutSequence": ("areno.api.models", "RolloutSequence"),
    "RolloutSession": ("areno.api.agentic", "RolloutSession"),
    "SamplingParams": ("areno.api.models", "SamplingParams"),
    "Trainer": ("areno.api.trainer", "Trainer"),
    "TrainSequence": ("areno.api.models", "TrainSequence"),
    "dpo_loss_fn": ("areno.api.algorithms", "dpo_loss_fn"),
    "get_algorithm": ("areno.api.algorithms", "get_algorithm"),
    "grpo_loss_fn": ("areno.api.algorithms", "grpo_loss_fn"),
    "gspo_loss_fn": ("areno.api.algorithms", "gspo_loss_fn"),
    "list_algorithms": ("areno.api.algorithms", "list_algorithms"),
    "ppo_loss_fn": ("areno.api.algorithms", "ppo_loss_fn"),
    "register_algorithm": ("areno.api.algorithms", "register_algorithm"),
    "sft_loss_fn": ("areno.api.algorithms", "sft_loss_fn"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = target
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value

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
