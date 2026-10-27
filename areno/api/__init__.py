"""Public surface of the areno training SDK.

This module re-exports the user-facing types so callers can write
``import areno.api`` and reach `Trainer`, the typed backend configs, the
sampling/rollout/training schemas, and the bundled loss functions without
having to know the internal package layout.
"""

from importlib import import_module
from typing import Any

from areno.api.config import CudaConfig, LoraConfig, MlxConfig, NpuConfig, default_backend_type
from areno.api.models import BackendType

# Friendly aliases mirroring the BackendType enum members. The default is
# selected from the host platform without importing either backend.
CUDA = BackendType.CUDA
MLX = BackendType.MLX
NPU = BackendType.NPU
DefaultBackend = default_backend_type()

_LAZY_EXPORTS = {
    "JevCudaConfig": ("areno.api.jev", "JevCudaConfig"),
    "JevMlxConfig": ("areno.api.jev", "JevMlxConfig"),
    "JevTrainerConfig": ("areno.api.jev", "JevTrainerConfig"),
    "DecisionScorer": ("areno.api.jev.scorer", "DecisionScorer"),
    "AgentBatch": ("areno.api.agentic", "AgentBatch"),
    "AgentItem": ("areno.api.agentic", "AgentItem"),
    "AgentTrainBatch": ("areno.api.agentic", "AgentTrainBatch"),
    "AgentTrajectory": ("areno.api.agentic", "AgentTrajectory"),
    "AgentTrajectoryTurn": ("areno.api.agentic", "AgentTrajectoryTurn"),
    "AlgorithmSpec": ("areno.api.algorithms", "AlgorithmSpec"),
    "LossMaskPolicy": ("areno.api.agentic", "LossMaskPolicy"),
    "HarnessRunner": ("areno.api.harness", "HarnessRunner"),
    "HarnessAttempt": ("areno.api.harness", "HarnessAttempt"),
    "HarnessEnvironment": ("areno.api.harness", "HarnessEnvironment"),
    "HarnessDataset": ("areno.api.harness", "HarnessDataset"),
    "RecordLoader": ("areno.api.harness", "RecordLoader"),
    "EnvironmentSpec": ("areno.agent.harness_trial", "EnvironmentSpec"),
    "WorkspaceDataset": ("areno.agent.harness_base", "WorkspaceDataset"),
    "TaskVerification": ("areno.agent.harness_base", "TaskVerification"),
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
    "jev_loss_fn": ("areno.api.algorithms", "jev_loss_fn"),
    "list_algorithms": ("areno.api.algorithms", "list_algorithms"),
    "list_harnesses": ("areno.api.harness", "list_harnesses"),
    "list_environments": ("areno.api.harness", "list_environments"),
    "list_datasets": ("areno.api.harness", "list_datasets"),
    "ppo_loss_fn": ("areno.api.algorithms", "ppo_loss_fn"),
    "register_algorithm": ("areno.api.algorithms", "register_algorithm"),
    "register_harness": ("areno.api.harness", "register_harness"),
    "register_environment": ("areno.api.harness", "register_environment"),
    "register_dataset": ("areno.api.harness", "register_dataset"),
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
    "JevCudaConfig",
    "JevMlxConfig",
    "JevTrainerConfig",
    "DecisionScorer",
    "AlgorithmSpec",
    "CudaConfig",
    "MlxConfig",
    "NpuConfig",
    "NPU",
    "LoraConfig",
    "PromptBatch",
    "PromptItem",
    "AgentBatch",
    "AgentItem",
    "AgentTrainBatch",
    "AgentTrajectory",
    "AgentTrajectoryTurn",
    "LossMaskPolicy",
    "HarnessRunner",
    "HarnessAttempt",
    "HarnessEnvironment",
    "HarnessDataset",
    "RecordLoader",
    "EnvironmentSpec",
    "WorkspaceDataset",
    "TaskVerification",
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
    "register_harness",
    "register_environment",
    "register_dataset",
    "list_harnesses",
    "list_environments",
    "list_datasets",
    "dpo_loss_fn",
    "gspo_loss_fn",
    "grpo_loss_fn",
    "ppo_loss_fn",
    "sft_loss_fn",
    "jev_loss_fn",
]
