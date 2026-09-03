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
    "DegenerateFilterConfig": ("areno.api.data", "DegenerateFilterConfig"),
    "DegeneratePolicy": ("areno.api.data", "DegeneratePolicy"),
    "DegenerateReason": ("areno.api.data", "DegenerateReason"),
    "LossMaskPolicy": ("areno.api.agentic", "LossMaskPolicy"),
    "PromptBatch": ("areno.api.data", "PromptBatch"),
    "PromptItem": ("areno.api.data", "PromptItem"),
    "SampleQualityReport": ("areno.api.data", "SampleQualityReport"),
    "RewardEvent": ("areno.api.rewards", "RewardEvent"),
    "RewardRecord": ("areno.api.rewards", "RewardRecord"),
    "RolloutResult": ("areno.api.models", "RolloutResult"),
    "RolloutSequence": ("areno.api.models", "RolloutSequence"),
    "RolloutSession": ("areno.api.agentic", "RolloutSession"),
    "SamplingParams": ("areno.api.models", "SamplingParams"),
    "Trainer": ("areno.api.trainer", "Trainer"),
    "TrainSequence": ("areno.api.models", "TrainSequence"),
    "apply_degenerate_policy": ("areno.api.data", "apply_degenerate_policy"),
    "check_preference_pair": ("areno.api.data", "check_preference_pair"),
    "check_prompt_text": ("areno.api.data", "check_prompt_text"),
    "check_response_text": ("areno.api.data", "check_response_text"),
    "check_tokenized_prompt": ("areno.api.data", "check_tokenized_prompt"),
    "check_trainable_tokens": ("areno.api.data", "check_trainable_tokens"),
    "dpo_loss_fn": ("areno.api.algorithms", "dpo_loss_fn"),
    "format_degenerate_reasons": ("areno.api.data", "format_degenerate_reasons"),
    "get_algorithm": ("areno.api.algorithms", "get_algorithm"),
    "grpo_loss_fn": ("areno.api.algorithms", "grpo_loss_fn"),
    "gspo_loss_fn": ("areno.api.algorithms", "gspo_loss_fn"),
    "list_algorithms": ("areno.api.algorithms", "list_algorithms"),
    "ppo_loss_fn": ("areno.api.algorithms", "ppo_loss_fn"),
    "register_algorithm": ("areno.api.algorithms", "register_algorithm"),
    "record_degenerate_reason": ("areno.api.data", "record_degenerate_reason"),
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
    "DegenerateFilterConfig",
    "DegeneratePolicy",
    "DegenerateReason",
    "PromptBatch",
    "PromptItem",
    "SampleQualityReport",
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
    "apply_degenerate_policy",
    "check_preference_pair",
    "check_prompt_text",
    "check_response_text",
    "check_tokenized_prompt",
    "check_trainable_tokens",
    "format_degenerate_reasons",
    "record_degenerate_reason",
    "get_algorithm",
    "list_algorithms",
    "register_algorithm",
    "dpo_loss_fn",
    "gspo_loss_fn",
    "grpo_loss_fn",
    "ppo_loss_fn",
    "sft_loss_fn",
]
