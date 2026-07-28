"""Data containers and helpers shared across engine, runtime and serving.

`batch` defines the dataclasses returned to the user (rollouts, train stats,
sampling parameters) plus tree-walking helpers to move them between devices.
Submodules `rollout_state`, `sampling`, and `tokenizer` are imported on demand
by the runtime and worker layers.
`conversation_normalizer` provides role normalization and tool-message pairing
validation for multi-turn agentic conversations.
"""

from areno.engine.data.batch import RolloutOutput, SamplingParams, TrainStats, to_cpu, to_device
from areno.engine.data.conversation_normalizer import (
    BatchNormalizeReport,
    ConversationValidationError,
    NormalizeResult,
    normalize_conversation,
    normalize_dataset,
    normalize_dataset_iter,
    normalize_role,
)

__all__ = [
    "RolloutOutput",
    "SamplingParams",
    "TrainStats",
    "to_cpu",
    "to_device",
    "BatchNormalizeReport",
    "ConversationValidationError",
    "NormalizeResult",
    "normalize_conversation",
    "normalize_dataset",
    "normalize_dataset_iter",
    "normalize_role",
]
