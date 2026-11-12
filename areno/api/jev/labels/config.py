"""Explicit opt-in to label-token JEV and full-parameter MLX QAT."""

from dataclasses import dataclass

from areno.api.config import MlxConfig
from areno.api.trainer_config import TrainerConfig


@dataclass
class LabelJevMlxConfig(MlxConfig):
    label_jev: bool = True


@dataclass
class LabelJevTrainerConfig(TrainerConfig):
    """Offline LM-label training, without a scalar decision-head configuration."""

    max_seq_len: int = 4096
    label_temperature: float = 1.0
    seed: int = 17
    data_start_index: int = 0

    def __post_init__(self):
        import math

        TrainerConfig.__post_init__(self)
        if self.seed is None:
            self.seed = 17
        if self.data_start_index < 0:
            raise ValueError("data_start_index must be nonnegative")
        if self.backend not in {"cuda", "mlx"} or self.world_size != 1 or self.tp_size != 1:
            raise ValueError("label JEV currently requires CUDA/MLX with world_size=tp_size=1")
        if self.max_context_len is not None:
            self.max_seq_len = self.max_context_len
        if self.save_path is None or self.max_seq_len < 2:
            raise ValueError("label JEV requires --save-path and max_seq_len>=2")
        if not math.isfinite(self.label_temperature) or self.label_temperature <= 0:
            raise ValueError("label_temperature must be finite and positive")

    def mlx_config(self):
        from dataclasses import fields

        base = TrainerConfig.mlx_config(self)
        return LabelJevMlxConfig(**{field.name: getattr(base, field.name) for field in fields(base)})
