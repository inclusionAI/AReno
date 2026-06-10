"""Per-instance runtime context shared with the backend.

`Context` is constructed once during `Trainer.init` and threaded through every
backend call. It carries model identity, tokenizer state, world-size topology,
and a monotonic global step counter owned by the high-level `Trainer`.
"""

from areno.api.config import BackendConfig

class Context:
    """Shared immutable-ish runtime context passed into backend calls."""

    def __init__(self, world_size, model_path: str, tokenizer, custom_config: BackendConfig | None = None, eos_token_ids: tuple[int, ...] = ()):
        """Store model path, tokenizer, world size, config, and EOS ids.

        `global_step` starts at -1 so the first training iteration produces
        step 0 when `Trainer` opens the step.
        """

        self.model_path = model_path
        self.world_size = world_size
        self.global_step = -1
        self.tokenizer = tokenizer
        self.custom_config = custom_config
        self.eos_token_ids = eos_token_ids

    def step(self) -> int:
        """Advance and return the global training step counter."""

        self.global_step += 1
        return self.global_step
