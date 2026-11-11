"""Optimizer package.

Exports the sharded AdamW optimizers used by training.
Other optimizer variants should be added here as separate modules and
re-exported through `__all__`.
"""


def __getattr__(name):
    # MLX imports the framework-neutral dynamic codebooks from this package.
    # Load Torch optimizers only when a CUDA consumer actually requests one.
    if name == "AdamW4bit":
        from areno.engine.optim.adamw_4bit import AdamW4bit

        return AdamW4bit
    if name in {"AdamW8bit", "set_optimizer_state_precision"}:
        from areno.engine.optim import adamw_8bit

        return getattr(adamw_8bit, name)
    if name == "AdamWFP32Master":
        from areno.engine.optim.adamw_fp32_master import AdamWFP32Master

        return AdamWFP32Master
    raise AttributeError(name)


__all__ = ["AdamW4bit", "AdamW8bit", "AdamWFP32Master", "set_optimizer_state_precision"]
