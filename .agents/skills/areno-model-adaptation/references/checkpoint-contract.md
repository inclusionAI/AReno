# Checkpoint Contract

- Keep model construction in `model.py` and model-specific load/save mapping in `checkpoint.py`.
- Use shared helpers under `areno/engine/checkpoints/` for indexes, shard reads, gathers, and HF writers.
- Load and save mappings must be inverse, including fused qkv/gate projections and expert layouts.
- Column-parallel weights split/gather output dimensions; row-parallel weights split/gather input dimensions according to the actual layer implementation.
- Replicated tensors must agree across ranks and be written once.
- Progress totals represent the exact claimed key set. Intentional omissions must be explicit.
- Preserve config, tokenizer, processor, and special-token assets.
- Load-save-load must preserve logits before training. Shape equality alone is insufficient.
