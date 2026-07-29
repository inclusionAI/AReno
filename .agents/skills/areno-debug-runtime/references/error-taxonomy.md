# Error Taxonomy

- Python/config: CLI validation, dataclass mismatch, missing callback, dataset shape.
- Checkpoint: key coverage, shape/layout, TP split, dtype, tied weights.
- Inference: cache indexing, scheduler admission, CUDA graph inputs, decode kernel.
- Training: packed dimensions, masks/positions, recompute, backward, optimizer.
- Distributed: inconsistent collective order, one-rank early exception, timeout during cleanup.
- Compilation: unsupported mutation/dtype, graph break, Triton autotune or compiler failure.
- Native memory: illegal access may surface later; rerun the minimal case with synchronization or DSA tooling where available.

NCCL watchdog messages are usually consequences when another rank failed first.
