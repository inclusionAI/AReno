"""Checkpoint loading and saving.

`common` defines declarative model specs and generic load/save ops.
`io` owns safetensors indexing, tensor-parallel gather, and HF shard writing.
Model files such as `qwen3.py` and `gemma4.py` should only describe layout
(specs and op tuples); they should not contain Python load/save handler code
unless a brand-new layout primitive is required, in which case the primitive
belongs in `common`.
"""
