# Model Kernel Policy

- Prefer existing `areno/engine/layers` and `areno/accel` operators.
- Required hot paths do not silently fall back to Python reference code.
- Training use requires a verified backward path.
- Kernel APIs describe mathematical operations, not one model brand.
- Prefill and decode may differ but must preserve normalization, rotary, scale, masking/window, and cache semantics.
- Warm compiled callables before CUDA graph capture. Decode validation must exercise graph replay.
- Avoid `.item()`, `.tolist()`, and CPU synchronization in model hot paths.
- If `areno/accel` changes, rebuild editable installation with `pip install -e . --no-deps --no-build-isolation`.
