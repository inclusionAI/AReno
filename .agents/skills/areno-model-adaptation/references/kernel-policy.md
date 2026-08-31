# Model Kernel Policy

- Prefer existing `areno/engine/layers` and `areno/accel` operators.
- Required hot paths do not silently fall back to Python reference code.
- Plain PyTorch reference implementations are short-lived correctness scaffolding. Convert rollout, prefill, decode, scoring, and training hot paths to an AReno-owned fused primitive before treating a model as production-ready.
- Training use requires a verified backward path.
- Kernel APIs describe mathematical operations, not one model brand.
- Runtime-critical paths use AReno-owned model code and kernels by default. When the project already declares a kernel dependency such as FLA and the user explicitly accepts it, expose it through a small `areno/accel` wrapper rather than importing it from model code or vendoring it.
- Treat SGLang as a semantic reference for recurrent linear attention. An explicitly accepted FLA dependency may implement gated-delta through `areno/accel`, but model code must call the AReno wrapper.
- Prefill and decode may differ but must preserve normalization, rotary, scale, masking/window, and cache semantics.
- Warm compiled callables before CUDA graph capture. Decode validation must exercise graph replay.
- Avoid `.item()`, `.tolist()`, and CPU synchronization in model hot paths.
- Any NaN in loss, logprobs, ratios, or grad norm is a correctness blocker. Reproduce on the smallest failing batch, locate the first non-finite tensor, and check reduction dtype, gate/beta domain, accumulators, masks, packed lengths, checkpoint transforms, and train/decode agreement.
- If `areno/accel` changes, rebuild editable installation with `pip install -e . --no-deps --no-build-isolation`.
