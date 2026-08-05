# Kernel Checklist

- Forward agrees with a clear reference at justified `atol`/`rtol`.
- Backward agrees for every differentiable input and parameter.
- Outputs and gradients are finite for representative values.
- Dispatch validates device, dtype, rank, shape, strides, and alignment assumptions.
- Stream use, temporary storage, and launch errors are correct.
- CUDA graph warmup, capture, and replay do not allocate or query unsupported state.
- Distributed callers pass rank-local shapes expected by the kernel.
- Benchmarks synchronize, warm up, report distribution, and identify hardware/software metadata.
- Existing operators receive regression coverage.
