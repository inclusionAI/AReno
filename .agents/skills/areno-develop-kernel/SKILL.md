---
name: areno-develop-kernel
description: Develop, optimize, debug, and validate an AReno CUDA, Triton, fused, attention, convolution, routing, or MoE operator. Use when changes touch areno/accel or a runtime kernel boundary and require forward, backward, dtype, layout, CUDA graph, or benchmark validation.
---

# Develop an AReno Kernel

Define the mathematical reference, shapes, dtype, layout, supported devices, and backward contract before implementation.

Develop and commit on a local branch. A remote GPU checkout is validation-only:
pull the committed branch there, and never edit or copy source files into it.

## Generic harnesses

Both callables must accept one tensor and return one tensor:

```bash
python .agents/skills/areno-develop-kernel/scripts/check_operator.py \
  --reference package.module:reference --candidate package.module:candidate \
  --shape 8,16 --dtype float32 --device cuda

python .agents/skills/areno-develop-kernel/scripts/benchmark_operator.py \
  --callable package.module:candidate --shape 8,16 --dtype float16 --device cuda
```

For multi-input or stateful kernels, extend a focused test under `tests/` rather than weakening this harness.

## Workflow

1. Locate Python wrapper, extension registration, C++/CUDA source, engine layer, and model call site.
2. Add a small PyTorch reference and shape/dtype assertions.
3. Implement forward and backward before using it in training.
4. Test representative shapes, boundary shapes, supported non-contiguous layouts, finite output, and gradients.
5. Validate TP/sequence-parallel local shapes and CUDA graph capture/replay.
6. After pulling a branch that changes `areno/accel`, rebuild remotely with `pip install -e . --no-deps --no-build-isolation`. Do not reinstall for Python-only changes.
7. Benchmark only after correctness. Read [references/kernel-checklist.md](references/kernel-checklist.md).

Do not add a silent production fallback for a required kernel. Report unsupported cases explicitly.
