Ascend NPU integration
======================

The target environment is Linux/aarch64, Ascend 910, CANN 9.0.0,
PyTorch 2.10.0+cpu and torch_npu 2.10.0.post2. The CPU-tagged PyTorch
installation is retained; torch_npu provides the NPU device and operators.

Installation detects ``torch_npu`` without importing it and selects
``requirements/npu.txt``. CUDA and MLX retain their existing dependencies and
builders. The NPU extension compiles its own Ascend C kernels with the CANN
development toolkit, and links their static library into a TorchNPU C++
extension. CMake and the CANN compiler are required. Source the toolkit's
``set_env.sh`` before building. No CUDA compiler is used.

The exact SoC is queried from ``aclrtGetSocName`` after TorchNPU initialization.
The current kernels target A2/A3 (Ascend 910B variants and 910_93xx); a generic
``Ascend910`` display name is not enough to select an ISA. Original Ascend 910
and other unsupported targets fail explicitly. ``ARENO_NPU_SOC`` provides an
optional override for cross-compilation without visible hardware.

.. code-block:: bash

   python -m pip install -e . --no-build-isolation
   python -m pip install pytest
   python -m pytest -q tests/test_npu_activation.py tests/test_npu_normalization.py tests/test_npu_optimizer.py

Current validation boundary
---------------------------

This is **not complete NPU training/serving support**. The compiled extension
currently exposes all ten CUDA activation entries: SiLU, sigmoid, softplus,
SiLU-and-multiply and tanh-GELU-and-multiply, with their backward operators.
They preserve the public accel wrapper signatures and use tiled Ascend C
vector instructions with FP32 intermediates, FP32/FP16/BF16 storage and exact
tail transfers. Activation math does not use ATen operations. Non-contiguous
output buffers use a layout copy after the kernel.

The six RMSNorm entries (standard, optional-scale and SiLU-gated, each with
forward/backward) also use Ascend C. Row reductions stream through bounded UB
tiles, store FP32 ``inv_rms`` for backward and accumulate FP32 weight gradients
with DMA atomic addition. The separate grouped ``rms_norm_gate_fwd`` entry
matches the CUDA Triton operator's sigmoid gate and per-group weights; it does
not substitute SiLU gating. Python wrappers and model call sites are unchanged.

The FP32-state, compact FP32-master, blockwise 8-bit and blockwise 4-bit AdamW
entries use Ascend C vector math. BF16/FP32 model weights and gradients
are supported independently. Compact master metadata retains the existing
uint16 low bits and packed rounding carries, including slices that start
inside a byte. No expanded FP32 master tensor is allocated. Quantized updates
retain CUDA's codebooks, nearest-code tie rule, packed offsets and canonical
odd tail nibble. A nonfinite gradient or update skips its entire quantization
block before any persistent data is written. Quantization lookup and compact
bit packing run on the AI Core scalar pipeline with explicit DMA/vector
synchronization. These implementations have not been benchmarked.

The optimizer acceptance tests include multi-step updates, mixed gradient
dtypes, metadata canaries, BF16 rounding ties, quantization block boundaries,
nonfinite block skipping and shared optimizer checkpoint save/load. Matrix
parameters in AdamW4bit still require the separate factored variance kernels;
the blockwise entry does not replace that algorithm.

Device guards and TorchNPU's current stream are used for each launch. The
acceptance suite covers tile boundaries, strided tensors, storage offsets,
empty inputs, softplus tails, saved normalization statistics, every RMSNorm
input gradient, many-row weight accumulation, non-default streams and
two-device execution.
This source has not yet been compiled or numerically validated on Ascend.

The backend source reuses the CUDA workflows for training, generation, losses,
optimizers, checkpoints and serving, with Ascend device initialization, HCCL
and memory probes. The backend directory contains only ``__init__.py`` and
``backend.py``. The shared TP/DP rank layout is reused. These paths have not
run on Ascend. Worker startup rejects the incomplete native extension before
starting a training or serving job.

Remaining native families include dense and
grouped linear, attention, embedding, convolution, routing/MoE, recurrent
operators, and AdamW4bit factored statistics/update. The existing opt-in
``tests/test_npu_end_to_end.py`` becomes the SFT/rollout/checkpoint acceptance
test once these kernels are complete; it is not expected to pass yet.

Development validation checks packaging, device dispatch and shared workflows
on CPU. NPU extension compilation and numerical execution must be validated in
the target CANN/torch_npu environment. Passing the CPU packaging checks does
not establish kernel correctness or performance.

Implementation references
-------------------------

* `Ascend C PyTorch kernel launch integration <https://asc.gitcode.com/guide/programming_guide/advanced_programming/ai_framework_adaptation/pytorch_framework.html>`_
* `TorchNPU 2.10 extension builder <https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/utils/cpp_extension.py>`_
* `Ascend C DataCopyPad API <https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910beta2/API/ascendcopapi/atlasascendc_api_07_0265.html>`_
* `CANN 9 DMA atomic addition <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0210.html>`_
* `Ascend C scalar memory synchronization <https://asc.gitcode.com/guide/technical_appendix/concepts_and_terms/memory_access/scalar_read_write.html>`_
