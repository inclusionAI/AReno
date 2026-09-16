Ascend NPU integration
======================

The target environment is Linux/aarch64, Ascend910_9382, CANN 9.0.0,
PyTorch 2.10.0+cpu and torch_npu 2.10.0.post2. The CPU-tagged PyTorch
installation is retained; torch_npu provides the NPU device and operators.

Installation detects ``torch_npu`` without importing it and selects
``requirements/npu.txt``. CUDA and MLX retain their existing dependencies and
builders. The NPU extension compiles its own Ascend C kernels with the CANN
development toolkit, and links their static library into a TorchNPU C++
extension. Dense matrix multiplication links CANN 9's ``opapi_nn`` and
``nnopbase`` libraries, so the CANN NN operator package is also required.
CMake and the CANN compiler are required. Source the toolkit's
``set_env.sh`` before building. No CUDA compiler is used.
Fused experts additionally links the toolkit's ``tiling_api`` and ``platform``
libraries to query Cube/Vector core counts and Matmul system workspace size.

The exact SoC is queried from ``aclrtGetSocName`` after TorchNPU initialization.
The current kernels target A2/A3 (Ascend 910B variants and 910_93xx); a generic
``Ascend910`` display name is not enough to select an ISA. Original Ascend 910
and other unsupported targets fail explicitly. ``ARENO_NPU_SOC`` provides an
optional override for cross-compilation without visible hardware.

Before building, ``python scripts/check_ascend.py`` probes the actual runtime
SoC and checks BF16 matrix multiplication and its gradient on NPU 0. Use
``--all-devices`` to check every visible NPU sequentially. The script requires
the existing CANN/torch_npu environment, but does not import AReno. It ignores
the build-time SoC override. A pass establishes the hardware target and basic
TorchNPU execution only; it does not test AReno kernels or HCCL.

.. code-block:: bash

   python -m pip install -e . --no-build-isolation
   python -m pip install pytest
   python -m pytest -q tests/test_npu_activation.py tests/test_npu_normalization.py \
     tests/test_npu_optimizer.py tests/test_npu_optimizer_factored.py tests/test_npu_embedding.py \
     tests/test_npu_linear.py tests/test_npu_conv.py
   python -m pytest -q tests/test_grouped_linear.py -k 'npu and not cpu and not cuda'
   python -m pytest -q tests/test_routing.py -k npu
   python -m pytest -q tests/test_moe_native.py -k npu
   python -m pytest -q tests/test_attention_native.py -k npu
   python -m pytest -q tests/test_fused_experts_native.py -k npu
   torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_npu_optimizer_distributed.py

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
parameters in AdamW4bit use native factored statistics and update kernels.
Statistics accumulate row and column sums directly from each parameter shard;
updates reconstruct variance within a bounded tile and reuse the blockwise
momentum update and packing implementation. Factor reduction across DP ranks,
EMA updates, optimizer state allocation and checkpoint handling reuse the
existing shared Torch implementation. The distributed tests compare identical
DP layouts on NPU/HCCL and CPU/Gloo, including empty shards and checkpoint
resume. Only the CPU/Gloo version of this test flow has run locally.

Vocab embedding also retains its shared autograd wrapper. Native forward uses
DMA to gather local vocabulary rows and writes zero for nonlocal tokens;
backward uses storage-dtype atomic addition for repeated local token IDs.
Tests cover strided inputs, scalar and empty IDs, uneven vocabulary shards,
nonfinite values and bitwise preservation of stored forward values.

Dense linear retains the existing Python autograd wrapper and calls CANN's
``aclnnMm`` in C++ for forward, input gradient and weight gradient, corresponding
to CUDA's cuBLAS calls. Transposes are matrix descriptors over existing storage.
The Cube precision mode retains the input dtype instead of reducing FP32 input
precision. Ascend C implements bias addition and FP32 bias-gradient accumulation.
Bias addition preserves CUDA's separate GEMM-output and bias-output rounding.
No alternate model or training workflow is introduced.

Grouped linear reuses the host-side expert loop extracted from CUDA into
``grouped_linear_common.h``. Both devices share counts validation, tensor
slicing, output allocation, empty-expert handling and gradient selection.
CUDA supplies cuBLAS GEMM; NPU supplies the same CANN GEMM used by dense linear.
The existing list and int32/int64 tensor-count entry points are preserved.
Tensor counts still synchronize to the CPU, matching the existing CUDA
implementation; this path is not suitable for graph capture. The list-count
path retains a CUDA graph regression test. The common C++ code has a separate
CPU test adapter, while the same contract suite targets CUDA and NPU kernels.

Causal depthwise Conv1d + SiLU has all five CUDA-compatible entries: ordinary
and packed forward/backward, and one-token decode. The existing Python wrapper
still casts weights to FP32 and handles autograd. Ascend C computes the
convolution, SiLU, input gradient and weight gradient with FP32 intermediates;
saved preactivations retain FP32. Bounded channel tiles use strided DMA and
vector Gather for weights and decode history. Each weight-gradient tile has
one owner and accumulates across tokens without atomics. Packed boundaries
are read on device; as with CUDA, callers must provide valid nondecreasing
offsets spanning all tokens. Empty segments are allowed. Decode leaves history
updates to the caller. Tests include packed sequence isolation, decode/prefill
agreement, storage offsets, empty shapes and unused nonfinite weight taps.
These native sources remain uncompiled and unvalidated on Ascend.

Routing now includes native softmax top-k forward/backward and grouped
sigmoid-plus-bias selection. CUDA and NPU use the same C++ top-k insertion
helper, including the lower-index tie rule. Softmax can return the selected
probabilities with or without renormalization; backward follows the same
selected-probability derivative as CUDA. Grouped selection retains CUDA's
``top_k // topk_group`` group-score count and applies expert bias only during
selection. Returned indices are int64 and weights are FP32. Probability math
uses Ascend C vector operations; bounded top-k insertion runs on the AI Core
scalar pipeline, as CUDA's final selection runs in one thread. These sources
have not run on Ascend and have not been benchmarked. A compiled CPU test
executes the actual common selection helper; the device contract suite covers
both CUDA and NPU, including gradients, ties and saturation.

MoE permutation/alignment exposes the same six native entries as CUDA.
Gather and unpermute reuse the Ascend embedding DMA/atomic kernels. Dense-map
permutation preserves expert-major, ascending-token order and retains routed
zero weights. Top-k permutation filters local expert shards and zero weights;
its output allocation, counts copy and host prefix scan are shared with CUDA
through ``moe_permute_common.h``. This retains CUDA's existing CPU synchronization
and is not graph-capturable. Device-side indexing uses tiled route histograms,
prefix scans and metadata writes. Top-k weight backward scatters into the
saved token/top-k slots. Block alignment retains the ``-1`` expert bucket,
route-count padding sentinel and caller-owned output buffers. The shared
unpermute autograd wrapper now saves the contiguous indices used by forward,
so backward also handles strided token indices correctly on both devices.
Compiled CPU tests cover the common allocator; device tests cover expert
shards, repeated indices, gradients, storage offsets, padding canaries and
CUDA graph replay for fixed-shape paths. The native Ascend implementation
still requires compilation and numerical validation on hardware.

The five native attention entries cover dense and packed forward/backward,
and paged single-token decode. They follow the CUDA diagnostic attention
kernel's FP32 online softmax and saved-output derivative, using Ascend C
vector arithmetic and FP32 atomic K/V gradient accumulation. Packed attention
supports grouped query heads and reads sequence boundaries on device,
including empty segments. Dense attention retains query offsets and sliding
windows. Paged decode copies K/V updates into the caller's cache, evaluates
the requested number of splits and merges their softmax statistics; empty
or fully masked splits contribute nothing. Cache lengths remain caller-owned.
Head dimensions stream through fixed UB tiles, without materializing a QK
matrix. Heads wider than a tile repeat score calculations for each output
tile; this implementation has not been benchmarked. The shared Python
attention/autograd wrappers are unchanged, including CUDA's existing
reference-based paged-decode backward. Device tests cover FP32/FP16/BF16,
saved-output gradients, GQA, packed isolation, storage offsets, cache writes,
empty splits, prefill/decode agreement, numerical stability and streams.
These Ascend sources have not yet been compiled or run on hardware.

Fused expert inference uses the existing public entry and model call sites.
It reuses native route alignment, embedding row transfers and gated activation.
The two projections run on Cube through Ascend C Matmul, using static
16-by-64-by-128 tiles and FP32 accumulation across K tiles. Separate vector
epilogues preserve the first projection's storage rounding and multiply the
second projection's FP32 result by its routing weight before storage rounding.
Reduction then follows original top-k slot order in FP32, applies the routed
scale and casts the final output. Both SiLU and tanh-GELU are supported.
``-1`` routes return zero even with nonfinite routing weights; valid zero-weight
routes still evaluate their expert, as on CUDA. Route counts and expert metadata
remain on device. Like CUDA's fused entry, this entry has no backward; training
retains the shared permutation/grouped-linear autograd path.

The initial Cube implementation uses packed input and a shared FP32 accumulation
buffer in addition to storage-dtype intermediates. It has more temporary memory
and GM traffic than CUDA's fused Triton implementation and has not been
benchmarked. The shared device suite checks tails, repeated experts, local
expert/TP shards, strided storage, streams and graph replay with changing routes.
Targeted cases reject premature down-projection rounding, FP16 overflow before
routing-weight multiplication, and summation in sorted expert order. CPU checks
validate those references only; the Ascend sources still require compilation
and numerical validation on the target machine.

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

Remaining native families include recurrent operators.
The existing opt-in
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
* `CANN 9 matrix multiplication <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/aolapi/context/ops-nn/aclnnMm.md>`_
* `CANN 9 tensor descriptors <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/aolapi/operatorlist_00019.html>`_
* `CANN 9 vector Gather <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0092.html>`_
* `CANN 9 strided DataCopyPad <https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0265.html>`_
* `Ascend C scalar memory synchronization <https://asc.gitcode.com/guide/technical_appendix/concepts_and_terms/memory_access/scalar_read_write.html>`_
* `CANN 9 static Matmul tiling <https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0666.html>`_
* `CANN 9 Matmul output and atomic accumulation <https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0640.html>`_
* `Ascend C platform query and linking <https://asc.gitcode.com/api/Utils-API/platform_info/PlatformAscendCManager.html>`_
