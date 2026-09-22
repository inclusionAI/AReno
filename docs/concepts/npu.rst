Ascend NPU integration
======================

The NPU backend integrates Ascend with AReno's shared training, rollout,
scoring and serving workflows. Its registered training algorithms are SFT,
DPO, GRPO, GSPO and PPO, with actor, reference, reward and critic model roles.
It also reuses distributed TP/DP execution, custom losses, optimizer offload
and checkpoint handling. SFT is one supported workflow, not the scope of the
NPU adaptation. Hardware validation status and operator limitations are
documented below.

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

Attention uses ``flash-attn-npu==0.3.0`` for supported shapes, with native
compatibility kernels for the cases described below. Linear attention reuses upstream
FLA at commit ``e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2``, which includes
Ascend backends. Both are NPU-only dependencies; CUDA dependencies are unchanged.
The FLA ``[npu]`` extra is deliberately omitted because it pins a different
TorchNPU stack. Keep the existing CANN-compatible ``torch``, ``torch_npu`` and
``triton-ascend`` installation. Upstream FLA's pinned CI uses CANN 9.1.0,
TorchNPU 2.9.0.post6 and Triton Ascend 3.2.2 on A2; compatibility with this
node's CANN 9.0.0 / TorchNPU 2.10.0.post2 / Triton Ascend 3.2.1 is unverified.

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
   python -m pytest -q tests/test_npu_library_attention.py
   python -m pytest -q tests/test_npu_linear_attention.py
   python -m pytest -q tests/test_attention_native.py -k npu
   python -m pytest -q tests/test_npu_runtime.py
   python -m pytest -q tests/test_fused_experts_native.py -k npu
   torchrun --standalone --nproc_per_node=2 -m pytest -q tests/test_npu_optimizer_distributed.py

Current validation boundary
---------------------------

The latest target-node run successfully started the Qwen3 HTTP server and
entered prefill, where ``flash-attn-npu`` rejected the ``Ascend910_9382``
device name. That import failure now selects native attention automatically.
The old validation-only startup guard and model whitelist have been removed.
NPU workers enter the shared model,
training and rollout lifecycle after extension loading, device selection and
HCCL initialization. Missing dependencies and unsupported operator arguments
still fail at their respective entry points. No AReno kernel numerical
acceptance or end-to-end model run is proven yet.

All ten NPU bindings query storage format through the exported
``get_npu_format`` API, retaining the rejection of packed storage layouts.
The builder checks CANN launcher symbols and imports the exact new extension
in a fresh process before completing installation. CPU tests exercise real
shared-library loading, missing symbols, and initialization failures.

The compiled extension
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
These native sources still require numerical validation on Ascend.

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
still requires a working extension import and numerical validation on hardware.

Attention implementation
------------------------

Native NPU attention is implemented in Ascend C and connected to the shared
accel API and engine attention routes. Dense and packed causal attention have
native forward and backward kernels. Paged decode has native KV-cache updates,
attention forward and split reduction; its diagnostic backward uses the shared
Torch reference rather than a native Ascend backward kernel.

The implementation lives in ``areno/accel/csrc/npu/attention_kernel.cpp``
and ``attention.cpp``, with device/library selection in
``areno/accel/npu/attention.py``. Select ``--attn-backend native`` for training
or serving to use these kernels directly. Implementation and runtime routing
are present; numerical acceptance on Ascend and performance benchmarking
remain pending.

With ``attn_backend="flash"`` (the default), dense and packed FP16/BF16
attention with head dimensions up to 256 call
``flash-attn-npu`` with its own autograd. Supported paged decode calls its
KV-cache API and updates the original cache. Both shared engine attention
routes select the implementation by tensor device, dtype and layout.
CUDA retains its existing FlashAttention and diagnostic native kernels. The adapter
truncates invisible K/V suffixes to preserve explicit query offsets and
passes sliding windows, GQA and packed boundaries to the library. Packed
lengths remain on device, with the total token count used as a conservative
maximum length. The library KV-cache path is inference-only and requires head
dimensions divisible by 8 and cache block sizes divisible by 256.
The NPU library acceptance suite covers outputs and gradients, packed
isolation, empty segments, cache writes, GQA and streams. It also exercises
native fallback when the library is unavailable, requiring the AReno C
extension in that case. These checks still need to run on the target node.

``areno serve --attn-backend native`` explicitly bypasses FlashAttention for
both prefill and decode. The same runtime option applies to NPU training.
With ``flash``, each worker checks library availability after selecting its
device. A missing top-level ``flash_attn_npu`` package or the library's
``Unsupported Ascend device:`` rejection triggers native attention and one
warning per device. Version 0.3.0 rejects the reported ``Ascend910_9382``
name. The decision is cached for the worker lifetime, so later layers and
decode steps do not repeat the failed import. Missing internal dependencies,
undefined symbols, and operator execution errors still propagate.

To exercise explicit native serving with a local model:

.. code-block:: bash

   areno serve --model-path /path/to/local/checkpoint --port 8000 \
     --max-running-prompts 1 --attn-backend native

Omit ``--attn-backend native`` to exercise automatic selection. Both modes
use the same shared serving engine and Ascend kernels. At the accel API,
``force_native=True`` also bypasses the optional library.

Decode graph capture now defaults to enabled on NPU, using
``torch.npu.NPUGraph`` through the shared ``DecodeGraph`` implementation.
CUDA continues to use ``torch.cuda.CUDAGraph``. Warmup, static input buffers,
scratch KV blocks, recurrent padding slots and cache invalidation are shared.
Streams, synchronization and memory checks select the worker's device API.
TP ranks agree on available memory and on whether capture succeeded; an OOM
on one rank discards that bucket on all ranks. Other capture errors propagate.
``--eager-decode`` explicitly disables capture and replay; ``compile_model``
remains disabled on NPU independently of graph capture.

With ``--decode-progress-interval-s 1``, NPU progress logs report
``npu_graph=True`` only after graph replay. ``npu_graph=False`` means the
reporting interval used eager decode or contained no replay. CUDA retains
its ``cuda_graph`` field. The native attention graph test and the NPU runtime
test below check changing inputs, cache metadata, padding and replay against
eager results. Actual graph execution and TP/HCCL capture still require
target-node acceptance:

.. code-block:: bash

   python -m pytest -q tests/test_npu_runtime.py -k decode_graph
   python -m pytest -q tests/test_attention_native.py -k 'npu and device_graph'

FP32, head dimensions above 256, and other paged layouts use native Ascend C
compatibility kernels through the same public accel API and shared autograd
wrappers as CUDA. Dense and packed forward/backward use FP32 intermediates and
bounded 512-column tiles; they do not allocate a full sequence-by-sequence
score matrix. Packed GQA boundaries and paged cache metadata remain on device.
Paged decode updates the original cache and reduces split partials in FP32;
its diagnostic backward reuses the existing shared Torch reference. The
compatibility path has no fixed head-dimension limit, but is unbenchmarked and
recomputes scores across head tiles. Only the availability failures listed
above trigger fallback. The shared CUDA/NPU native suite checks outputs,
gradients, cache canaries, empty segments, sliding windows, streams and head
dimensions through 1025. The updated launchers still require a target rebuild
and numerical acceptance before model support can be claimed.

The unfinished seg-LA Ascend C sources have also been removed. KDA training
reuses the existing Torch/FLA wrapper, including gate rounding, normalization
and state layout. Decode calls FLA's recurrent KDA with fused gate and beta
activation, then writes the selected state slots back. GatedDeltaNet and
Lightning Attention use upstream FLA kernels. Bailing's shared Lightning
entry forwards CUDA calls unchanged; the NPU adapter consumes the legacy
``head_first`` argument and passes explicit TP-local decay slopes to FLA
``simple_gla``. This prevents upstream Lightning from replacing the model's
slopes with values derived from the local head count. Regular seg-LA
prefill/decode adapts the state pool and head decay to FLA simple GLA.
These adapters do not implement attention arithmetic. The standalone
``tests/test_npu_linear_attention.py`` checks dense/packed Lightning forward,
backward and final state, plus seg-LA prefill followed by decode and untouched
state slots. It requires Ascend FLA, but not the AReno C++ extension.
FLA state slots must be allocated nonnegative indices. Seg-LA state snapshots
and tree masks are not integrated and fail explicitly. Numerical equivalence
and compilation of the FLA paths on the target machine remain unverified.

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
validate those references only; the Ascend sources still require a working
extension import and numerical validation on the target machine.

Device guards and TorchNPU's current stream are used for each launch. The
acceptance suite covers tile boundaries, strided tensors, storage offsets,
empty inputs, softplus tails, saved normalization statistics, every RMSNorm
input gradient, many-row weight accumulation, non-default streams and
two-device execution.
These operator contracts have not yet passed numerical validation on Ascend.

The backend source reuses the CUDA workflows for training, generation, losses,
optimizers, checkpoints and serving, with Ascend device initialization, HCCL
and shared memory probes. Training measures each microbatch's peak separately;
rollout cache probes, synchronization and allocator cleanup select the worker's
Torch device module. NPU no longer overrides these worker methods. Disk
optimizer offload uses the same pinned buffers, bounded prefetch and completion
events for FP32-master, 8-bit and 4-bit AdamW. Prefetch selects pinned memory
from the bucket's device instead of probing CUDA availability. Checkpoint D2H
copies reuse the bounded stream queue and source lifetime tracking on NPU;
pageable staging remains synchronous. ``tests/test_npu_runtime.py`` checks
non-default streams, strided checkpoint tensors, disk/CPU offload and optimizer
checkpoint resume on devices 0 and 1. It has not run on Ascend.

The backend directory contains only ``__init__.py`` and
``backend.py``. The shared TP/DP rank layout is reused. The shared serving
adapter selects the engine's ``rollout`` role, so serving does not allocate
optimizer state or a training manager. Training retains the selected
FP32-master, 8-bit or 4-bit optimizer. CPU tests cover those ownership rules
and device-before-HCCL startup for single-device and partitioned layouts.

Remaining work includes the recurrent features listed above, library-stack
validation, and native extension numerical acceptance. The opt-in
``tests/test_npu_end_to_end.py`` exercises SFT with all three optimizers,
rollout, checkpoint reload and HTTP serving against a local checkpoint.
This suite's SFT training coverage does not limit the backend's algorithm
scope; DPO, GRPO, GSPO and PPO also require end-to-end hardware acceptance.
The HTTP test runs with both ``native`` and ``flash`` selection and checks
model listing, greedy repeatability, batched completions and worker shutdown.
Run these in a fresh process, separately from kernel
tests, so the coordinator has not acquired a worker's NPU:

.. code-block:: bash

   python -m pip install pytest httpx
   ARENO_NPU_TEST_MODEL=/path/to/local/checkpoint python -m pytest -q tests/test_npu_end_to_end.py

Set ``ARENO_NPU_TEST_WORLD_SIZE`` and ``ARENO_NPU_TEST_TP_SIZE`` to exercise
multiple devices. For example, world size 4 and TP size 2 also exercise two
data-parallel replicas. HTTP serving uses visible devices ``0..world_size-1``,
as the CLI does. These acceptance tests still need to pass on Ascend hardware.

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
* `Ascend FlashAttention library <https://github.com/MinghuasLab/flash-attention-npu/tree/v0.3.0>`_
* `FLA Ascend installation and dependency matrix <https://github.com/fla-org/flash-linear-attention/blob/e52dbc0ea19d3a40d7ab7f9eed855d2b473994d2/INSTALL.md>`_
* `FLA Ascend C operators <https://github.com/flashserve/flash-linear-attention-npu>`_
