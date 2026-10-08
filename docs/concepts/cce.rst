Cut cross-entropy
=================

CUDA training enables cut cross-entropy (CCE) by default. The vocabulary
projection computes selected next-token log-probabilities without storing the
full token-by-vocabulary logits matrix. Existing SFT, DPO, GSPO, GRPO and PPO
actor losses keep their masks, reductions and upstream gradients.

To use the previous training path, add ``--no-cce`` to ``areno train``.
Use ``--cce`` to enable it explicitly. SDK trainer configurations accept
``cce=False``; lower-level ``CudaConfig`` accepts ``runtime={"cce": False}``.
Other backends retain their existing loss implementation.

The implementation tiles the projection in forward and recomputes tiles in
backward. Every vocabulary entry participates: there is no approximate
gradient filtering. It preserves the output head's projection dtype and
logit softcap. Tensor-parallel heads combine vocabulary normalizers and
retain their existing input-gradient reduction boundary.

BF16/FP16 rounding and FP32 reduction ordering mean that results need not be
bitwise identical to a materialized head. Validate token log-probabilities,
losses and gradients together; a policy loss scalar alone can hide differences
because some objectives detach the current log-probability in their ratio.

Memory and performance
----------------------

CCE removes the full logits allocation and its associated backward storage.
It retains hidden states, the head weight and per-token normalizers. The
partial reduction workspace is bounded to 1024 tokens at a time. Backward
uses FP32 gradient accumulation; a trainable head still requires its weight
gradient, while a frozen LoRA base head does not.

On unified-memory systems such as DGX Spark, these savings reduce physical
DRAM demand rather than moving it to a separate CPU allocation. Throughput
depends on head dtype, vocabulary, token count and device; memory savings
do not imply a speedup for every shape.

Validation
----------

Run numerical tests and the isolated head benchmark on a CUDA host::

   NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
     python -m pytest -q -s tests/test_cce.py tests/test_cce_cpu.py
   NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
     python scripts/benchmark_cce.py --tokens 1024 --hidden 1536 --vocab 157184

The precision overrides prevent an environment-level cuBLAS TF32 override
from weakening the FP32 reference. The benchmark reports additional CUDA
allocation above the resident inputs, forward/backward time and numerical
errors; it does not measure total system DRAM or end-to-end trainer speed.

The algorithm follows the tiled projection/reduction principle from
`Cut Your Losses in Large-Vocabulary Language Models
<https://arxiv.org/abs/2411.09009>`_, with AReno-owned Triton kernels and no
additional runtime package.
