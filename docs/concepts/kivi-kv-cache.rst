KIVI KV cache
=============

AReno provides an opt-in KIVI KV cache for CUDA rollout. It stores the
attention keys and values of decoded tokens as asymmetric 2-bit or 4-bit
codes, following `KIVI <https://arxiv.org/abs/2402.02750>`_, while model
weights, training, and checkpoints are unchanged. The default model-dtype KV
cache is used unless the option is set. The option belongs to the engine
runtime configuration; ``areno train`` and ``areno serve`` do not expose it as
a command-line flag yet.

Storage representation
----------------------

KIVI quantizes keys and values along different axes because their outliers
differ: key outliers concentrate in a few channels, value errors matter per
token.

- Keys are quantized **per channel**: each channel of a group of ``G``
  consecutive tokens shares one FP16 scale and zero point.
- Values are quantized **per token**: each group of ``G`` consecutive channels
  of one token shares one FP16 scale and zero point.
- Codes use round-to-nearest min/max quantization and are packed
  ``8 // bits`` per byte. Scales and zeros saturate at the FP16 range.

The scheduler's paged block table is unchanged: each paged block stores codes
and metadata instead of model-dtype rows, so ``G`` must divide
``runtime.kv_block_size``. The most recent tokens of each sequence stay in
full precision in a per-sequence residual ring owned by the rollout slot.

Residual window
---------------

With residual length ``R`` and group size ``G``, a sequence holding ``n``
tokens stores positions ``[0, q(n))`` quantized and ``[q(n), n)`` in its ring,
where ``q(n) = max(0, floor((n - 1 - R) / G) * G)``. Every new token therefore
attends to at least ``R`` preceding tokens in full precision. The ring holds
``R + G`` entries; when ``n - R`` reaches a positive multiple of ``G`` during
decode, the oldest ``G`` ring tokens are quantized into their paged block
before the new token reuses their ring entries.

Prefill attention over fresh prompt tokens runs at full precision with the
regular attention backend, after which completed groups are quantized. A
prefill chunk that extends an already cached sequence (chunked prefill or a
multi-turn continuation) reads the cached prefix through a bounded per-row
dequantization. Decode runs a fused kernel that dequantizes paged tiles in
registers and never materializes a model-dtype copy of the cache.

Rollout keeps the log-probabilities produced by the KIVI attention that was
actually used for sampling. Training recomputes log-probabilities with the
model-dtype forward as usual, so ``train/logp_abs_diff_mean`` includes the KV
quantization error.

Memory and numerical behavior
-----------------------------

Per token and KV head with a 128-channel head and ``G = 32``, the model-dtype
cache uses 512 bytes for K and V. KIVI-2 stores 96 bytes (64 bytes of codes and
32 bytes of scales and zeros), about 81% less; KIVI-4 stores 160 bytes, about
69% less. Each rollout slot additionally keeps ``R + G`` full-precision tokens
in its ring, which is constant in the context length. The scheduler still
reserves the same number of paged blocks, so the saving appears as lower KV
memory at a fixed concurrency or as headroom for more concurrent sequences or
longer contexts.

Quantization changes the attention scores and outputs, so generated tokens can
differ from a model-dtype run. On Qwen3-0.6B, KIVI-4 kept GSM8K accuracy and
long-context perplexity within noise of the model-dtype cache and trained GSPO
to the same reward, while KIVI-2 with ``R = 128`` lost several points of GSM8K
accuracy and fell 0.1 behind in GSPO reward; a 512-token residual window
recovered its GSM8K accuracy and kept its GSPO reward within 0.04.
``examples/kivi/README.md`` reports these measurements with the commands that
produce them. Validate quality on the target task before switching a workload
to 2-bit.

Configuration
-------------

Engine runtime options (``areno.engine.config.RuntimeConfig``):

``kv_cache_dtype``
   ``"auto"`` (default, model dtype), ``"kivi2"`` or ``"kivi4"``.
``kivi_group_size``
   Tokens per key group and channels per value group, a power of two of at
   least 32. Default 32.
``kivi_residual_length``
   Full-precision recent tokens ``R``. Default 128.

The CUDA SDK forwards its ``runtime`` dictionary to the engine:

.. code-block:: python

   from areno import Trainer
   from areno.api.config import CudaConfig

   trainer = Trainer(
       1,
       "/path/to/Qwen3-0.6B",
       custom_config=CudaConfig(
           runtime={"kv_cache_dtype": "kivi4", "kivi_group_size": 32, "kivi_residual_length": 128},
       ),
   )

``examples/kivi/train_seeded.py`` passes the same options to an otherwise
unchanged ``areno train`` command line.

Supported configurations and fallback
-------------------------------------

KIVI storage covers dense decoder models whose attention layers use the shared
full-causal ``CausalSelfAttention`` module; currently the ``qwen3`` family.
The runtime warns with a ``RuntimeWarning`` and keeps the model-dtype cache
(``kv_cache_dtype="auto"``) for other model families, speculative decoding,
non-CUDA devices, head dimensions that are not a power of two divisible by the
group size, and group sizes that do not divide the KV block size.

When the KIVI cache is allocated, the engine logs one line with its settings
and size, for example
``KIVI KV cache: dtype=kivi4 group_size=32 residual_length=128 blocks=1281 slots=129 bytes=14120386560``.
A run without that line used the model-dtype cache.

Validation
----------

``tests/test_kivi_kv_cache_cpu.py`` covers configuration, fallbacks, residual
bookkeeping and prefill planning. ``tests/test_kivi_kv_cache_cuda.py`` compares
the kernels with an independent PyTorch reference: stored codes and metadata,
decode attention on the same quantized tensors at BF16 tolerance for 1024,
4096 and 8192 cached tokens across residual boundaries, chunked prefill, slot
reuse, CUDA graph replay, storage release on clear and cache size.
``examples/kivi/README.md`` reports memory, decode speed, generation quality,
long-context rollouts and a three-seed GSPO comparison against the
model-dtype cache, with the harnesses in ``examples/kivi/`` that produce them.
