KIVI reference verification and PR #641 semantic checklist
==========================================================

Scope and baseline
------------------

This is an independent, pure PyTorch reference for issue #590, based on commit
``db3dbdf``. It verifies cache semantics on CPU; it is not a runtime integration
and does not contain code from PR #641. The PR-oriented section below is an
acceptance checklist, not a claim that PR #641 currently passes or fails those
checks. No assessment of that implementation is made here.

The tested environment was Python 3.12.7, PyTorch 2.7.1+cpu, with
``torch.cuda.is_available() == False``. Reproduce the regression with:

.. code-block:: console

   C:/Users/AORUS/anaconda3/python -m pytest tests/test_kivi_reference_cpu.py -q

Final result: ``25 passed``.

Reference contract
------------------

Inputs and reconstructed outputs use ``(batch, kv_heads, tokens, head_dim)``.
The reference supports float16, bfloat16, and float32 tensors. It preserves the
input shape, dtype, device, and chronological token order. Autograd is disabled.

For a group with minimum ``mn``, maximum ``mx``, bit width ``B``, and
``Q = 2**B - 1``, quantization is asymmetric and affine:

.. code-block:: text

   scale = (mx - mn) / Q
   safe_scale = scale when scale > 0, otherwise 1
   code = clamp(round((x - mn) / safe_scale), 0, Q)
   reconstructed = code * scale + mn

``scale`` and ``mn`` are saved as float32 tensors. A constant group has zero
scale and zero codes, and reconstructs as ``mn`` without division by zero. An
implementation using a zero point instead of ``mn`` is equivalent only if its
reconstructed affine grid and rounding behavior are the same.

The two layouts are deliberately different:

* **Key, per channel:** transpose token and head-dimension axes, then compute
  independent metadata for each channel over each group of ``group_size``
  tokens.
* **Value, per token:** compute independent metadata for every token over each
  group of ``group_size`` head-dimension channels.

Codes remain unpacked in uint8 storage. This keeps the arithmetic inspectable
and avoids making a packed-storage or memory-savings claim for the reference.

Residual and pending boundaries
-------------------------------

The last ``residual_length`` tokens are always stored exactly in their input
precision. Only complete groups strictly older than that window are quantized.
Because key quantization needs a complete token group, at most
``group_size - 1`` older tokens can remain in a separate full-precision pending
buffer. Pending tokens are not part of the recent residual window.

For buffered length ``N``, residual length ``R``, and group size ``G``:

.. code-block:: text

   stable = max(0, N - R)
   flush = floor(stable / G) * G
   pending = stable - flush

The first ``flush`` tokens are quantized, the next ``pending`` tokens await a
complete group, and the final ``min(N, R)`` tokens form the residual. Global
group boundaries do not depend on prefill/decode update chunking. ``reset()``
drops quantized blocks, scales, minima, pending tokens, residual tokens, token
counts, and tensor metadata while retaining the numeric configuration.

Numerical regression against the BF16 cache baseline
----------------------------------------------------

Here, the BF16 baseline is the original, unquantized BF16 key/value input. It is
not an end-to-end BF16 attention or generation baseline. Before rounding back
to the input dtype, affine round-to-nearest quantization has per-element error
at most ``scale / 2`` for each nonconstant group; constant groups are exact.

The float32 test uses ``atol = max(scale) / 2 + 1e-6`` and ``rtol = 0``. The
BF16 test additionally allows a conservative output-cast term
``torch.finfo(torch.bfloat16).eps * max_abs`` and the same float32 arithmetic
margin. Pending and residual tokens are compared to the BF16 baseline with
zero tolerance. These are cache reconstruction bounds, not bounds on softmax
attention outputs, model logits, or generation quality.

CPU regression coverage
-----------------------

The tests cover:

* key-channel versus value-token scale axes and saved metadata;
* constant groups, quantized shapes, reconstructed shapes, and all three input
  dtypes;
* float32 and BF16 reconstruction error bounds with explicit tolerances;
* an empty cache, zero-sized/one-token/full residuals, and window overflow;
* pending-group drainage and one-shot versus chunked update equivalence;
* token ordering, cache reset, and reuse with different tensor metadata;
* invalid rank, shape, dtype, token count, nonfinite values, configuration, and
  partial groups;
* explicit rejection of unsupported attention types rather than BF16 fallback.

Semantics that must agree with PR #641 integration
---------------------------------------------------

Under the same bit width, group size, residual length, input values, and update
boundaries, the runtime/reference comparison must establish all of these:

#. Key metadata is channel-wise over tokens; value metadata is token-wise over
   head-dimension channels. Neither axis may be accidentally transposed.
#. Quantization is affine with the same rounding/clamping and constant-group
   behavior. Scale plus minimum and an equivalent scale plus zero-point format
   must reconstruct the same values.
#. Every quantized element maps to the correct saved scale and affine offset,
   including GQA/MQA KV-head indexing and the final group.
#. The residual contains exactly the most recent configured number of tokens,
   in original precision and chronological order. It must be included in
   attention before and after the empty/full/overflow transitions.
#. Quantization never consumes residual tokens. Incomplete key groups remain
   representable and are eventually quantized without reordering or duplication.
   Prefill/decode chunking must not change the resulting quantized groups.
#. Materialized/reconstructed KV has the original batch, KV-head, token, and
   head-dimension dimensions and the original compute dtype. Cache lengths used
   by attention include quantized, pending, and residual tokens exactly once.
#. Reset/release clears every quantized and full-precision buffer and all
   associated lengths/metadata. Reuse cannot expose tokens from the prior state.
#. Unsupported attention families fail explicitly. They must not silently use
   an unquantized BF16 path while reporting KIVI as enabled.

The comparison should first inspect codes/scales/minima and reconstructed KV,
then compare attention outputs. Matching final attention outputs alone would
not localize an axis, scale-index, residual-order, or reset defect.

Not verified on CPU
-------------------

**This validation was performed on CPU and does not cover actual CUDA memory
savings.** In particular, it does not establish:

* numerical parity with PR #641 or with any CUDA quantization/dequantization
  kernel;
* real bit-packing, packed-layout indexing, device memory allocation, peak
  memory, memory savings, throughput, or latency;
* paged-attention/block-table/cache-position wiring, CUDA graphs, multi-layer
  execution, concurrent requests, or runtime reset/release integration;
* FP16/BF16 CUDA behavior, long-context behavior, attention logits, generated
  tokens, perplexity, or task quality;
* production storage overhead from scales, pending tokens, or residual tokens.

Those items require a GPU integration run and comparisons against PR #641 under
identical inputs and configurations. No CUDA-only assertion is disguised as a
passing CPU test in this regression.
