Activation compression reference verification
=============================================

Scope
-----

``areno.accel.activation_compress_reference`` is an independent, pure-PyTorch
reference for issue #591 boundary testing. It is not connected to the training
runtime and does not modify or execute the PR #646 integration. The reference
owns its saved tensors, supports CPU validation without a CUDA extension, and
makes storage and reconstruction accounting reproducible from public metadata.

Format and numerical contract
-----------------------------

The input is logically contiguous and flattened. Strides are not preserved;
shape, dtype, device, and logical values are. Supported input dtypes are FP16,
BF16, FP32, and FP64. Scalar and empty shapes are valid. The format supports
2-, 4-, and 8-bit signed symmetric quantization represented with offset-binary
codes.

For ``N`` values, block size ``B``, and bit width ``W``:

.. math::

   Q = 2^{W-1}-1,\quad S = \lceil N/B\rceil,\quad P = SB

Each block uses ``absmax / Q`` as its scale, with zero and subnormal scales
handled explicitly. Scales are FP32 for FP16/BF16/FP32 inputs and FP64 for FP64
inputs. Codes are rounded to ``[-Q, Q]``, shifted by ``Q``, and packed into
consecutive low-order bit groups of bytes. Only ``N`` codes are significant;
bit padding at the end of the final byte is not reconstructed as an activation.

Ignoring arithmetic rounding, a reconstructed value differs from its source by
at most half a quantization interval, ``scale / 2``. Tests independently add
conservative allowances for computation and final output casting:

* eight times ``eps_compute * max(abs(source))``;
* for FP16/BF16, ``eps_output * max(abs(source)) + tiny_output``;
* ``tiny_compute`` for the zero and subnormal boundary.

These bounds validate this reference format; they are not claimed for a
production format with a different scale or packing rule.

Fallback and release semantics
------------------------------

``max_abs_error`` is an inclusive acceptance budget: compression remains in use
when observed error equals the budget and falls back only when error is strictly
greater. NaN or infinite inputs always take the uncompressed path. A fallback
payload discards packed codes and scales and owns a clone of the original input.
Restoration therefore matches the original bits, including NaN and infinity.

Both compressed and fallback payloads own the data needed for restoration.
Tests mutate and release source variables, restore two payloads in interleaved
order, and mutate a returned tensor to check that later restorations are neither
aliased nor contaminated by another checkpoint.

Accounting formulas
-------------------

The following formulas are asserted independently in the tests:

============================  ===============================================
Field                         Compressed payload
============================  ===============================================
``scale_count``               ``S``
``packed_nbytes``             ``ceil(N * W / 8)``
``scale_nbytes``              ``S * sizeof(scale_dtype)``
``raw_nbytes``                ``0``
``source_nbytes``             ``N * sizeof(input_dtype)``
``storage_nbytes``            packed + scales + raw
reconstruction workspace      ``N + P * sizeof(scale_dtype)`` bytes
============================  ===============================================

For fallback, packed bytes, scale bytes, scale count, and reconstruction
workspace are zero; raw bytes and persistent storage equal ``source_nbytes``.
Workspace accounting covers the explicit unpacked-code buffer (one byte per
value) and padded scale-dtype reconstruction buffer. It excludes the returned
output, per-chunk arithmetic temporaries, Python objects, and allocator overhead,
so it is logical workspace accounting rather than peak RSS or VRAM.

Cross-configuration invariants
------------------------------

``chunk_size`` is rounded up to a whole number of blocks and changes traversal
only. With fixed input, bit width, and block size, packed codes, scales, observed
error, and restored bits must be identical across chunk sizes.

Changing ``block_size`` changes scale membership, so identical reconstructed
values are not required. Every block-size configuration must instead preserve
shape and dtype and satisfy its own independently calculated error bound.

Reproduction and limits
-----------------------

Run the focused suite with::

   python -m pytest tests/test_activation_compress_reference_cpu.py -q

The CUDA round-trip test is guarded by ``skipif(not torch.cuda.is_available())``
and reports that it needs a GPU host. It is skipped in the documented
``torch 2.7.1+cpu`` environment. No CUDA result is inferred from CPU success.

This suite does not validate PR #646 call sites, fused-kernel parity, autograd or
backward behavior, distributed/TP layouts, CUDA graph capture, end-to-end
training, or throughput and memory savings. Those require separate integration
and GPU validation against the mainline implementation.
