Profiling and optimization
==========================

AReno exposes a small profiler for measuring model regions without requiring
extra dependencies.  It records wall-clock latency on every device and, for
CUDA/NPU regions, synchronizes the device and records peak and final allocated
memory::

   from areno.engine.profiling import profile_region

   with profile_region("audio-encoder", device=audio_features.device) as sample:
       audio_embeds = audio_encoder(audio_features, audio_mask)
   print(sample.elapsed_ms, sample.peak_memory_bytes)

Use a warm-up iteration before collecting measurements.  For CUDA/NPU, pass
the same device that owns the tensors; synchronization is intentionally limited
to accelerator regions so CPU tests and CPU fallback paths stay lightweight.

For comparisons, use repeated measurements rather than a single region. The
benchmark helper performs warm-up and reports median, p95, minimum, maximum,
and peak allocated memory::

   from areno.engine.profiling import benchmark_callable

   result = benchmark_callable(
       "audio-forward",
       lambda: audio_encoder(audio_features, audio_mask),
       device=audio_features.device,
       warmup=3,
       iterations=20,
   )
   print(result.median_ms, result.p95_ms, result.peak_memory_bytes)

The Phi-4 multimodal audio feed-forward and convolution paths use the existing
fused SiLU kernel on CUDA/NPU. The Conformer attention path uses PyTorch SDPA,
which avoids separate score/softmax intermediates and lets the active backend
select an optimized attention implementation. CPU execution keeps the
reference SiLU expression, so correctness tests and environments without the
native extension remain independent of accelerator installation.

Optimization validation
-----------------------

The SDPA change was measured on an RTX 3090 with PyTorch 2.9.1, CUDA 12.8,
FP16, batch size 1, hidden size 1024, and 16 heads. These numbers are a
regression reference rather than a guarantee for other hardware:

.. list-table::
   :header-rows: 1
   :widths: 28 18 18 18

   * - Region
     - Implementation
     - Median latency
     - Peak allocated memory
   * - Inference, 500 frames
     - explicit softmax
     - 0.564 ms
     - 45.88 MiB
   * - Inference, 500 frames
     - SDPA
     - 0.475 ms
     - 37.34 MiB
   * - Forward/backward, 256 frames
     - explicit softmax
     - 2.064 ms
     - 43.27 MiB
   * - Forward/backward, 256 frames
     - SDPA
     - 1.906 ms
     - 37.27 MiB

Re-run the benchmark on the deployment GPU because attention backend choice,
driver version, sequence length, and dtype can change the result.
