Shared-prompt KV verification
=============================

Scope
-----

``areno/engine/runtime/shared_kv_reference.py`` is an executable CPU
specification for issue #592.  It deliberately imports neither the production
scheduler nor an attention backend.  The independent full-sequence forward is
the numerical oracle; the alternative path prefills one immutable prompt KV
prefix and appends a private KV suffix for each sample.

The reference and tests were added on baseline ``db3dbdf``.  This baseline did
not contain PR #639, so these results are **not** a claim that the PR's CUDA
implementation has passed integration or numerical validation.

CPU numerical validation
------------------------

The model has two randomly initialized Transformer layers, hidden size 32,
four heads, vocabulary size 64, learned positions, and FP32 parameters.  No
checkpoint or network access is needed.  Run:

.. code-block:: console

   python -m pytest tests/test_shared_prompt_kv_reference_cpu.py -q -s

On torch 2.7.1+cpu, all seven cases passed.  Across the comparisons in that
run, the maximum absolute deviation was ``3.57627869e-07`` for logits and
``4.76837158e-07`` for behavior log-probs.  The assertion tolerance is
``atol=1e-6`` and ``rtol=1e-6``.  This small FP32 budget covers reduction-order
noise between full and incremental attention; cache contents and token
sequences use exact assertions instead.  These tolerances must not be copied
to FP16/BF16 CUDA results without a dtype-specific baseline analysis.

The test matrix covers:

* **Prefix consistency:** all prefill logits match an independent full prompt
  forward, and every branch references the same prefix tuple.
* **Fork isolation:** three branches decode different tokens in interleaved
  order.  Every step is compared with a full forward of exactly that branch's
  token sequence.  The shared prefix remains bit-wise unchanged.
* **Unequal lengths and N=1:** branches have zero through four private tokens;
  each layer's private KV length must equal its branch token count.  A separate
  degenerate case uses one branch.
* **Capacity boundary:** a token beyond ``max_seq_len`` is rejected without
  changing shared or private KV, while another branch can still decode.
* **Reset/reuse:** reset invalidates old branches and drops prompt, prefix, and
  suffix references.  Reusing the owner with a different same-length prompt
  matches a fresh full forward and cannot mutate the retired prefix.
* **Behavior log-probs:** for each generated token, the shared path's predictor
  logits are scored with log-softmax and compared with AReno's training-side
  ``next_token_logprobs`` over the independent full sequence.

CUDA integration correctness handoff
------------------------------------

Before measuring performance, maintainers should run equivalent assertions on
the actual PR #639 path with the same weights, prompts, forced continuations,
TP topology, dtype, and attention backend.  Compare prefill and every decode
step against independent non-shared execution.  Repeat with ``N=1``, divergent
lengths, interleaved scheduling, prefix release/reuse, and batches mixing
shared and unshared requests.  A final sampled-text comparison alone cannot
localize a KV ownership or position error.

Performance evaluation method
-----------------------------

No CUDA performance was measured in this CPU-only task.  For a meaningful
comparison, use two builds differing only in shared-prompt KV support and run
each configuration in a fresh process after correctness passes.

1. Fix model/checkpoint, tokenizer, dtype, TP size, batch composition, attention
   backend, CUDA-graph setting, prompt set, and generated token count.  Force
   the same continuation tokens in baseline and candidate runs so timing does
   not depend on sampling divergence.
2. Sweep samples per prompt ``N = 1, 2, 4, 8`` and several short/long prompts
   that fit the deployment context limit.  Include mixed prompt lengths and
   unequal generation lengths.  ``N=1`` is the overhead guard, not a speedup
   case.
3. Baseline performs ``N`` independent prompt prefills.  Candidate performs one
   prompt prefill followed by ``N`` isolated continuations.  Warm up each
   configuration, then record at least 20 timed iterations (or enough to reach
   a stable median); retain raw samples, median, and p95 rather than only a
   best-case number.
4. Use CUDA events around prefill and decode regions, with synchronization at
   measurement boundaries.  Report prompt prefill latency/TTFT, decode TPOT,
   end-to-end latency, and generated tokens/s separately.
5. After warmup, call ``torch.cuda.reset_peak_memory_stats()`` and record peak
   allocated and reserved bytes for the same measured region.  Also report
   allocator block size and cache capacity: page rounding and fragmentation can
   dominate the theoretical ``(N-1)`` prompt-KV saving.
6. Keep timing and memory runs separate from diagnostic logging.  Publish GPU
   model, driver/CUDA versions, exact commands, raw results, and any rejected
   configurations.

Conclusion boundary
-------------------

The CPU results establish the reference semantics only: numerical agreement,
branch isolation, clean lifecycle behavior, and preserved behavior log-probs.
They provide no evidence of CUDA kernel correctness, latency improvement,
throughput improvement, or memory reduction.  Those claims require the actual
PR implementation and the controlled CUDA measurements above.
