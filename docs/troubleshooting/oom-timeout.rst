:orphan:

Training OOM and timeout
========================

Out-of-memory and timeout failures usually come from rollout volume, sequence
length, tensor parallelism, model size, or slow external agent work.

First reductions:

* Lower ``--batch-size``.
* Lower rollout or sequence length settings.
* Use a smaller checkpoint for the first reproduction.
* Reduce agent concurrency for tool or environment tasks.
* Confirm no unrelated GPU process is consuming memory.

For agentic tasks, distinguish model time from environment time. Tool calls,
tests, sleeps, browser work, and sandbox actions can dominate rollout wall
time before the model is called again.

See :doc:`/cli/observability` for timing and metric interpretation.

Stage-specific OOM guidance
---------------------------

When a CUDA out-of-memory error occurs, AReno automatically detects which
stage the error happened in and prints actionable suggestions to stderr.  The
original error is re-raised unchanged and is never replaced or hidden.

Three stages are recognised:

1. **Model loading** — OOM during weight construction or checkpoint loading.
2. **Rollout generation** — OOM during inference, KV-cache allocation, or
   decode CUDA graph capture.
3. **Training** — OOM during forward, backward, or optimizer stepping.

The guidance output looks like::

   CUDA OOM during training. Suggestions (in priority order):

     1. Reduce --mini-bs (currently 16) to shrink the training microbatch ...
        Option: --mini-bs  (current value: 16)
     2. Add --drop-rollout-state to release rollout state before training ...
        Option: --drop-rollout-state  (current value: False)
     ...

   See https://github.com/inclusionAI/AReno/blob/main/docs/cli/training.rst#troubleshooting-oom
   for detailed OOM troubleshooting.

Suggestions are ordered by priority and only include options relevant to the
failing stage.  Each suggestion shows the AReno option name and the resolved
value currently in effect so the user knows exactly what to change.

Suggestion summary by stage:

**Model loading**

* Increase ``--tp-size`` to shard the model across more GPUs.
* Inspect competing GPU processes with ``nvidia-smi`` and stop stale jobs you
  own.
* Increase ``--tp-size`` to the next divisor of ``--world-size`` when one is
  available.
* Try ``--attn-backend native`` if flash-attn workspace allocations
  contribute (slower but lower peak workspace memory).

**Rollout generation**

* Reduce ``--max-running-prompts`` to lower concurrent decode memory and
  KV-cache footprint.
* Reduce ``--batch-size`` or ``--n-samples`` to lower total concurrent
  rollout sequences.
* Add ``--eager-decode`` to disable decode CUDA graph capture.
* Increase ``--tp-size`` to shard KV-cache across more GPUs.

**Training**

* Reduce ``--mini-bs`` to shrink the training microbatch.
* Enable ``--activation-checkpointing`` to trade compute for memory.
* Add ``--drop-rollout-state`` to free GPU memory before backward pass.
* Enable ``--adam-8bit`` to reduce optimizer memory.
* Increase ``--gradient-accumulation-steps`` and further reduce
  ``--mini-bs``.
* Increase ``--tp-size`` to shard gradients and optimizer states.

The guidance is purely informational.  AReno does not automatically mutate
configuration or retry the run.  When the OOM stage cannot be determined
(e.g., ambiguous traceback), no guidance is emitted and the original error
passes through unchanged, preserving backward compatibility.

For programmatic access, the :mod:`areno.engine.oom_diagnostics` module
exposes ``build_oom_guidance()``, ``format_oom_guidance()``,
``is_oom_error()``, and ``oom_stage()`` with both human-readable and structured
(``OOMGuidance.to_dict()``) output.
