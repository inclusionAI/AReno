:orphan:

Observability
=============

AReno exposes training state through plain logs, the per-step ``train_stats``
dictionary, TensorBoard scalar metrics, and optional agentic trajectory
diagnostics. The implementation is intentionally local and lightweight:
metrics are recorded by ``areno.api.metrics`` and the CLI exposes the output
directory through ``--metrics-log-dir``. AReno does not currently provide a
built-in wandb integration.

Console logs
------------

The trainer logs a compact lifecycle for every epoch and step. For rollout
algorithms, ``areno.api.trainers.policy_only.PolicyOnlyTrainer`` emits:

* ``epoch=<n> stage=epoch_start`` and ``epoch=<n> stage=epoch_end``.
* ``role=policy stage=rollout_start`` and ``stage=rollout_end`` around
  sampling or agentic execution.
* ``metric=reward_mean`` after rewards are computed.
* ``metric=rollout_logprob_mean`` when rollout logprobs are available.
* ``role=policy stage=train_start`` and ``stage=train_end`` around the
  optimizer step.
* ``train_stats={...}`` with the scalar dictionary returned by the backend and
  loss function.

The local AReno backend also logs step timing from
``areno.api.backend.areno.backend``:

.. code-block:: text

   time rollout=213.929721 train=17.327612 total=231.297132

Those numbers mean the backend measured roughly 214 seconds in rollout work,
17 seconds in the training step, and 231 seconds end-to-end for that step. The
same values are copied into ``train_stats`` as ``step_rollout_time_s``,
``step_train_time_s``, and ``step_e2e_time_s``.

During rollout, ``areno.engine.inference`` logs decode progress per data
parallel shard:

.. code-block:: text

   rollout decode progress: dp=0/2 active=6 cuda_graph=True tokens_per_second=60.8

``active`` is the number of currently scheduled decode requests on that shard
at the time of the progress log. In agentic runs it can dip to zero while the
external agent code is executing tools, tests, sleeps, background commands, or
other non-model work before asking the OpenAI-compatible proxy for the next
model response.

``train_stats``
---------------

``train_stats`` is the easiest place to inspect one completed optimizer step.
It is logged as a Python dictionary and also passed to TensorBoard by
``areno.api.metrics.MetricsRecorder`` when metrics recording is enabled.

Common fields include:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Field
     - Meaning
   * - ``loss`` / ``policy_loss`` / ``total_loss``
     - Loss scalars returned by the selected loss function.
   * - ``advantage_mean``
     - Mean policy advantage over trainable response tokens.
   * - ``response_len``
     - Mean response-token count for rows used in the step.
   * - ``rollout_logprobs_mean``
     - Mean rollout-time logprob over trainable response tokens.
   * - ``train_logprobs_mean``
     - Mean current-policy logprob during the training forward pass.
   * - ``logp_diff_mean`` / ``logp_abs_diff_mean``
     - Difference between rollout and train logprobs, useful for stale-policy
       or masking debugging.
   * - ``ratio_mean`` / ``ratio_std``
     - Policy-ratio diagnostics used by rollout policy losses such as GSPO,
       GRPO, and PPO.
   * - ``grad_norm``
     - Global gradient norm after clipping/accounting.
   * - ``grad_zero_ratio`` / ``grad_nonzero_ratio``
     - Fraction of parameter-gradient entries that are zero/non-zero.
   * - ``lr``
     - Current optimizer learning rate.
   * - ``step_rollout_time_s`` / ``step_train_time_s`` / ``step_e2e_time_s``
     - Per-step wall-clock timing from the local backend.

For example, if a debugging log shows:

.. code-block:: text

   metric=reward_mean value=0.125000
   train_stats={'loss': 0.0, 'advantage_mean': 0.0, 'response_len': 912.5625,
                'rollout_logprobs_mean': -0.17456013709306717,
                'train_logprobs_mean': -0.2500366196036339,
                'logp_diff_mean': 0.07547648251056671,
                'step_rollout_time_s': 213.92972119152546,
                'step_train_time_s': 17.327912725508213}

read it as: the sampled batch achieved low positive reward on average, the
training rows were long, rollout dominated wall time, and the current policy
assigned lower logprob than the rollout policy on average. ``loss`` can print
as ``0.0`` for some policy-gradient batches when normalized advantages cancel
in the scalar value at the current ratio, while gradients can still be non-zero
because the derivative depends on the logprob term.

TensorBoard metrics
-------------------

Pass ``--metrics-log-dir`` to control where TensorBoard event files are written.
The default is shown by ``areno train --help`` and is also surfaced in the
training config printout.

.. code-block:: bash

   areno train ... --metrics-log-dir /tmp/areno/tfevent
   tensorboard --logdir /tmp/areno/tfevent

The writer lives in ``areno.api.metrics``. It records three namespaces:

``rollout/*``
   Sample-side statistics computed from the train batch, including
   ``rollout/rewards_mean``, ``rollout/rewards_std``,
   ``rollout/rewards_max``, ``rollout/rewards_min``,
   ``rollout/accuracy``, ``rollout/advantages_mean``,
   ``rollout/advantages_std``, ``rollout/logprobs_mean``,
   ``rollout/seq_len_mean``, ``rollout/prompt_len_mean``,
   ``rollout/response_len_mean``, ``rollout/num_sequences``,
   ``rollout/skipped_long``, and ``rollout/total_skipped_long``.

``train/*``
   Every scalar returned in ``train_stats``. Typical examples are
   ``train/loss``, ``train/policy_loss``, ``train/total_loss``,
   ``train/ratio_mean``, ``train/ratio_std``, ``train/grad_norm``,
   ``train/lr``, ``train/rollout_logprobs_mean``, and
   ``train/train_logprobs_mean``.

``time/*``
   Stage timings when available: ``time/rollout``, ``time/reward``,
   ``time/advantage``, and ``time/train``.

GPU history
-----------

Pass ``--gpu-stats`` to sample the GPUs used by a CLI training run without
adding work to rollout or optimizer steps:

.. code-block:: bash

   areno train \
     --algo sft \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path tatsu-lab/alpaca \
     --model-hub hf \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --world-size 1 \
     --tp-size 1 \
     --batch-size 1 \
     --mini-bs 1 \
     --max-prompt-tokens 128 \
     --max-new-tokens 64 \
     --attn-backend native \
     --max-steps 2 \
     --gpu-stats \
     --gpu-stats-interval-s 1 \
     --gpu-stats-history 120 \
     --metrics-log-dir /tmp/areno/gpu-check

The feature is disabled by default. ``--gpu-stats-interval-s`` must be
positive and defaults to ``5.0``. ``--gpu-stats-history`` must be positive,
defaults to ``1000``, and bounds both the in-memory history and the JSONL
snapshot. With multiple visible GPUs, each tick contributes one sample per
device, so the bound is a total row count rather than a per-device count.

AReno follows ``CUDA_VISIBLE_DEVICES`` order. For example,
``CUDA_VISIBLE_DEVICES=3,1`` maps logical ``device 0`` to physical GPU 3 and
logical ``device 1`` to physical GPU 1. GPU UUID selectors are also supported.
The summary preserves logical index, physical index, UUID, and model name.
MIG UUIDs depend on whether the installed driver exposes them through
``nvidia-smi --query-gpu``; unsupported selectors produce no matching sample
and are visible in ``device_selectors`` in the summary.

Two per-process artifacts are refreshed under ``--metrics-log-dir``:

``gpu_stats.<pid>.jsonl``
   A bounded, atomically replaced snapshot with one JSON object per retained
   device sample. Fields are ``timestamp_s`` (Unix time), logical ``index``,
   ``physical_index``, ``uuid``, ``name``, ``mem_used_mb``,
   ``mem_total_mb``, ``util_pct``, and ``temp_c``.

``gpu_stats_summary.<pid>.json``
   Run metadata and per-device aggregates including peak used memory, total
   memory, mean utilization, maximum temperature, sample count, device
   mapping, duration, and any sampling failure stage/message.

The command also prints a human-readable summary when training exits. Missing
``nvidia-smi``, query failures, shutdown timeouts, and artifact write failures
degrade to a diagnostic warning; they do not replace the original training
exception.

For a deterministic CPU-only boundary check, run:

.. code-block:: bash

   python -m pytest \
     tests/test_gpu_stats_cpu.py::test_cli_lifecycle_writes_bounded_artifacts_with_fake_sampler \
     tests/test_train_cli_config_cpu.py::test_train_config_validates_common_positive_fields

The first test exercises the successful CLI lifecycle with two fake GPUs and
bounded local artifacts. The second suite includes invalid zero values for
both GPU-stat bounds.

Agentic diagnostics
-------------------

Agentic rollout adds two diagnostic surfaces.

First, the trainer logs batch-level information before and after agent
execution:

.. code-block:: text

   agentic rollout batch prompts=2 n_samples=8 expected_requests=16 max_running_prompts=16
   agentic train batch built samples=16 tokens=223308 messages=242 tool_calls=133 tool_results=97

These lines are useful for checking whether the configured concurrency,
trajectory length, and tool-call volume match expectations.

Second, set ``ARENO_LOG_COMPLETIONS`` to a positive integer to log a bounded
number of prompt/sample trajectories. For agentic rollouts this includes the
rendered prompt, message list, final answer, parsed tool calls, sampled tool
results, token row prefix, and loss-mask summary:

.. code-block:: bash

   ARENO_LOG_COMPLETIONS=2 areno train ...

When a trajectory is dropped for exceeding the model context window,
``PolicyOnlyTrainer`` logs ``agentic trajectory filtered: ...`` with token
counts, message counts, assistant turn counts, tool-result counts, and a short
prompt preview. This is the fastest way to debug overlong agentic examples
without dumping every token in every trajectory.
