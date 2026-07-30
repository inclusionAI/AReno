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

Reward hook timing
------------------

AReno can measure each reward function invocation, flag slow samples as
outliers, and enforce an optional per-sample timeout. This feature is
**disabled by default** and adds near-zero overhead when not enabled.

Enable it with three CLI flags:

.. code-block:: bash

   areno train --algo gspo --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main \
     --reward-fn-path examples/math/math_verify_reward.py \
     --reward-timing-enabled \
     --reward-slow-threshold-s 0.5 \
     --reward-timeout-s 10.0

Input contract
~~~~~~~~~~~~~~

``--reward-timing-enabled``
    Master switch. When omitted, reward timing is disabled and the reward
    function is called directly with no overhead.

``--reward-slow-threshold-s SECONDS``
    Samples whose reward computation takes longer than this are flagged as
    outliers in the log and the timing report. Must be positive. Set to
    ``None`` (the default) to disable outlier flagging.

``--reward-timeout-s SECONDS``
    Per-sample wall-clock timeout. Samples exceeding this receive ``NaN``
    as their reward and are listed in the report's ``timeouts`` field. Must
    be positive and >= ``--reward-slow-threshold-s`` when both are set.
    Timeout enforcement uses ``signal.SIGALRM`` and is only effective on
    POSIX platforms; on Windows the timeout is silently ignored.

Defaults and validation
~~~~~~~~~~~~~~~~~~~~~~~

All three options default to values that preserve current behavior
(timing disabled). Invalid combinations -- such as a negative threshold or
``timeout_s < slow_threshold_s`` -- produce a clear ``click.UsageError``
before any model or worker initialization.

Observable output
~~~~~~~~~~~~~~~~~

When enabled, the trainer produces two output channels per training step:

1. **Console logs** -- two log levels are emitted:

   .. code-block:: text

      WARNING reward_slow hook=reward_fn sample=p2_s5 elapsed=0.6231s threshold=0.5s
      INFO reward_timing hook=reward_fn step=3 n=32 total=4.213s mean=0.132s max=0.623s p95=0.401s slow_samples=[p2_s5,p7_s1]

   The ``sample`` identifier is a short opaque tag (``p{prompt_index}_s{sample_index}``)
   that distinguishes samples **without exposing prompt or completion text**.

2. **Dashboard state** -- the timing report is persisted as structured JSON
   via ``record_dashboard_state(stage="reward_timing", ...)`` with fields:
   ``hook_name``, ``step``, ``num_samples``, ``total_elapsed_s``,
   ``mean_elapsed_s``, ``max_elapsed_s``, ``p95_elapsed_s``, ``outliers``
   (list of ``{sample_id, elapsed_s, timed_out}``), and ``timeouts``
   (list of ``{sample_id, elapsed_s}``).

Limitations
~~~~~~~~~~~

- Timeout enforcement is POSIX-only (``SIGALRM``).
- ``NaN`` rewards from timeouts propagate through ``compute_group_advantages``
  and will produce ``NaN`` advantages; filter or replace them downstream if
  needed.
- Timing overhead is one ``time.perf_counter()`` call per sample when
  enabled; it is not collected at all when disabled.

Copyable example
~~~~~~~~~~~~~~~~

.. code-block:: python

   from areno.api.reward_timing import RewardTimingConfig, TimedRewardFn
   from areno.api.rewards import RewardRecord

   def my_reward_fn(record: RewardRecord) -> float:
       return 1.0 if "correct" in record.completion else 0.0

   config = RewardTimingConfig(
       enabled=True,
       slow_threshold_s=0.1,
       timeout_s=5.0,
       hook_name="my_reward",
   )
   timed = TimedRewardFn(my_reward_fn, config)

   record = RewardRecord(prompt="2+2=?", completion="4", metadata={"prompt_index": 0, "sample_index": 0})
   score = timed(record)
   report = timed.finalize_batch(step=0)
   if report is not None:
       print(report.format_human())
       # reward_timing hook=my_reward step=0 n=1 total=0.000012s mean=0.000012s max=0.000012s p95=0.000012s
