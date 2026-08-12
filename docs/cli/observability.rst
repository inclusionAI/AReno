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

Trajectory observability contract
---------------------------------

One logical trajectory is one prompt/sample pair, even when the agent makes
multiple model requests. Agent code returns an ``AgentTrajectory`` containing
one or more ``AgentTrajectoryTurn`` objects. AReno merges turns that carry the
same ``AgentItem`` and exposes the completed trajectory to the reward function
as one ``RewardRecord``. This is the v0.0.3 observability boundary; it defines
what a debugger or future exporter should be able to inspect, but it is not a
versioned serialization schema.

The completed trajectory has these logical inspection fields:

.. list-table::
   :header-rows: 1
   :widths: 22 33 45

   * - Information
     - Current source
     - Meaning
   * - Identity and source
     - ``RewardRecord.metadata`` and ``source_record``
     - ``prompt_index`` and ``sample_index`` identify the expanded
       prompt/sample pair; ``source_record`` retains its dataset record.
   * - Messages
     - ``RewardRecord.messages``
     - Full OpenAI-style conversation, including the final assistant message
       and tool-role result messages.
   * - Assistant responses
     - ``completion``, ``final_answer``, and ``rendered_completion``
     - Concatenated generated assistant spans, the last assistant response,
       and the tokenizer chat-template rendering respectively.
   * - Tool calls and results
     - ``tool_calls`` and ``tool_results``
     - Parsed function names/arguments and environment observations. The
       message list remains the source for their conversation ordering.
   * - Response tokens and logprobs
     - ``tokens`` and ``logprobs``
     - Aligned model-generated response spans across all merged turns. Prompt
       and tool-context token rows are internal training data, not part of
       these response-only arrays.
   * - Reward input and score
     - ``RewardRecord`` plus the reward function result
     - ``prompt``, ``answer``, ``source_record``, and the trajectory fields are
       the reward input. The scalar reward is computed immediately afterward
       and stored on the train batch; it is not a field of ``RewardRecord``.
   * - Loss-mask spans
     - ``RewardRecord.loss_mask`` and ``AgentTrainBatch.loss_masks``
     - ``RewardRecord.loss_mask`` aligns with response-only ``tokens``.
       ``AgentTrainBatch.loss_masks`` aligns with the complete training token
       row, with context and tool-result spans masked by default.
   * - Trace events
     - ``RewardRecord.trace``
     - Ordered normalized ``request``, ``assistant_text``,
       ``assistant_tool_call``, ``tool_result``, ``finish``, and ``error``
       events. Events carry only fields relevant to their type.
   * - Filtered reason
     - ``agentic trajectory filtered`` warning
     - Today the only post-trajectory filter is an overlong complete training
       row. The warning reports the context limit, counts, token-length
       distribution, trajectory identity, message/tool/trace counts, and a
       bounded prompt preview. Filtered trajectories do not produce a
       ``RewardRecord`` or scalar reward.

The contract is exercised by ``tests/test_agentic_cpu.py``. In particular,
``test_agent_trajectory_turn_extracts_response_metadata`` covers response
tokens/logprobs, ``test_agentic_tool_request_returns_tool_call_and_reward_record``
covers tool and reward-record fields,
``test_agentic_multi_turn_calls_merge_into_one_training_sample`` covers merged
messages, results, masks, and trace events, and
``test_agentic_overlong_trajectory_is_filtered_with_warning`` covers filtered
diagnostics.

Artifact decision
~~~~~~~~~~~~~~~~~

v0.0.3 documents the in-memory and log contract only. It does not add a full
trajectory artifact writer. When metrics recording and
``ARENO_LOG_COMPLETIONS`` are enabled, ``rollout_samples.<pid>.jsonl`` contains
a bounded diagnostic projection: prompt, messages, final answer, tool calls,
up to four tool results, and prefixes/summaries of the training token and mask
rows. It intentionally omits the complete response token/logprob arrays,
reward record, scalar reward, trace, and filtered trajectories, so consumers
must not treat it as the canonical trajectory format. A later artifact issue
should define schema versioning, full-versus-redacted fields, retention, and
multi-process file ownership before adding such a writer.

Privacy and safety
~~~~~~~~~~~~~~~~~~

Trajectory diagnostics can contain prompts, dataset records, model responses,
tool arguments, tool outputs, local paths, source code, credentials returned by
tools, and other sensitive or untrusted text. Set
``ARENO_LOG_COMPLETIONS=0`` when sample diagnostics are not needed, restrict
access to ``metrics_log_dir``, set a short retention period, and redact secrets
and personal data before sharing logs. Treat logged tool output as data, never
as instructions to execute. The bounded sample count and truncated
token/result fields reduce volume; they are not a privacy or secret-redaction
boundary.
