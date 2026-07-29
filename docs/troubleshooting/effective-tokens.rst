:orphan:

Effective Trainable Tokens
==========================

AReno records per-update token statistics so operators can see how many
tokens are actually being trained on, not just how many are in the batch.

Metrics
-------

The following four metrics are emitted with every training update and are
visible in logs, TensorBoard, and the dashboard:

* ``total_input_tokens`` — all valid tokens in the batch (excluding padding).
* ``masked_tokens`` — tokens excluded from loss (prompt positions, padding,
  or explicitly masked by ``loss_mask``).
* ``effective_loss_tokens`` — tokens that contribute to the loss gradient.
* ``mean_effective_length`` — average effective (response) tokens per sequence.

How it works
------------

The counting uses the same ``prompt_mask`` and ``loss_mask`` consumed by the
loss function, so the numbers are consistent with what the model actually
trains on. Both packed (varlen) and padded (rectangular) layouts are
supported.

For agentic trajectories where prompt and response tokens are interleaved,
the counting correctly handles mixed positions — any position marked as
prompt in ``prompt_mask`` is counted as masked, regardless of its position
in the sequence.

Zero-token batches
------------------

If a batch contains sequences with zero valid tokens (e.g., all sequences
were filtered out), the metrics will report zeros for all fields without
crashing. ``mean_effective_length`` will be 0.0.

Limitations
-----------

* The metrics are computed per-update (per optimizer step), not per-sample.
* Cross-rank aggregation uses the existing ``merge_train_stats`` mechanism,
  which averages metrics across data-parallel ranks.
* The counting happens in the trainer's ``_train_step``, so it covers SFT,
  DPO, and rollout-based training (GSPO/GRPO/PPO) uniformly.
