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

The metrics are computed in ``TrainingManager._train_step`` after packing
and before the loss, then merged into the per-step metrics dict that flows
through ``merge_train_stats`` for cross-rank aggregation. This means SFT,
DPO, and rollout-based training (GSPO/GRPO/PPO) all get the metrics
automatically — they all go through ``_train_step``.

Example
-------

.. code-block:: python

   from areno.engine.token_counts import compute_token_counts_from_padded

   # One sequence: [P, P, P, R, R] (3 prompt, 2 response tokens)
   # Actions at positions 1-4: pos1=P(masked), pos2=P(masked), pos3-4=R(effective)
   counts = compute_token_counts_from_padded(
       lengths=[5],
       prompt_mask_rows=[[True, True, True, False, False]],
   )
   print(counts.to_dict())
   # {'total_input_tokens': 5.0, 'masked_tokens': 2.0,
   #  'effective_loss_tokens': 2.0, 'mean_effective_length': 2.0}

For agentic trajectories where prompt and response tokens are interleaved,
the counting correctly handles mixed positions — any position marked as
prompt in ``prompt_mask`` is counted as masked, regardless of its position
in the sequence.

Zero-token batches
------------------

If a batch contains sequences with zero valid tokens, the metrics will
report zeros for all fields without crashing. ``mean_effective_length``
will be 0.0.

Limitations
-----------

* The metrics are computed per-update (per optimizer step), not per-sample.
* Cross-rank aggregation uses the existing ``merge_train_stats`` mechanism,
  which averages metrics across data-parallel ranks.
* Position 0 of each sequence is not an action position (no previous token
  to predict from), so ``masked_tokens + effective_loss_tokens`` equals
  ``total_input_tokens - num_sequences``.