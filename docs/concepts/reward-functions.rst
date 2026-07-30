Reward functions
================

Reward functions turn generated completions or trajectories into numeric scores.
They are task-specific Python files loaded by AReno's training path and should
be deterministic while you are debugging a run.

The public shape is:

.. code-block:: python

   def reward_fn(example, completions) -> list[float]:
       ...

The function receives the original example plus generated completions, then
returns one score per completion. For agentic workflows, keep enough task state
in the dataset row for the reward function to explain why a trajectory passed or
failed.

Practical rules
---------------

* Keep parsing and scoring explicit.
* Return one score for each completion.
* Avoid network calls in the hot path unless the task requires them.
* Log enough context to debug wrong scores.

Where to go next
----------------

* :doc:`/cli/training` documents the training CLI flag.
* :doc:`/troubleshooting/reward-function` covers debugging workflow.
* :doc:`/reference/reward-function-api` documents the API contract.

Reward transformation
---------------------

After ``reward_fn`` scores each sample, AReno optionally transforms the raw
reward values before advantage computation. This is controlled by
``--reward-transform-mode`` on the training CLI:

* ``disabled`` (default) — no transformation; rewards pass through unchanged.
  This preserves the existing behavior exactly.
* ``clip`` — clamp each reward to ``[reward-clip-min, reward-clip-max]``.
  Both bounds are required. NaN inputs raise an error.
* ``standardize`` — z-score rewards across the full batch (cross-group, not
  per-group) using ``mean(r)`` and ``std(r) + eps``. This runs *before* the
  per-group advantage normalization in GRPO/GSPO, so it does not cancel out.

When enabled, both raw and transformed reward distributions are logged to
TensorBoard under ``rollout/reward_raw_*`` and ``rollout/reward_transformed_*``.

.. note::

   In distributed training (``world-size > 1``), ``standardize`` computes
   statistics from the local rank's batch only — there is no cross-rank
   all-reduce. Each rank standardizes against its own partial batch, so the
   global statistics are approximate. This is acceptable for reward shaping
   but means results may vary with different parallelism configurations.
