:orphan:

Health check issues
===================

The startup health check (Issue #249) classifies the first training updates as
``pass`` / ``warn`` / ``fail``. It is **off by default**; enable it with
``--health-check-enabled``. See :doc:`/cli/observability` for the output format.

The check never alters your configuration, deletes artifacts, or kills
unrelated processes. A ``FAIL`` only aborts the run when ``--health-check-on-fail
fail`` is set; the default ``warn`` logs and continues.

Locating a failure
------------------

Every check result carries a ``stage`` and an ``input``. The ``input`` field
names the triggering config field (or ``total_batches`` for a data anomaly) so
you can locate the cause without inspecting training samples — failure
messages never embed prompt or token text.

* ``stage=trainer`` — backend-reported signal (``loss`` / ``grad_zero_ratio``)
  or the effective-token count.
* ``stage=rollout`` — sample-side signal (rewards / skipped-long).

Read the structured artifact at
``<metrics-log-dir>/health_check/<run_id>.json`` for the full per-check
breakdown and the ``original_errors`` list (e.g. NaN detections). The
``metric_ref`` on each check points into the existing ``rollout/*`` / ``train/*``
TensorBoard namespace so you can cross-reference the underlying series.

Common signals
--------------

* ``effective_tokens FAIL`` — every batch in the window had zero response
  tokens. Inspect the prompt format and ``--max-prompt-tokens`` (overlong
  prompts are skipped before training).
* ``reward_variance FAIL`` — reward is constant while variation is required.
  Verify the reward function parses the completion and that the task actually
  admits variation. For legitimately constant-reward tasks, pass
  ``--health-check-allow-constant-reward``.
* ``loss_change FAIL`` — loss is unchanged or barely changed across the
  window (delta at or below ``min_abs_delta_fail``, which defaults to 0).
  Check the learning rate (``--lr``), gradient norm in ``train_stats`` (a
  near-zero ``grad_norm`` pairs with a high ``grad_zero_ratio``), and whether
  the loss mask leaves any response tokens trainable.
* ``skipped_batches FAIL`` — either the rollout skip ratio or the
  ``grad_zero_ratio`` proxy exceeded the fail threshold. A zero-batch window
  (``total_batches=0``) is itself a data/input anomaly.

False-positive resistance
-------------------------

Defaults are warn-prone, not fail-prone. Constant reward passes when
``--health-check-allow-constant-reward`` is set. A single-step window
(``--health-check-window 1``) only warns on ``loss_change`` because a one-sample
delta is unreliable. If you see unexpected ``WARN`` on a healthy run, widen the
window or relax the relevant threshold rather than disabling the check.

NaN / Inf
---------

A non-finite loss or reward is a dedicated ``FAIL`` and is recorded in
``original_errors`` — it is not masked by a threshold comparison. Fix the
upstream numeric issue (e.g. gradient explosion → lower ``--lr`` / tighten
``--grad-clip-norm``) rather than suppressing the check.