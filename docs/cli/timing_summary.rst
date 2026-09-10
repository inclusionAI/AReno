:orphan:

Timing summary CLI reference
============================

``areno timing-summary`` aggregates the wall time spent in each RL training
phase (generation / reward / training / sync / waiting) for a run's existing
metrics artifacts. It reads ``--metrics-log-dir`` output and reports two views:
the **latest update** and the **whole run**. It is a read-only snapshot — it
writes nothing, initializes no models or workers, and can be run against a
directory even while the run is still in progress.

.. code-block:: bash

   areno timing-summary /tmp/areno/tfevent

For machine-readable output:

.. code-block:: bash

   areno timing-summary /tmp/areno/tfevent --json

The ``RUN_DIR`` argument is the metrics directory written during training (the
``--metrics-log-dir`` value). It must contain ``events.out.tfevents.*`` or
``dashboard_state.<pid>.json``; validation runs before TensorBoard loading and
names the missing input on failure.

Output fields (``--json``)
--------------------------

* ``run_status`` — ``"active"`` or ``"completed"``, resolved from process
  liveness.
* ``num_steps`` — number of steps with timing data.
* ``latest_update`` — per-phase breakdown for the highest step, with a
  ``partial`` flag and reconciliation columns.
* ``whole_run`` — per-phase sums across all steps, with reconciliation columns.
* ``overlap`` — declared containment relations (currently always empty).
* ``missing`` — canonical phases never recorded in any step.
* ``divergences`` — steps where the rollup and ``time/*`` echo disagreed.

Reconciliation columns (``reported_total`` / ``reconstructed_total`` / ``diff``
/ ``total_source``) let the aggregated totals be checked against the raw
step-end-to-end events.

Phases follow the dashboard's canonical segment vocabulary (``rollout``,
``reward``, ``train``, ``advantages``, ``save``, …). Phases that a trainer does
not time appear as ``missing``; out-of-vocabulary sub-phase timers (e.g. PPO's
critic forwards) fold into ``other`` and stay in the reconstructed sum.

Limitations
-----------

* Reading requires the ``tensorboard`` package (the same package the dashboard
  uses to read events). If it is not installed, the command exits with a clear
  message rather than emitting an empty report.
* ``run_status`` uses process liveness, not the ``dashboard_state`` ``status``
  field (the trainer never writes a terminal status). It may misreport when the
  metrics dir is read on a different host than where the run executed, or after
  pid reuse.
* Only phases already emitted by the trainer are surfaced; this command
  aggregates existing timing events and does not add new instrumentation.
