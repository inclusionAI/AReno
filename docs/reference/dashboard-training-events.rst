Training Event Overlay
======================

The AReno dashboard overlays **structured training events** on TensorBoard
metric charts, making it easier to spot anomalies during and after training
runs.

Overview
--------

Four event kinds are derived **read-only** from existing metrics and
captured logs — no new artifact files are produced:

================== ============== ===========================================
Kind               Severity       Signal source
================== ============== ===========================================
``non_finite``     ``warn``       NaN or ±Inf in TensorBoard scalars
``constant_reward`` ``info``       ``rollout/rewards_std`` ≈ 0 with
                                  ``rewards_max == rewards_min``
``invalid_batch``  ``info``/``warn`` ``rollout/advantages_std`` ≤ 1e-6
                                  for consecutive steps (streak ≥ 3 →
                                  ``warn``, streak 1–2 → ``info``)
``oom``            ``error``       OOM pattern in captured subprocess logs
================== ============== ===========================================

Events appear as **clickable markers** at the bottom of each metric chart.
Filters (checkboxes per kind) are independent of the selected metric curve.

API
---

``GET /api/jobs/<job_id>/events``
    Returns a list of derived training events.

    Query parameters:
    - ``types``: comma-separated kind filter (e.g. ``?types=non_finite,oom``)
    - ``limit``: maximum number of events (default 500, max 5000)

    Response:

    .. code-block:: json

       {
         "events": [
           {
             "kind": "non_finite",
             "step": 12,
             "time": "2026-07-31T01:00:00Z",
             "severity": "warn",
             "detail": "rollout/rewards_mean=NaN at step 12",
             "fields": {"tag": "rollout/rewards_mean", "value": "NaN"},
             "log_hint": {"kind": "metric_context", "ref": "rollout/rewards_mean"}
           }
         ]
       }

Event fields:

``kind``
    One of ``non_finite``, ``constant_reward``, ``invalid_batch``, ``oom``.
``step``
    Training step at which the event was detected.
``time``
    Job ``updated_at`` at detection time (not the original TensorBoard write
    time).
``severity``
    ``info``, ``warn``, or ``error``.
``detail``
    Human-readable description.
``fields``
    Structured supplementary data (varies by kind).
``log_hint``
    Click-through anchor. ``kind`` is ``keyword`` (OOM: log line index
    for ±20-line window), ``metric_context`` (non-OOM: metric tag for
    step ± 3 context), or ``none``.

Agent tool
~~~~~~~~~~

``fetch_events(job_id, types?, limit?)``
    Returns the same event list as the HTTP endpoint, available to the
    dashboard's built-in agent.

Click-through context
---------------------

- **OOM events** (``log_hint.kind = "keyword"``): popover shows ±20 log
  lines around the matching OOM line.
- **Other events** (``log_hint.kind = "metric_context"``): popover shows
  the relevant metric values for steps ± 3 around the event. No log
  context is shown — captured logs do not carry step/timestamp metadata.
- ``log_hint.kind = "none"``: only event metadata is displayed.

Limitations
-----------

- **500-step truncation**: ``_load_tensorboard_scalars`` loads only the
  last 500 events per TensorBoard tag. Event detection therefore covers
  the most recent ~500 steps. After a dashboard restart, ``job.metrics``
  is rebuilt from scratch and earlier events are not recoverable.
- **OOM semantics**: OOM causes the training subprocess to crash (job
  status → ``failed``). The dashboard does **not** track OOM recovery —
  ``recovered`` is never set. Auto-tune OOM retries create separate jobs
  and are not linked.
- **NaN behaviour change**: NaN scalars were previously silently
  discarded; they now appear as ``non_finite`` events. NaN values still
  do not enter the metric series (``metric_series`` output is unchanged).
- **Events are not persisted**: Events are derived in memory on each
  ``_load_metric_files`` call and cached on the job object. They are not
  written to ``to_json``/``from_json`` or any artifact file.

Default behaviour
-----------------

When no events are detected (e.g. legacy runs without TensorBoard data),
the chart and filters are absent — identical to pre-feature behaviour.

Minimal example
---------------

The following examples run without external databases, GPU, or network
services. They demonstrate the successful path and one boundary input.

**1. Query events via curl** (requires a running dashboard):

.. code-block:: bash

   # All events for a job
   curl -s http://localhost:8007/api/jobs/<job_id>/events | python3 -m json.tool

   # Filter to OOM and non_finite only
   curl -s "http://localhost:8007/api/jobs/<job_id>/events?types=oom,non_finite" | python3 -m json.tool

**2. Python fixture** (no dashboard process required):

.. code-block:: python

   import json, tempfile, threading
   from pathlib import Path
   from areno.dashboard.server import DashboardState, Job

   mdir = Path(tempfile.mkdtemp()) / "metrics"
   mdir.mkdir()

   # Write a .jsonl metric file: step 1 has constant reward, steps 3-5
   # have a 3-step invalid-batch streak (advantages_std=0).
   rows = []
   for step, rwd_std, rwd_max, rwd_min, adv_std in [
       (1, 0.0, 1.0, 1.0, 0.5),   # constant reward
       (2, 0.5, 2.0, 0.0, 0.5),   # normal
       (3, 0.5, 2.0, 0.0, 0.0),   # invalid batch (streak starts)
       (4, 0.5, 2.0, 0.0, 0.0),
       (5, 0.5, 2.0, 0.0, 0.0),   # streak = 3 -> warn
       (6, 0.5, 2.0, 0.0, 1.0),   # recovery
   ]:
       rows.append({"name": "rollout/rewards_std", "value": rwd_std, "step": step})
       rows.append({"name": "rollout/rewards_max", "value": rwd_max, "step": step})
       rows.append({"name": "rollout/rewards_min", "value": rwd_min, "step": step})
       rows.append({"name": "rollout/advantages_std", "value": adv_std, "step": step})
   (mdir / "custom_metrics.jsonl").write_text(
       "\n".join(json.dumps(r) for r in rows) + "\n"
   )

   # Use an absolute path for metrics_dir so it works regardless of ROOT.
   state = DashboardState.__new__(DashboardState)
   state.jobs = {}
   state.lock = threading.RLock()
   job = Job(kind="train", name="example", command=[], config={}, metrics_dir=str(mdir))
   state.jobs[job.id] = job
   state._load_metric_files(job)

   # Successful path: constant_reward + invalid_batch detected
   kinds = {e["kind"] for e in job._events}
   assert "constant_reward" in kinds
   assert "invalid_batch" in kinds
   ib = [e for e in job._events if e["kind"] == "invalid_batch"][0]
   assert ib["severity"] == "warn" and ib["fields"]["streak"] == 3

   # Boundary input: a single advantages_std=0 (streak=1) is info, not warn
   job2 = Job(kind="train", name="boundary", command=[], config={}, metrics_dir=None)
   job2.metrics.append({"name": "rollout/advantages_std", "value": 0.0, "step": 1, "time": "t"})
   job2.metrics.append({"name": "rollout/advantages_std", "value": 1.0, "step": 2, "time": "t"})
   state._detect_training_events(job2)
   ib2 = [e for e in job2._events if e["kind"] == "invalid_batch"][0]
   assert ib2["severity"] == "info" and ib2["fields"]["streak"] == 1
   print("All assertions passed.")
