:orphan:

Graceful Shutdown
=================

AReno provides an opt-in two-stage shutdown for training and serving. The
default remains the platform signal behavior. Enable the coordinator with
``--graceful-shutdown`` and optionally set a positive deadline::

   areno train --graceful-shutdown --shutdown-deadline-s 30 \
      --ckpt Qwen/Qwen3-0.6B --dataset-path gsm8k:main --algo gspo

   areno serve --graceful-shutdown --shutdown-deadline-s 30 \
      --model-path Qwen/Qwen3-0.6B

``--shutdown-deadline-s`` must be greater than zero and defaults to 30
seconds. Click rejects invalid values before model or worker initialization::

   areno train --graceful-shutdown --shutdown-deadline-s 0 ...

Behavior
--------

On the first SIGINT or SIGTERM, AReno records the initial reason, stage, and
monotonic deadline. It stops accepting new serving requests or scheduling new
rollout, scoring, and training work, lets the current operation reach its next
safe point, flushes metrics, and closes workers through the existing runtime
protocol. The same structured event is included in the worker ``SHUTDOWN``
command so every local rank receives the coordinator's reason and deadline.

A second signal, or expiry of the deadline, forces immediate exit with the
exit code derived from the initial signal. The initial reason remains present
in the forced-exit event.

Training safe points
--------------------

SFT stops before the next train batch. DPO stops before the next reference
score and between reference scoring and policy training. GSPO and GRPO stop
before the next rollout, after a completed rollout, and before policy
training. PPO additionally checks between reward, reference, actor, and critic
scoring and before critic or actor training.

The current backend operation is not interrupted in place. A long-running
kernel or distributed call must finish before a graceful safe point can run;
the deadline or a second signal remains the escape hatch.

Serving safe point
------------------

When enabled, AReno owns SIGINT and SIGTERM instead of Uvicorn. The first
signal marks the application as closing, causes new completion requests to
return HTTP 503, asks Uvicorn to drain in-flight requests, and then runs the
existing FastAPI shutdown hook to close the engine.

Observable output
-----------------

The first signal prints a human-readable message to stderr. Logs also contain
machine-readable records prefixed with ``shutdown_event=``. The JSON fields
include ``event``, ``state``, ``signal_number``, ``stage``, ``reason``,
``timestamp``, ``deadline``, and ``first_signal``. Training with a metrics
directory also stores the initial event under ``shutdown`` in the dashboard
state artifact before the metrics writer is closed.

Dashboard display
-----------------

When a train or serve job is visible in ``areno dashboard``, its detail page
shows a shutdown card after the first signal. The card reports the graceful or
forced state, signal, interrupted stage, initial reason, and remaining
deadline. A dashboard-started job stays in ``stopping`` while it drains, and
the **Stop** action changes to **Force stop** so a second signal remains
available. Completed and forced exits retain the initial shutdown reason in
the card and in the regular log view.

Limitations
-----------

* Signal handlers can only be installed from the main thread.
* Coordination currently covers the local worker ranks owned by one AReno
  coordinator. Coordination between separate launcher processes remains the
  launcher's responsibility.
