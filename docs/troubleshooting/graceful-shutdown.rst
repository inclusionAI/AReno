:orphan:

Graceful Shutdown
=================

AReno implements a two-stage graceful shutdown for training and serving:

1. **First signal** (Ctrl-C / SIGINT or SIGTERM): Sets a shutdown-requested
   flag, logs the reason and current stage (training, rollout, or serving),
   starts a deadline timer (default 30s). The training loop checks the flag
   at each step and breaks at the next safe point. If the deadline expires
   before the loop exits, forces immediate exit.
2. **Second signal**: Forces immediate exit via ``os._exit()``, preserving
   the initial termination reason in the exit message.

This means Ctrl-C will not hang indefinitely — the first press initiates a
clean shutdown, and the second press (or deadline timeout) forces
immediate termination.

Training integration
--------------------

In ``areno/cli/train.py``, ``run()`` wraps ``trainer.fit(shutdown=...)`` in
a ``GracefulShutdown`` context manager. The trainer receives the shutdown
coordinator and checks ``shutdown.should_stop`` at the top of each training
step. If a signal was received, the loop breaks at the next safe point
(after the current step completes).

The trainer also calls ``shutdown.set_stage()`` to track whether the
current step is in rollout or training phase, so the shutdown message
accurately reports which stage was interrupted.

Serving integration
-------------------

In ``areno/cli/serve.py``, ``serve_command()`` wraps ``uvicorn.run()`` in
a ``GracefulShutdown`` context manager with stage set to ``SERVING``.
The first Ctrl-C lets uvicorn finish in-flight requests; the second
forces immediate exit.

Behavior
--------

When a signal is received, AReno prints to stderr::

   SIGINT received during training. Stopping gracefully... (press Ctrl-C again to force exit, or wait 30s for auto-force)

If the deadline expires::

   Deadline (30s) expired after graceful shutdown request. Forcing exit.

If a second signal arrives::

   Forced exit: Forced exit (SIGINT) — second signal received during training.

Limitations
~~~~~~~~~~~

* Signal handlers can only be installed from the main thread.
* The graceful shutdown depends on the training loop checking
  ``shutdown.should_stop`` — long-running operations within a step
  that do not check will only be interrupted by the deadline or second
  signal.
* In distributed runs, each rank receives signals independently.  The
  coordinator process handles the signal and closes the worker cluster
  via ``TPCluster.close()``. Cross-rank reason/deadline synchronization
  is not implemented; workers are terminated via the SHUTDOWN command.
* The default deadline is 30 seconds. Pass ``deadline_s`` to
  ``GracefulShutdown()`` to customize.
