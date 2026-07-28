:orphan:

Graceful Shutdown
=================

AReno implements a two-stage graceful shutdown for training and serving:

1. **First signal** (Ctrl-C / SIGINT or SIGTERM): Sets a shutdown-requested
   flag, logs the reason and current stage (training, rollout, or serving),
   and allows the main loop to reach a safe point, flush outputs, and close
   workers cleanly.
2. **Second signal**: Forces immediate exit via ``os._exit()``, preserving
   the initial termination reason in the exit message.

This means Ctrl-C will not hang indefinitely — the first press initiates a
clean shutdown, and the second press forces immediate termination.

Behavior
--------

When a signal is received, AReno prints to stderr::

   SIGINT received during training. Stopping gracefully... (press Ctrl-C again to force exit)

The training loop checks ``shutdown.shutdown_requested`` and breaks at the
next safe point (end of current step).  Workers are then closed via the
existing ``TPCluster.close()`` path.

If a second signal arrives before the graceful shutdown completes::

   Forced exit: Forced exit (SIGINT) — second signal received during training.

The process exits with code 130 (SIGINT) or 143 (SIGTERM).

Programmatic API
----------------

The :mod:`areno.engine.shutdown` module exposes:

* :class:`GracefulShutdown` — coordinator object with ``install()``,
  ``uninstall()``, ``set_stage()``, ``shutdown_requested``, and
  ``begin_shutdown()`` methods.
* :class:`ShutdownStage` — enum: ``TRAINING``, ``ROLLOUT``, ``SERVING``,
  ``IDLE``.
* :class:`ShutdownState` — enum: ``RUNNING``, ``SHUTDOWN_REQUESTED``,
  ``SHUTTING_DOWN``, ``FORCED``.
* :class:`ShutdownInfo` — structured info with ``to_dict()`` for JSON output.
* :func:`format_shutdown_reason` — human-readable reason string.

Example::

    from areno.engine.shutdown import GracefulShutdown, ShutdownStage

    with GracefulShutdown() as shutdown:
        shutdown.set_stage(ShutdownStage.TRAINING)
        for batch in training_loop:
            if shutdown.shutdown_requested:
                break
            train_step(batch)

Limitations
~~~~~~~~~~~

* Signal handlers can only be installed from the main thread.
* The graceful shutdown depends on the main loop checking
  ``shutdown_requested`` — long-running operations that do not check will
  only be interrupted by the second (forced) signal.
* In distributed runs, each rank receives signals independently.  The
  coordinator process handles the signal and closes the worker cluster.