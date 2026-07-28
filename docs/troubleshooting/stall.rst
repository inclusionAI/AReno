Stage stall warnings
====================

A *stall* means a logical training stage (``loading``, ``data``, ``rollout``,
``reward``, or ``training``) has had no progress event for longer than a
configured idle threshold. AReno surfaces this through the stage stall watcher,
described in :doc:`/cli/observability`.

The warning looks like:

.. code-block:: text

   stall stage=rollout wait_s=312.4 threshold_s=300.0

It is emitted by the ``areno.stall_watch`` logger and attached to the current
step's ``train_stats`` as ``stall_stage`` / ``stall_wait_s`` /
``stall_threshold_s``. The warning never stops the run.

Common causes by stage:

* ``loading``: tokenizer or checkpoint loading is slow, or the model path is
  on a cold network mount. Check disk I/O and the model path.
* ``data``: dataset iteration is blocked, e.g. streaming a remote dataset with
  a slow connection, or preprocessing is single-threaded for a large corpus.
* ``rollout``: generation is slow, or an agentic rollout is waiting on
  external tool/test/sandbox work. For agentic runs, distinguish model time
  from environment time (see :doc:`agentic-rollout`).
* ``reward``: the reward function is slow or blocked on an external call.
* ``training``: the optimizer step is slow, or a checkpoint save is in
  progress (``save_checkpoint_start`` / ``save_checkpoint_end`` also tick the
  ``training`` stage).

Tuning the watcher:

* ``--stall-warn-interval-s 0`` disables it (default).
* Increase ``--stall-warn-interval-s`` if the run legitimately spends long
  periods in one stage (e.g. a large checkpoint save).
* Increase ``--stall-warn-min-interval-s`` to reduce repeat warnings for a
  persistent stall.

The watcher is validated before model or worker initialisation, so an invalid
configuration (negative interval, ``min > interval``, unknown stage name)
fails fast with an actionable error rather than mid-run.

