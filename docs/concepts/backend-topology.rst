Backend Topology
================

The current SDK runtime follows one AReno backend path:

.. code-block:: text

   Trainer
     -> Backend
       -> ArenoBackend
         -> ArenoEngine

``Trainer`` is the public coordinator. In ``areno/api/trainer.py``,
``Trainer.init`` resolves a registered backend implementation, while
``Trainer.rollout_token_batch`` and ``Trainer.train`` delegate rollout and
training to that backend.

``Backend`` is the execution contract in ``areno/api/backend/base.py``. Its
``rollout_batch`` and ``train`` methods define the operations required by the
training loop. ``ArenoBackend`` is the registered AReno implementation in
``areno/api/backend/areno/backend.py``.

One colocated engine
--------------------

``ArenoBackend.initialize`` creates one ``ArenoEngine`` and stores it in
``self._engine``. The same engine handles both sides of the loop:

* ``ArenoBackend.rollout_batch`` calls ``ArenoEngine.generate_rollout``.
* ``ArenoBackend.train`` calls ``ArenoEngine.step``.

``ArenoEngine`` is implemented in ``areno/engine/api.py``. It coordinates the
worker cluster used by both rollout and training, so the current backend does
not split those calls across separate engines or external runtimes.

Resource ownership and cleanup
------------------------------

Every runtime resource has a single owner and a deterministic close order.
The ownership chain mirrors the construction chain:

.. code-block:: text

   Trainer.close()  ->  ArenoBackend.close()  ->  ArenoEngine.close()  ->  TPCluster.close()

**Idempotency**: ``close()`` at every layer is idempotent. Calling it multiple
times—whether from a ``finally`` block, a context manager ``__exit__``, or
manual cleanup—is safe and does not raise.

**Fault safety**: If initialization fails at any stage, partially-created
resources are cleaned up before the error propagates:

* If ``ArenoBackend.initialize`` fails after engine creation, the engine is
  closed before re-raising.
* If ``Trainer.init`` fails after backend creation, the backend is closed
  before re-raising.
* If ``TPCluster.start`` fails after spawning workers, workers are terminated
  and queues are closed before re-raising.

**Resources tracked**:

* Worker processes (``multiprocessing.Process``) — terminated and joined.
* Command and result queues (``multiprocessing.Queue``) — closed and joined.
* Process groups (``torch.distributed``) — destroyed via
  ``destroy_process_group()``.
* Shared memory tensors — coordinator-side references released after workers
  exit.
* Temporary TCP sockets — held until worker rendezvous completes, then closed.
* Proxy server threads and sockets (``RolloutSession``) — shut down and joined.

**Recommended usage**:

.. code-block:: python

   # Option 1: context manager (preferred)
   with Trainer(world_size=8, model_path="/path/to/model") as trainer:
       trainer.init()
       trainer.rollout_batch(prompts, n_samples=8, sampling_params=params)
       trainer.train(train_batch, loss_fn)

   # Option 2: explicit close in finally
   trainer = Trainer(world_size=8, model_path="/path/to/model")
   try:
       trainer.init()
       trainer.rollout_batch(prompts, n_samples=8, sampling_params=params)
       trainer.train(train_batch, loss_fn)
   finally:
       trainer.close()

All trainer ``fit()`` methods already use the ``try/finally: self.areno.close()``
pattern, so CLI users do not need to add explicit cleanup.

**Observable output**: After ``close()``, the following state changes are
observable programmatically:

* ``trainer._closed`` is ``True``.
* ``trainer._backend`` is ``None``.
* ``trainer._metrics`` is ``None``.
* ``trainer._initialized`` is ``False``.
* Worker processes have been terminated and joined (no orphan processes).
* IPC queues have been closed and their feeder threads reaped.
* Shared-memory tensor references have been released.

When initialization fails, the original error message propagates unchanged
—cleanup errors are swallowed so they never mask the root cause.  The error
identifies the affected stage (e.g. ``"CUDA out of memory during model
loading"``) without exposing training samples.

**Limitations**: Daemon worker processes are killed if the coordinator process
is terminated by ``SIGKILL``; graceful cleanup only occurs on normal exit,
``SIGTERM``, or unhandled exceptions in Python code.
