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

Colocated and partitioned engines
---------------------------------

By default, ``ArenoBackend.initialize`` creates one ``ArenoEngine``. The same
engine handles both sides of the loop:

* ``ArenoBackend.rollout_batch`` calls ``ArenoEngine.generate_rollout``.
* ``ArenoBackend.train`` calls ``ArenoEngine.step``.

``ArenoEngine`` is implemented in ``areno/engine/api.py``. It coordinates the
worker cluster used by both rollout and training.

Online RL runs may instead assign CUDA devices to an independent rollout
engine. Training and rollout workers then join one
distributed world but use separate TP and DP process groups. This permits, for
example, training with TP 8 while generating rollouts with TP 2.

After an optimizer step, the rollout engine keeps its current policy until the
next rollout begins. AReno then streams the new policy directly between GPUs
with NCCL. Tensors are distributed over training DP rows by byte size, moved
through a bounded bucket, and written into the rollout TP shards without a CPU
or filesystem staging copy.

Both device lists use logical indices within the parent process'
``CUDA_VISIBLE_DEVICES``. They may overlap; overlapping devices hold both a
training worker and a rollout worker, so the combined model, optimizer, cache,
and CUDA-context memory must fit on those GPUs.
