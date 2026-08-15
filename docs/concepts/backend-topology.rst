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
and CUDA-context memory must fit on those GPUs. For an overlapping topology,
AReno selects train and rollout relay ranks on different physical GPUs, then
fans each received bucket through rollout-only TP/DP groups. This keeps
duplicate physical GPUs out of an NCCL communicator.

Every completed synchronization logs its total time, collective transfer time,
bytes, tensor count, and effective throughput. The same values are emitted
with the next training metrics as ``policy_sync_time_s``,
``policy_sync_transfer_time_s``, ``policy_sync_bytes``,
``policy_sync_tensors``, and ``policy_sync_throughput_gbps``.

Runtime scope
-------------

The current backend runtime is entirely AReno-owned CUDA/NVIDIA code.
``ArenoEngine`` (``areno/engine/api.py``) manages a local worker cluster on a
single node. Both rollout and training use custom CUDA kernels shipped in
``areno/accel/`` for attention, activation, normalization, embedding, linear,
MoE, and routing operations. All inter-GPU communication goes through NCCL
collectives managed by AReno's cluster protocol (``areno/engine/protocol/``).

The colocated path runs one ``ArenoEngine`` that handles both
``generate_rollout`` and ``step``. The partitioned path starts two independent
``ArenoEngine`` instances — one for training, one for rollout — within the
same distributed world.

Out of scope
------------

This backend topology document describes the current implementation only. The
following are explicitly **not** promised or implied by this document:

* MLX, TPU, or non-NVIDIA accelerator backends.
* vLLM, SGLang, Megatron-LM, or Ray integration.
* Multi-node distributed training (the cluster protocol is single-node).
* A pluggable backend interface that third-party engines can implement without
  modifying AReno source.

Future architecture changes will be documented in the PR that introduces them.
