Backend Topology
================

The SDK keeps one public training and serving contract with two native backend
implementations:

.. code-block:: text

   Trainer
     -> Backend
       +-> CudaBackend -> ArenoEngine -> process/GPU workers
       +-> MlxBackend  -> MLX model   -> in-process Metal runtime

``Trainer`` is the public coordinator. In ``areno/api/trainer.py``,
``Trainer.init`` resolves a registered backend implementation, while
``Trainer.rollout_token_batch`` and ``Trainer.train`` delegate rollout and
training to that backend.

``Backend`` is the execution contract in ``areno/api/backend/base.py``. Its
rollout, scoring, role, training, and checkpoint methods define the operations
required by the shared trainer. ``CudaBackend`` lives in
``areno/api/backend/cuda/`` and ``MlxBackend`` lives in
``areno/api/backend/mlx/``. Backend modules are imported lazily, so selecting
MLX does not import Torch/CUDA and selecting CUDA does not import MLX.

The CLI selects the backend from the host: Linux uses CUDA and native Apple
Silicon uses MLX. There is no runtime fallback. SDK callers may pass
``backend_type=CUDA`` with ``CudaConfig`` or ``backend_type=MLX`` with
``MlxConfig`` explicitly. These backend symbols are exported by ``areno.api``.

CUDA colocated and partitioned engines
--------------------------------------

By default, ``CudaBackend.initialize`` creates one ``ArenoEngine``. The same
engine handles both sides of the loop:

* ``CudaBackend.rollout_batch`` calls ``ArenoEngine.generate_rollout``.
* ``CudaBackend.train`` calls ``ArenoEngine.step``.

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

MLX integrated runtime
----------------------

``MlxBackend`` owns one in-process model used by rollout, scoring, and
training. A long-lived scheduler performs prefill and token generation with
continuous batching. Training updates the same policy object after the rollout
session closes, so there is no second rollout model and no cross-device weight
copy. Reference, reward, and critic roles are backend-owned model roles for
DPO and PPO.

MLX runs with ``world-size=1`` and ``tp-size=1`` on unified memory. CUDA
device lists, independent rollout partitions, NCCL policy synchronization,
and CUDA graph capture do not apply. ``--drop-rollout-state`` controls whether
completed rollout cache state is retained across session boundaries.

Shared behavior
---------------

Both backends implement the same ``TrainSequence`` batches, algorithm loss
names, microbatch semantics, role APIs, rollout-session lifecycle, metrics,
and checkpoint entry point. The numerical kernels and native checkpoint
formats differ. CUDA saves its Hugging Face-oriented checkpoint layout; MLX
saves the model in native MLX format with tokenizer/processor assets and AReno
metadata.
