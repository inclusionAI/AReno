8-bit AdamW
===========

AReno provides opt-in 8-bit AdamW moment storage on CUDA and MLX. Model
weights keep their normal precision and checkpoint format; this option does
not produce an INT8 inference model. For the CUDA mode that also factors
second moments and streams BF16 gradient shards, see :doc:`adamw-4bit`.

State representation
--------------------

AdamW tracks a first moment (an exponential average of gradients) and a second
moment (an exponential average of squared gradients). The 8-bit optimizer
stores each moment as a byte-sized codebook index, with separate FP32 scales
for blocks of 128 values by default. Blockwise normalization limits the effect
of an outlier to its block. A signed dynamic codebook represents the first
moment, and an unsigned dynamic codebook represents the nonnegative second
moment. Updates reconstruct moment values, apply AdamW, and quantize the new
moments for storage.

On CUDA, moments are DP-sharded, updates use fused kernels, and there is no
persistent FP32 master-weight copy. The existing FP32 gradient accumulation
path remains in use. MLX implements blockwise moment storage with streamed
updates through its native backend.

Recognized token-embedding parameters retain FP32 moments by default to avoid
quantizing embedding-gradient outliers. This does not add a normalization
layer, reinitialize embeddings, or otherwise change the model's forward pass.
Explicit parameter precision policies can also retain FP32 moments.

Memory and numerical behavior
-----------------------------

For a full quantized block, the two moments cost approximately
``2 + 8 / 128 = 2.0625`` bytes per parameter instead of 8 bytes for FP32
moments: about 74% less moment storage. Partial blocks, padding, and FP32
embedding exemptions reduce the overall saving. CUDA also avoids the FP32
master-weight copy used by its default optimizer.

These savings do not apply to activations, KV caches, gradients, or all other
GPU allocations. Total training memory reduction depends on the model,
sequence length, parallel layout, and optimizer-state residency.

Quantized moments and writing updates back to model precision introduce
numerical differences from FP32-master AdamW. This is not a prescribed
learning-rate increase and does not guarantee faster convergence. Compare
reward or validation loss under the same training settings when switching
optimizers.

Command line
------------

Add ``--adam-8bit`` to a training command:

.. code-block:: bash

   areno train \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path gsm8k:main \
     --dataset-loader-fn examples/math/dataset_loader.py \
     --reward-fn-path examples/math/math_verify_reward.py \
     --algo gspo \
     --world-size 1 \
     --tp-size 1 \
     --batch-size 2 \
     --n-samples 2 \
     --mini-bs 1 \
     --adam-8bit

The existing ``--lr``, ``--adam-beta1``, ``--adam-beta2``, and weight-decay
settings still apply. ``--adam-8bit`` and ``--adam-4bit`` are mutually
exclusive. Omit both flags to use the backend's default optimizer.

For MLX setup and training commands, see :doc:`/getting-started/mlx`.

Trainer configuration
---------------------

Set ``adam_8bit=True`` on a trainer configuration:

.. code-block:: python

   from areno.api.trainer_config import PolicyTrainerConfig

   config = PolicyTrainerConfig(
       algo="gspo",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="gsm8k:main",
       dataset_loader_fn="examples/math/dataset_loader.py",
       reward_fn_path="examples/math/math_verify_reward.py",
       backend="cuda",
       world_size=1,
       tp_size=1,
       adam_8bit=True,
   )

For the lower-level CUDA SDK, set
``CudaConfig(optimizer={"adam_8bit": True})``. The equivalent engine setting
is ``OptimizerConfig(adam_8bit=True)``.

CUDA state offload and metrics
----------------------------------

The CUDA optimizer supports CPU and disk state offload:

.. code-block:: bash

   areno train ... --adam-8bit --optimizer-state-offload cpu

   areno train ... --adam-8bit \
     --optimizer-state-offload disk \
     --optimizer-state-offload-dir /local/nvme/areno-optimizer \
     --optimizer-state-offload-batch-size 1

Use fast local storage for disk offload. Runtime mmap files are temporary
scratch storage, not restartable checkpoints. Model-weight checkpoints remain
usable without the 8-bit flag; serialized optimizer states require the
matching optimizer representation.

CUDA training reports initialized, DP-local moment storage through:

* ``adam8_quantized_state_bytes``: byte-sized moment arrays.
* ``adam8_fp32_exempt_bytes``: moments retained in FP32.
* ``adam8_block_metadata_bytes``: FP32 block scales.
* ``adam8_total_bytes``: the sum of those three quantities.

These are logical state sizes, not total or peak GPU memory; offloaded state
can still contribute to them.
