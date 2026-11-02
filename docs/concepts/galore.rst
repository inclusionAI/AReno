GaLore AdamW
============

AReno provides opt-in GaLore (Gradient Low-Rank Projection) for full-parameter
CUDA training on one GPU. Every weight stays trainable; GaLore only changes how
AdamW stores its moments for large linear weights. No adapter parameters are
added, and checkpoints keep the normal model format.

For quantized moments instead of low-rank moments, see :doc:`adamw-8bit` and
:doc:`adamw-4bit`. Measurements are in :doc:`galore-results`.

Update rule
-----------

For an eligible weight matrix ``W`` (``m x n``) with gradient ``G``, GaLore keeps
an orthonormal basis ``P`` of rank ``r`` on the smaller side of the matrix and
Adam moments of the projected gradient. With ``m < n`` (the other case is the
transpose), optimizer step ``t`` is::

   if (t - 1) % update_proj_gap == 0:
       P = top-r left singular vectors of G       # projection refresh
   R = P^T G                                      # r x n
   m = beta1 * m + (1 - beta1) * R
   v = beta2 * v + (1 - beta2) * R * R
   N = (m / (1 - beta1^t)) / (sqrt(v / (1 - beta2^t)) + eps)
   W = W * (1 - lr * weight_decay) - lr * scale * P N

The update is AReno's AdamW (decoupled weight decay, bias correction, the same
``eps``) applied in the projected space and mapped back to the full weight. As
in the reference implementation, the moments are kept when the basis is
refreshed.

* The basis comes from a symmetric eigendecomposition of the smaller Gram
  matrix (``G G^T`` or ``G^T G``), whose eigenvectors are the singular vectors
  of ``G``. On the benchmark GPU this was six to nine times faster per block
  than a full SVD, and it never forms the large singular-vector factor.
* Each basis vector is oriented so its largest-magnitude entry is positive.
  The basis therefore does not depend on solver sign conventions, which keeps
  refreshed bases consistently oriented with the moments they inherit.
* Refreshes and projections run in IEEE FP32, even though workers enable TF32
  matrix multiplication for the model.

Eligible tensors
----------------

GaLore applies to the weights of AReno's tensor-parallel linear layers, which
cover the attention and MLP projections of dense models. A fused weight such as
``qkv_proj`` (``[q; k; v]``) or ``gate_up_proj`` (``[gate; up]``) is split into its
logical row blocks, and each block gets its own basis and moments, matching
GaLore applied to unfused ``q_proj``/``k_proj``/``v_proj``/``gate_proj``/``up_proj``
modules. A block uses GaLore only when the basis plus low-rank moments is
smaller than the two full-rank moments, ``r * (min(m, n) + 2 * max(m, n)) <
2 * m * n``.

Every other trainable tensor keeps the regular AdamW update with FP32 moments:
token embeddings, the LM head, norms, biases, expert banks and blocks too small
for the chosen rank. GaLore matrices keep the same lossless FP32 master weights
as the regular optimizer, so BF16 rounding is identical and only the moments
are approximate.

Memory
------

For Qwen3-0.6B at rank 128, the 196 projection blocks hold 440M of the 752M
trainable parameters. Their FP32 Adam moments shrink from 3.28 GiB to
0.41 GiB, plus 0.10 GiB of bases. Other memory is unchanged:

* BF16 weights and the FP32 master low bits for every parameter;
* the full-model FP32 gradient accumulated before the optimizer step;
* the AdamW state of embeddings and the LM head (311M parameters in Qwen3-0.6B,
  because the AReno Qwen3 head is a separate parameter);
* activations and any rollout KV cache.

A projection refresh adds a transient workspace (the Gram matrix and its
eigenvectors, one block at a time) and takes noticeably longer than a regular
optimizer step. Training reports both: ``optimizer_peak_allocated_bytes`` and
``optimizer_step_time_s`` cover the update, and ``galore_refreshed_matrices``
marks refresh steps. The optimizer also reports its persistent state as
``galore_projection_bytes``, ``galore_moment_bytes``, ``galore_master_bytes``,
``galore_adamw_state_bytes`` and ``galore_full_rank_moment_bytes``.

Command line
------------

Add ``--galore-rank`` to a single-GPU CUDA ``areno train`` command:

.. code-block:: bash

   areno train \
     --algo sft \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path yahma/alpaca-cleaned \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --world-size 1 \
     --tp-size 1 \
     --lr 1e-5 \
     --galore-rank 128 \
     --galore-update-proj-gap 100 \
     --galore-scale 4

* ``--galore-rank``: rank of the projected moments; enables GaLore.
* ``--galore-update-proj-gap``: optimizer steps between projection refreshes
  (default 200).
* ``--galore-scale``: multiplier on projected updates (default 1.0).

GaLore requires ``--world-size 1 --tp-size 1``. It cannot be combined with
LoRA, ``--adam-8bit`` or ``--adam-4bit``, and supports
``--optimizer-state-offload`` ``none`` or ``cpu``.

Choosing the scale
------------------

A rank-``r`` projection spreads each normalized update over a subspace, so the
per-element update is roughly ``sqrt(r / min(m, n))`` times that of full-rank
AdamW at the same learning rate. Keep the AdamW learning rate and raise
``--galore-scale`` instead. With rank 128 on Qwen3-0.6B (``min(m, n) = 1024``),
scales 2 to 4 matched AdamW validation loss within 1% in the reported SFT runs,
while the default 1.0 trailed by 1.5%.

Trainer configuration
---------------------

.. code-block:: python

   from areno.api.trainer_config import TrainerConfig

   config = TrainerConfig(
       algo="sft",
       ckpt="Qwen/Qwen3-0.6B",
       dataset_path="yahma/alpaca-cleaned",
       dataset_loader_fn="examples/sft/alpaca/dataset_loader.py",
       backend="cuda",
       world_size=1,
       tp_size=1,
       optimizer_lr=1e-5,
       galore_rank=128,
       galore_update_proj_gap=100,
       galore_scale=4.0,
   )

Checkpoint and resume
---------------------

``Trainer.save_checkpoint`` writes model weights only. To resume training
exactly, also save the optimizer state, which for GaLore holds the projection
bases, low-rank moments, per-matrix step counters, the FP32 master weights and
the GaLore settings:

.. code-block:: python

   trainer.save_checkpoint(path)
   trainer.save_optimizer_state(path)    # writes areno_optimizer_rank00000.pt

   # Later, in a new process:
   trainer = Trainer(1, path, custom_config=config.backend_config(), ...)
   trainer.init()
   step = trainer.load_optimizer_state(path)

``load_optimizer_state`` restores the optimizer step that drives the learning
rate schedule, overwrites the live weights with the saved FP32 masters, and
continues trainer step numbering. It rejects state saved by another optimizer
class, a different DP/TP layout, or different GaLore rank, refresh interval or
scale. The same methods work for the other CUDA AdamW optimizers.
``examples/galore/train_seeded.py`` shows the same flow around ``areno train``.

Limitations
-----------

* One GPU only: a basis needs the whole matrix gradient, so data-parallel
  sharded state and tensor-parallel projection are not implemented.
* The full FP32 gradient is still materialized before the optimizer step; GaLore
  reduces optimizer state, not gradient memory.
* Projections built outside AReno's linear layers keep AdamW.
* Moments are not rotated into a refreshed basis, as in the original method.
