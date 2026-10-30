Activation Compression
======================

AReno provides opt-in compression of saved training activations for
single-GPU, full-parameter training of dense text models, following GACT
(Liu et al., ICML 2022). The forward pass is unchanged: only the tensors that
autograd keeps for the backward pass are stored in fewer bits and are
reconstructed, with their original shape and dtype, when backward needs them.
Weights, gradients, optimizer state, the rollout KV cache and adapters are not
touched, and nothing is offloaded to host memory.

Each saved tensor gets its own precision (8, 4 or 2 bits, or uncompressed),
chosen by a gradient-sensitivity calibration so that the relative L2 error of
the parameter gradients stays within an error budget (default 1%) on a
calibration microbatch and a held-out microbatch. Measurements are in
:doc:`activation-compression-results`.

What is compressed
------------------

During the actor's training forward, ``torch.autograd.graph.saved_tensors_hooks``
see every tensor that autograd saves. Each eligible tensor is a *slot*:

* With activation checkpointing (the default), the slots are the checkpoint
  boundaries (the input hidden state of every decoder layer) and the tensors
  saved outside recomputed layers, such as the final norm input. Activations
  recomputed inside a layer live only while that layer runs backward and are
  not compressed.
* Without activation checkpointing, every eligible saved activation of the
  model forward is a slot: norm and projection inputs, MLP inputs and RoPE
  tables when the compiled graph saves them.

Never compressed: parameters and buffers and views of them; integer tensors
(token ids, ``cu_seqlens``, indices); 1-D tensors such as per-row RMSNorm
statistics; tensors with fewer than 65,536 elements; non-contiguous tensors;
and everything the training attention kernel saves (Q, K, V, its output and
log-sum-exp statistics), because its backward recombines them exactly. A
tensor whose storage one of these uncompressed saves keeps alive (for example
the output projection's input, a view of the saved attention output) is not
compressed either: a compressed copy would add memory instead of freeing it.
The LM head and the loss run after the model forward and keep their saved
tensors. A tensor saved by several operations is stored once.

Storage format
--------------

A compressed slot stores, for every group of 64 contiguous elements, an FP32
minimum and an FP32 step ``(max - min) / (2**bits - 1)`` plus the codes
``floor((x - min) / step + u)``, packed little-endian into bytes. ``u`` is a
uniform draw from a keyed counter-based hash of the element index, so the
rounding is stochastic and unbiased (the expected reconstruction equals the
input, as GACT's analysis requires) and still reproducible.

Transformer hidden states carry a few channels that are orders of magnitude
larger than the rest (Qwen3's residual stream has values above 8,000 in one
channel). One channel in 64, those with the largest magnitude in the tensor,
is therefore kept exactly in the original dtype and excluded from the group
ranges. Bytes per element of a BF16 tensor with a hidden size of 1,024:

.. list-table::
   :header-rows: 1

   * - Precision
     - Codes
     - Scales
     - Exact channels
     - Total
     - vs. BF16
   * - 8-bit
     - 1.0
     - 0.125
     - 0.031
     - 1.156
     - 57.8%
   * - 4-bit
     - 0.5
     - 0.125
     - 0.031
     - 0.656
     - 32.8%
   * - 2-bit
     - 0.25
     - 0.125
     - 0.031
     - 0.406
     - 20.3%

On CUDA, Triton kernels pack and unpack the codes; they produce the same codes,
scales and reconstructions bit for bit as the PyTorch path used on CPU. The
reconstruction workspace is one dequantized tensor at a time.

Choosing precisions
-------------------

The calibration runs before the first optimizer step. The step's first
microbatch calibrates and the second is the held-out validation batch;
gradients computed during calibration are discarded, so the step itself
trains exactly as without calibration.

1. FlashAttention's backward accumulates its query gradient with atomics, so
   two identical BF16 backward passes differ (typically 0.4-0.5% relative L2
   over all parameter gradients). Training reports that difference as
   ``actc_baseline_nondeterminism`` and calibrates with FlashAttention's
   deterministic backward, so the noise is not charged to compression.
2. Sensitivity: for every slot and every candidate precision, from 8 bits
   down, the gradients are recomputed with only that slot compressed. The
   squared relative L2 change of all parameter gradients, minus the squared
   noise floor, is the slot's sensitivity at that precision. Lower precisions
   of a slot are not measured once one exceeds the budget.
3. Plan: a greedy multiple-choice knapsack lowers, one level at a time, the slot
   whose next level adds the least sensitivity per byte saved, while the summed
   sensitivity stays below a target.
4. Validation: the plan is applied to the calibration and the held-out
   microbatch. The summed single-slot sensitivities overestimate the joint
   error (BF16 backward rounding amplifies any perturbation of a layer input
   to a similar gradient change, and these changes do not add up), so the
   target is rescaled by the measured error and the plan rebuilt, aiming at
   80% to 90% of the budget. 4-bit slots, by contrast, do add up, so a rescaled
   plan can overshoot; once one target's plan met the budget and another's did
   not, the next target bisects that bracket geometrically. After at most six
   rounds the plan with the fewest stored bytes whose errors on both
   microbatches are within the budget is kept; if none is, every slot stays
   uncompressed.

The plan is keyed by each slot's dtype, rank and last dimension and its
occurrence among slots with the same signature, so tensors that a compiled
region stops saving after a recompilation do not shift the other slots.
Unexpected slots stay uncompressed and are counted in ``actc_plan_misses``.

Calibration cost is roughly ``3 + slots * levels + 2 * rounds`` forward and
backward passes of one microbatch: about 70 passes for Qwen3-0.6B with
activation checkpointing (29 slots), and a few hundred without it.

A plan that meets the budget early in training does not necessarily meet it
later: as the loss converges, the gradient shrinks and loses coherence while
quantization noise does not, so the same plan's relative error grows (for
Qwen3-0.6B SFT it roughly doubled over 300 steps, as did the default
backward's own run-to-run difference). Every
``activation_compression_check_interval`` optimizer steps (default 50) the
plan is therefore re-measured on the step's first microbatch, two
deterministic passes, and recalibrated on that step's microbatches when its
error exceeds the budget, which can mean fewer compressed slots or none. Set
the interval to 0 to calibrate only once.

Command line
------------

.. code-block:: bash

   areno train \
     --algo sft \
     --ckpt Qwen/Qwen3-0.6B \
     --dataset-path yahma/alpaca-cleaned \
     --dataset-loader-fn examples/sft/alpaca/dataset_loader.py \
     --world-size 1 \
     --tp-size 1 \
     --batch-size 128 \
     --mini-bs 64 \
     --activation-compression adaptive \
     --activation-compression-error-budget 0.01

* ``--activation-compression {none,adaptive}``: enable compression (default
  ``none``).
* ``--activation-compression-error-budget``: relative L2 gradient error
  allowed on the calibration and held-out microbatches (default ``0.01``).

Compression requires the CUDA backend, one training GPU
(``--world-size 1 --tp-size 1``), a dense model and full-parameter training
(no LoRA). Use at least two microbatches per step (``--batch-size`` larger than
``--mini-bs``) so the plan is validated on a held-out microbatch.

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
       batch_size=128,
       mini_bs=64,
       activation_compression="adaptive",
       activation_compression_error_budget=0.01,
       activation_compression_bits=(8, 4, 2),  # candidate precisions
       activation_compression_check_interval=50,  # 0: calibrate once
   )

The same fields exist on ``RuntimeConfig`` for direct engine use; with
``CudaConfig`` pass them as ``runtime={"activation_compression": "adaptive", ...}``.

Metrics
-------

Every training step reports, per microbatch (averaged over the step):

* ``actc_saved_bytes``: the slots' size in their original dtype, which is what
  the uncompressed path keeps for them;
* ``actc_stored_bytes`` (and ``actc_stored_fraction``): what is kept instead,
  split into ``actc_packed_bytes`` (codes), ``actc_scale_bytes`` (group minima
  and steps), ``actc_outlier_bytes`` (exact channels and their indices) and
  ``actc_kept_bytes`` (uncompressed slots);
* ``actc_workspace_bytes``: the largest single reconstruction;
* ``actc_slots``, ``actc_compressed_slots`` and ``actc_plan_misses``.

The calibration step adds ``actc_calibration_error``,
``actc_held_out_error``, ``actc_noise_floor``, ``actc_baseline_nondeterminism``,
``actc_reference_drift``, ``actc_predicted_error``, ``actc_plan_rounds``,
``actc_plan_stored_fraction``, ``actc_plan_slots_{8,4,2,16}bit``,
``actc_calibration_passes``, ``actc_calibration_s``, ``actc_pinned_slots``
and ``actc_calibration_peak_allocated_bytes``; the full plan with per-slot
sensitivities is logged as ``activation compression calibration``. Check steps
report ``actc_check_error``, and ``actc_recalibrated`` when the check
triggered a new calibration.

Independently of compression, CUDA training also reports allocator peaks per
optimizer step: ``train_peak_allocated_bytes`` and
``train_peak_reserved_bytes`` for the forward and backward microbatches,
``optimizer_peak_allocated_bytes``, ``optimizer_peak_reserved_bytes`` and
``optimizer_step_time_s`` for the update, and ``allocated_bytes`` after it.

Checkpoint and resume
---------------------

``Trainer.save_checkpoint`` also writes the calibrated plan as
``activation_compression.json``. A compressed run that starts from such a
checkpoint (``--ckpt`` or ``Trainer(..., model_path)``) with the same budget
and candidate precisions reuses the plan instead of calibrating again, so a
resumed run compresses exactly like the run that saved it. Delete the file to
force a new calibration.

Limitations
-----------

* One GPU and dense text models only; tensor-parallel layouts, MoE routing and
  multimodal towers are not supported.
* A 1% budget sits close to BF16's own gradient noise. With small
  microbatches most slots therefore stay at 8 bits or uncompressed; relative
  errors shrink as microbatches grow, and so does the uncompressed fraction.
* The full training peak only falls when saved activations are part of it.
  When the peak is reached after backward (gradients and the FP32 gradient
  accumulation of a small model), compression lowers saved-activation memory
  but not the peak.
* Recomputed activations inside checkpointed layers, the LM head and the loss
  are not compressed.
