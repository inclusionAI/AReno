GaLore single-GPU full-parameter training
=======================================

GaLore projects matrix gradients, applies AdamW in the reduced coordinates,
and reconstructs updates to the original trainable weights. It retains FP32
master weights. It does not create adapters or freeze base weights.

See :doc:`the measured Qwen3-0.6B comparison <../reports/galore-sft-frontier>`
for the fixed three-seed quality, memory, timing and checkpoint evidence.

Enable through the public SDK's existing optimizer dictionary::

    from areno import Trainer
    from areno.api import CudaConfig

    config = CudaConfig(
        devices=[0],
        optimizer={
            "lr": 1e-5,
            "galore_rank": 128,
            "galore_update_proj_gap": 200,
            "galore_scale": 1.0,
        },
        runtime={"attn_backend": "native", "compile_model": False},
    )
    trainer = Trainer(world_size=1, model_path="/path/to/local/model", custom_config=config)
    trainer.init()
    # TrainSequence batches and trainer.train(...) use the ordinary SDK.
    trainer.save_checkpoint("/path/to/checkpoint")
    trainer.close()

The initial scope is a single CUDA GPU, dense models, resident optimizer
state, and full-parameter training. Low-bit/paged AdamW, LoRA, DP/TP, MoE,
explicit optimizer offload, and ``keep_rollout_state=False`` are rejected.
No additional runtime dependency is required. Existing CLI flags are unchanged.

Projection and update convention
-------------------------------

For an m-by-n gradient, the right singular-vector basis is used if m >= n;
otherwise the left singular-vector basis is used. Decomposition and updates
use FP32. Refresh occurs before a parameter's first update, then whenever its
number of completed updates is divisible by ``galore_update_proj_gap``.
Parameters without gradients do not advance their individual step counters.
Projected moments are retained across basis refreshes by default, following the original
GaLore projection rule. AdamW uses PyTorch-style bias correction, epsilon on
the corrected denominator, and decoupled decay before the adaptive update.

Vectors, tensors with more than two dimensions, ranks at least as large as
the smaller dimension, and projections whose moments plus basis would be at
least as large as ordinary moments use standard FP32-master AdamW.
By default the rank applies to every eligible matrix, including large vocabulary
weights; their decomposition cost is included in refresh timing and peak memory.
Set ``galore_exclude_vocab=True`` to keep ordinary AdamW on input embeddings and
output vocabulary weights while projecting eligible transformer matrices. The
worker identifies actual vocabulary modules rather than guessing from names or
matrix sizes. Tied input/output weights retain one optimizer owner and receive
one update and one decay. This option increases full-sized moment storage for
the vocabulary weights; measure its quality and total-memory tradeoff separately.

For a 1024-by-1024 FP32 matrix at rank 128, two projected moments plus a
persistent basis occupy 1.5 MiB, versus 8 MiB for two full moments. Both paths
also require master weights. Full gradients, activations, reconstruction
temporaries, SVD workspace, and CUDA allocator reservations remain separate
costs. State savings do not establish total peak-memory or speed improvements.

Optional execution backends
---------------------------

The default projector is exact SVD. ``galore_projector_backend="recycled"``
uses a smaller-side recycled range finder, independent per-parameter CPU RNG,
and a small Ritz SVD. ``galore_projector_oversampling=16`` and
``galore_projector_iterations=1`` control its work. The current policy applies
a final canonical QR before sign alignment; this changes the coordinate rule
slightly and is versioned in checkpoints. Legacy recycled checkpoints retain
their earlier rule. The whole optimizer scopes FP32 products to highest
precision and restores the caller's precision setting after the update.

For contiguous BF16 weights, ``galore_compact_master=True`` stores exact FP32
master information as BF16 model high words, uint16 low words, and packed
rounding carries. It rejects nonfinite values and unsupported conversions.
FP32 and noncontiguous weights retain ordinary masters. Pair this option with
``galore_tile_rows=16384``, ``galore_release_gradients=True``, and
``galore_fused_writeback=True`` for bounded reconstruction and fused CUDA
writeback. Global clipping finishes before any gradient is released.

``galore_transport_moments=True`` enables an experimental bias-corrected
mean/diagonal-second-moment transport at refreshes. This is a separately
tested statistical approximation, without error feedback; it does not
implement full LDAdam. It defaults to False. Scale, rank, refresh gap and
transport alter the optimization trajectory and require matched quality
comparisons. No backend promises a universal quality or throughput gain.

Checkpoint resume
-----------------

GaLore checkpoints include ``training_state.pt`` and ``training_state.json``
alongside ordinary model safetensors. These sidecars contain FP32 masters,
moments, bases, per-parameter steps, optimizer hyperparameters, worker step,
learning-rate schedule, parameter identities, worker Torch/CUDA RNG, and
versioned projector/backend policies. Recycled per-parameter RNG and compact
master information are preserved exactly; masters have canonical FP32
checkpoint representation.
Use the same model/configuration and explicitly opt into restoration::

    config.optimizer["resume_training_state"] = True
    trainer = Trainer(world_size=1, model_path="/path/to/checkpoint", custom_config=config)
    trainer.init()

Loading only model weights remains the default. Missing or mismatched training
state raises an error when resume is requested. To save ordinary AdamW/low-bit
state for a single-GPU control, set ``save_training_state=True`` in the optimizer
dictionary. The application still owns its dataset cursor/order, host Python
and NumPy RNG, and any externally managed reward/rollout state.

Vocabulary-exclusion checkpoints additionally store a versioned target policy
and the per-parameter identity mask. Restore rejects a changed policy or target
ownership before replacing live optimizer state. Legacy all-matrix checkpoints
retain their existing format and behavior. Scalar optimizer hyperparameters must
be finite; malformed NaN or infinity values are rejected before restoration.

Verification and bounded trial
------------------------------

``pytest tests/test_galore_cpu.py`` checks FP32 updates against an independent
reference built with ``torch.optim.AdamW`` and tensor contractions, tall/wide
and square matrices, fallback updates, BF16 masters, resume across refresh
boundaries, state bytes, and training sidecars.

``scripts/bench_galore.py`` runs fixed short repository text through the public
Trainer, with baseline, low-bit, and GaLore variants. Its context is fixed at
64 tokens; it rejects examples whose response targets do not fit. For the
response-preserving Dolly comparison at context512, use the separate
``scripts/galore_sft/README.md`` entrypoints and pinned data preparation.
Use separate processes
and output directories, fixed model bytes, and seeds 0, 1, and 2. Each run
warms up for two steps and records at least ten further steps when ``--steps
12`` is used. ``--save`` writes a checkpoint; ``--resume`` restores it. Native
profiling records allocated/reserved CUDA peaks and synchronized step time.
The benchmark scores each batch before SFT so the native diagnostic is real.
``--data-dir`` reads fixed train/validation JSONL folds verbatim. Backend
flags permit reproducing a separately calibrated configuration, for example::

    python scripts/bench_galore.py --model /path/to/model --output /path/to/run \
        --optimizer galore --seed 0 --steps 12 --data-dir /path/to/fixed-folds \
        --rank 128 --gap 4 --scale 4 --projector-backend recycled \
        --compact-master --fused-writeback --transport-moments --save

Repeat with seeds 1 and 2 and matched ``--optimizer adamw`` / ``adam8``
controls. Gap 4 deliberately stresses refresh boundaries. The example's scale
and optional transport describe one short-run configuration, not recommended
defaults for other datasets or budgets. Use separate train/eval processes and
training-only selection when comparing calibrated candidates.

Native SFT emits ``train/logp_abs_diff_mean`` when every supplied sequence has
complete token-aligned logprobs. For a same-model consistency diagnostic,
obtain these with ``trainer.score_logprobs("actor", token_rows)`` immediately
before the SFT update. Missing/mixed/incomplete lists suppress this optional
metric. Caller-provided placeholder zeros do not establish consistency.
The loss and its gradients are unchanged by the diagnostic. Profiled CUDA
training time excludes this preceding score call; also report whole-iteration
time when using it. Short local-text runs do not establish convergence or
general-task quality, even when their fixed-budget quality gate passes.

Dense Qwen3 honors ``tie_word_embeddings`` with one shared Parameter and one
optimizer owner. This is common to every optimizer: never count correcting
model identity as a GaLore-specific memory saving. Old untied checkpoints
with a true tie flag omitted a separately trained head from model-only saves;
they cannot be used as equivalent corrected-model resumes.

References: `GaLore paper <https://proceedings.mlr.press/v235/zhao24s.html>`_,
`original projector <https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py>`_.
