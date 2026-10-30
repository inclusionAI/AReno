Activation Compression Results
==============================

Measurements for :doc:`activation-compression`. Every run used one GPU. Raw
logs, run configurations and the summary tables are under
``examples/activation_compression/results``; ``examples/activation_compression/README.md``
lists the commands.

Environment
-----------

.. list-table::
   :widths: 30 70

   * - GPU
     - NVIDIA RTX PRO 5000 72GB Blackwell (one per run), driver 580.126.20
   * - Software
     - Python 3.12.11, PyTorch 2.8.0+cu128 (CUDA 12.8), Triton 3.4.0,
       Transformers 5.22.0, flash-attn 2.8.1
   * - Code
     - Base ``0de39d1``. The change was later rebased onto ``9188ec1`` (#643,
       #644, #637), where it adds and removes the same ``areno/`` lines; #637's
       MLX and SFT options only sit next to some of them. ``areno/`` tree
       ``d960eef4`` is this change on ``0de39d1``. It ran the compressed arms
       of the 44K-token, 87K-token and 1.7B benchmark rows, the long benchmark
       run, the 5% budget runs and the compressed 1.7B SFT run. Trees
       ``0b5fd0b0`` and ``83fdd760`` (the other benchmark runs, the
       periodically checked 0.6B SFT and RL runs, the resume runs, the BF16
       1.7B SFT run and the gradient-error check of the checked run) lack only
       the bracket bisection, which changes a plan only after a calibration
       round over budget; none of their calibrations had one. Tree ``fbcd53a2``
       (the other gradient-error checks, which do not calibrate) also predates
       the periodic check; tree ``dc5c7678`` (the 0.6B BF16 SFT and RL
       baselines and the once-calibrated arms) also predates the alias rule,
       which has nothing to act on with activation checkpointing. Every SFT and
       benchmark run's commit and tree is in its ``runs.json``.
   * - Models
     - ModelScope ``Qwen/Qwen3-0.6B`` and ``Qwen/Qwen3-1.7B`` (``master``),
       identified by safetensors SHA-256 in the ``runs.json`` files (0.6B:
       ``f47f7117...996874b``); BF16 weights and activations, FP32 master
       weights and optimizer math
   * - SFT data
     - ``yahma/alpaca-cleaned`` ``alpaca_data_cleaned.json`` (SHA-256
       ``bd844b82...fe878b``, 51,760 rows)
   * - RL data
     - GSM8K ``main`` train split, math-verify 0/1 reward

Numerical correctness
---------------------

``tests/test_activation_compression_cpu.py`` checks the PyTorch path against
an independent element-by-element NumPy FP32 reference (group minima and
steps, codes, outlier-channel selection) with the noise hash re-implemented in
exact Python integers. Codes, minima, steps and outlier indices match exactly
and reconstructions match bit for bit, for BF16, FP16 and FP32 inputs, 8, 4 and
2 bits, stochastic and nearest rounding, groups that straddle rows and partial
last groups. Further CPU tests cover the per-element error bound (at most one
step, half a step when rounding to nearest), unbiasedness over 400 seeds,
independence of the noise across seeds, byte accounting, slot eligibility
(weight views, buffer slices and integer tensors are never slots), saving a
tensor once when several operations save it, release of the originals after
the forward and of the packed data after backward, compression of a
non-reentrant checkpoint boundary with recomputation, the planner, calibration
within budget and its uncompressed fallback, aliases of uncompressed saves,
and plan save and load.

``tests/test_activation_compression_cuda.py`` checks that the Triton kernels
produce the PyTorch path's codes, minima, steps, outlier channels and
reconstructions bit for bit, and runs a three-layer Qwen3 built from AReno's
CUDA layers with and without activation checkpointing, with FlashAttention and
the native attention backend: an all-uncompressed plan reproduces the exact
gradients, all slots at 8 bits stay within 5% and closer than at 4 bits, the
live forward memory falls by the bytes compression did not store, every byte is
released after backward, and FlashAttention's deterministic backward repeats
every projection gradient exactly.

Gradient error
--------------

Relative L2 error of all parameter gradients against the uncompressed
gradients, measured with FlashAttention's deterministic backward. For context,
the default backward's run-to-run difference on the same microbatch (two
uncompressed passes) is reported alongside.

SFT with Qwen3-0.6B (microbatches of 64 Alpaca rows, about 9,000 to 12,000
tokens, activation checkpointing, 1% budget; seeds 1/2/3):

.. list-table::
   :header-rows: 1
   :widths: 26 26 24 24

   * - When
     - Plan
     - Relative gradient error
     - Default backward run-to-run
   * - Step 1 calibration
     - 69.5 / 73.8 / 70.9% of BF16 bytes stored
     - 0.77 / 0.66 / 0.78% (calibration), 0.76 / 0.73 / 0.81% (held-out)
     - 0.37 / 0.39 / 0.40%
   * - Check before step 51
     - step-1 plan
     - **1.63 / 1.34 / 1.62%**: over budget, recalibrated
     -
   * - Recalibration before step 51
     - 89.8 / 88.4 / 89.8% stored
     - 0.72 / 0.76 / 0.80% (calibration), 0.80 / 0.74 / 0.78% (held-out)
     -
   * - Checks before steps 101, 151, 201, 251
     - step-51 plan
     - 0.67% to 0.83% (12 checks, none over budget)
     -
   * - Step 300, eight held-out batches of 64 rows (seed 1)
     - step-51 plan
     - mean 0.77%, max 0.80% (8 of 8 within budget)
     - 0.80%
   * - Step 300 of a run that calibrated only once (seed 1)
     - step-1 plan
     - mean 1.65%, max 1.69% (0 of 8)
     - 0.80%
   * - Same checkpoint and batches
     - uniform 8 bits (57.8%) / uniform 4 bits (32.8%)
     - mean 2.05% / 21.6%, max 2.14% / 22.4%
     - 0.80%
   * - Base model, seed 1's step-1 plan, four held-out batches
     - step-1 plan
     - mean 0.80%, max 0.85%
     - 0.38%

The deterministic noise floor (two uncompressed passes with the
deterministic backward) was 0.05% to 0.18% throughout. The same plan's error
more than doubles during the first 50 steps, as does the default backward's
own run-to-run difference: the gradient shrinks and loses coherence while the
loss falls from 1.83 to 1.38, and incoherent perturbations, quantization noise
or BF16 rounding, grow relative to it. The periodic check catches this; after
one recalibration the error stayed between 0.67% and 0.83%. With a 5% budget
the checks before step 51 measured 7.1% to 8.4%, each run recalibrated once,
and the later checks measured 3.3% to 4.5%. ``gradient_error.py`` reproduces
the step-300 and base-model rows from the saved checkpoints.

Memory and step time
--------------------

SFT steps with identical batches in both arms (``--seed 1``); steps 1 and 2
are warm-up (step 1 of the adaptive arm also calibrates), steps 3 to 14 are
measured: median step time and the largest allocator peaks of the forward and
backward microbatches (``train_peak_*``). The optimizer-step peaks are the
same in both arms (12.42 GiB for 0.6B, 31.65 GiB for 1.7B). *Slots* are the
compressible saved activations of one microbatch, in BF16 and as stored
(codes, scales, exact channels and uncompressed slots).

.. list-table::
   :header-rows: 1
   :widths: 30 12 20 13 13 12

   * - Configuration (microbatch)
     - Arm
     - Slots: BF16 -> stored
     - Train peak allocated
     - Train peak reserved
     - Step time
   * - 0.6B, recompute, 64 rows (10.8K tokens)
     - BF16
     - 0.595 GiB
     - 14.41 GiB
     - 15.15 GiB
     - 2.062 s
   * -
     - adaptive
     - 0.439 GiB (73.8%)
     - 14.41 (+0.00)
     - 15.22 (+0.07)
     - 2.055 s (-0.3%)
   * - 0.6B, recompute, 256 rows (44K tokens)
     - BF16
     - 2.455 GiB
     - 17.94 GiB
     - 19.39 GiB
     - 8.691 s
   * -
     - adaptive
     - 1.398 GiB (57.0%)
     - **16.97 (-0.97)**
     - **17.93 (-1.46)**
     - 8.730 s (+0.4%)
   * - 0.6B, recompute, 512 rows (87K tokens)
     - BF16
     - 4.823 GiB
     - 22.68 GiB
     - 27.68 GiB
     - 17.28 s
   * -
     - adaptive
     - 2.705 GiB (56.1%)
     - **20.77 (-1.91)**
     - **22.95 (-4.73)**
     - 17.40 s (+0.7%)
   * - 0.6B, no recompute, 64 rows (10.8K tokens)
     - BF16
     - 2.319 GiB
     - 21.96 GiB
     - 24.71 GiB
     - 1.696 s
   * -
     - adaptive
     - 1.597 GiB (68.9%)
     - **21.01 (-0.95)**
     - **22.75 (-1.96)**
     - 1.743 s (+2.8%)
   * - 1.7B, recompute, 256 rows (44K tokens)
     - BF16
     - 4.910 GiB
     - 41.28 GiB
     - 44.26 GiB
     - 18.57 s
   * -
     - adaptive
     - 2.627 GiB (53.5%)
     - **39.13 (-2.15)**
     - **41.99 (-2.27)**
     - 18.66 s (+0.5%)

Stored bytes are below BF16 in every configuration. With 64-row microbatches
and activation checkpointing the step peak is reached after backward, when
gradients and the FP32 gradient accumulation are resident and the saved
activations already released, so it does not move; the more of the peak
saved activations make up, the more of their reduction reaches the peak
(87K tokens: -1.91 GiB of the -2.12 GiB not stored). Without recompute every
saved activation is a slot, but 48 of 113 stay uncompressed and the attention
kernel's saves are never compressed, so -0.72 GiB of slots gives -0.95 GiB of
peak. The overhead is the packing at save time and the unpacking in backward:

.. list-table::
   :header-rows: 1

   * - 65,536 x 1,024 BF16 tensor
     - 8 bits
     - 4 bits
     - 2 bits
   * - Pack (incl. outlier channels)
     - 0.57 ms
     - 0.51 ms
     - 0.49 ms
   * - Unpack
     - 0.31 ms
     - 0.28 ms
     - 0.26 ms
   * - Stored fraction
     - 57.8%
     - 32.8%
     - 20.3%

Step-1 calibration of the adaptive arms (deterministic backward):

.. list-table::
   :header-rows: 1

   * - Configuration
     - Plan (slots at 8 / 4 bits, uncompressed)
     - Calibration / held-out error
     - Default backward run-to-run
     - Passes
     - Time
   * - 0.6B, recompute, 10.8K tokens
     - 18 / 0, 11
     - 0.78 / 0.71%
     - 0.40%
     - 70
     - 61 s
   * - 0.6B, recompute, 44K tokens
     - 28 / 1, 0
     - 0.71 / 0.72%
     - 0.30%
     - 81
     - 330 s
   * - 0.6B, recompute, 87K tokens
     - 27 / 2, 0
     - 0.84 / 0.89%
     - 0.29%
     - 91
     - 753 s
   * - 0.6B, no recompute, 10.8K tokens
     - 34 / 31, 48 (113 slots)
     - 0.83 / 0.78%
     - 0.39%
     - 325
     - 228 s
   * - 1.7B, recompute, 44K tokens
     - 24 / 5, 0
     - 0.80 / 0.79%
     - 0.20%
     - 97
     - 851 s

The 1.7B plan shows the bracket at work: the first rescale overshot (46.6%
stored, 2.01%) and bisection settled at 53.5% and 0.80%; without it the loop
stopped at 69.5%. Without recompute, outputs saved by the attention kernel
make the output projection's input an alias, which is never a slot.

Over a longer run the plan does not stay at these early-training levels. With
44K-token microbatches and a check every 30 steps, the step-1 plan (57.0%
stored) measured **2.43%** at the step-31 check; the recalibrated plan stores
89.8% of the BF16 bytes (7 slots at 8 bits; 0.70 / 0.72%) and measured 0.73%
at the step-61 check, which matches the SFT runs below.

SFT validation
--------------

Qwen3-0.6B, 300 steps of 128 Alpaca rows in two microbatches of 64 (about
9,000 to 11,000 tokens each), activation checkpointing, cosine learning rate
1e-5 to 1e-6, no weight decay. The validation split is 1,000 held-out rows
scored by the live actor after every 50 steps. Each seed permutes the training
rows; both arms of a seed see identical batches.

.. list-table::
   :header-rows: 1
   :widths: 22 11 11 11 22 23

   * - Arm
     - Seed 1
     - Seed 2
     - Seed 3
     - Mean +- std
     - Change vs BF16
   * - BF16 (baseline)
     - 1.3330
     - 1.3326
     - 1.3327
     - 1.3328 +- 0.0002
     -
   * - **Adaptive, 1% budget**
     - 1.3331
     - 1.3327
     - 1.3329
     - 1.3329 +- 0.0002
     - **+0.006%** (+0.001, +0.005, +0.013% per seed)
   * - Adaptive, 1%, calibrated once
     - 1.3331
     - 1.3327
     - 1.3328
     - 1.3329 +- 0.0002
     - +0.006%
   * - Adaptive, 5% budget
     - 1.3332
     - 1.3328
     - 1.3328
     - 1.3329 +- 0.0002
     - +0.009%

Final validation loss after 300 steps (initial: 1.8263). The acceptance bound
is an increase of at most 2%. At every evaluation (steps 50 to 300) each
seed's validation loss is within 0.0004 of the BF16 run with the same seed.

Saved-activation bytes stored relative to BF16, per arm: the 1% runs store
69.5% to 73.8% until step 50 and 88.4% to 89.8% after the recalibration; the
once-calibrated runs keep 69.5% to 73.8% throughout (and exceed the budget
later, see above); the 5% runs store 50.9% to 51.8% at first and 55.2% to
56.1% after one recalibration each (two or three slots at 4 bits, later
checks 3.3% to 4.5%). The SFT runs shared their GPUs with the RL runs, so
their step times are not comparable; overhead is measured in the benchmark
above.

Qwen3-1.7B, seed 1, same schedule: final validation loss 1.1078 (BF16) and
1.1082 adaptive, **+0.03%** (initial 1.9134). Its step-1 plan stored 55.2% of
the BF16 bytes (26 slots at 8 bits, 3 at 4 bits; 0.80% / 0.74% error), the
check before step 51 measured 2.98% and recalibrated to 86.9% (9 slots at 8
bits), and the later checks measured 0.72% to 0.81%.

Reinforcement learning
----------------------

GSPO on GSM8K with Qwen3-0.6B (thinking disabled): 100 steps of 32 prompts with
eight samples each, at most 1,024 new tokens, learning rate 1e-6, activation
checkpointing, training microbatches of 16 sequences (5,200 to 9,600 tokens on
average per step, median 6,600), rollout seed ``seed * 1_000_003 + step``. The
final-window reward is the mean of the last ten steps' mean 0/1 reward. The
adaptive plans compress 5, 4 and 3 of 29 slots (92.7%, 94.2% and 95.6% of BF16
bytes stored; calibration / held-out errors 0.93 / 0.76%, 0.73 / 0.74% and
0.70 / 0.48%), as small microbatches leave little room under a 1% budget. With
the default check every 50 steps the plan stayed within budget at step 50 in
all three runs; a second set of runs that calibrated only once (``areno/``
tree ``dc5c7678``, equivalent to ``activation_compression_check_interval=0``)
is listed for comparison.

.. list-table::
   :header-rows: 1
   :widths: 16 11 11 11 18 15 18

   * - Arm
     - Seed 1
     - Seed 2
     - Seed 3
     - Mean +- std
     - Change vs BF16
     - ``logp_abs_diff_mean``
   * - BF16
     - 0.7207
     - 0.7125
     - 0.7086
     - 0.7139 +- 0.0062
     -
     - 0.0112
   * - **Adaptive**
     - 0.7258
     - 0.7184
     - 0.7258
     - 0.7233 +- 0.0043
     - **+0.0094**
     - 0.0114
   * - Adaptive, calibrated once
     - 0.7223
     - 0.7199
     - 0.7250
     - 0.7224 +- 0.0025
     - +0.0085
     - 0.0113

The acceptance bound is a final-window reward no more than 0.05 below the
baseline. ``logp_abs_diff_mean`` is the trainer-native
``train/logp_abs_diff_mean`` averaged over all steps (largest single step:
0.0128 BF16, 0.0131 adaptive, 0.0132 calibrated once). Compression only
changes what backward sees,
so the training forward, and with it the rollout/training log-probability
mismatch, is unchanged.

Mean reward of each 10-step window, averaged over the three seeds:

.. list-table::
   :header-rows: 1

   * - Steps
     - 1-10
     - 11-20
     - 21-30
     - 31-40
     - 41-50
     - 51-60
     - 61-70
     - 71-80
     - 81-90
     - 91-100
   * - BF16
     - 0.575
     - 0.615
     - 0.695
     - 0.665
     - 0.709
     - 0.688
     - 0.716
     - 0.720
     - 0.746
     - 0.714
   * - Adaptive
     - 0.587
     - 0.623
     - 0.700
     - 0.663
     - 0.713
     - 0.691
     - 0.720
     - 0.713
     - 0.744
     - 0.723
   * - Adaptive, calibrated once
     - 0.579
     - 0.619
     - 0.707
     - 0.662
     - 0.707
     - 0.686
     - 0.722
     - 0.715
     - 0.751
     - 0.722

Save and resume
---------------

Seed 1 of each RL arm saved checkpoints after steps 50 and 100 through the
stock trainer (``--save-interval 50``); a second ``areno train`` process then
resumed from the step-50 checkpoint for the remaining 50 steps
(``examples/activation_compression/train_seeded.py --resume-step 50
--resume-prompts 1600``).

* These resume runs use the once-calibrated seed-1 arms. The adaptive
  checkpoint holds ``activation_compression.json``. The resumed
  process logged ``reusing the plan``, ran no calibration, compressed exactly
  the same slots (92.73% of BF16 bytes stored on every step, as in steps 51
  to 100 of the uninterrupted run) and had no plan misses.
* Neither arm continues its uninterrupted run step for step, for two reasons
  outside this change: AReno checkpoints hold weights only, so the optimizer
  restarts, and Qwen3-0.6B declares ``tie_word_embeddings``, so the loader
  copies the embedding into the separately trained LM head. Both arms are
  affected alike:

.. list-table::
   :header-rows: 1

   * - Arm
     - Mean reward, steps 51-100
     - Same steps, resumed
     - Final-window reward, uninterrupted / resumed
     - ``logp_abs_diff_mean``, uninterrupted / resumed
   * - BF16
     - 0.7212
     - 0.7121
     - 0.7207 / 0.7113
     - 0.0114 / 0.0111
   * - Adaptive
     - 0.7182
     - 0.7095
     - 0.7223 / 0.7125
     - 0.0115 / 0.0112

Error budget
------------

The budget trades memory for gradient fidelity; quality was unaffected over
the range tried (SFT table above). With a 5% budget the SFT plans keep two or
three slots at 4 bits and store 55% to 56% of the BF16 bytes for the whole
run, against 88% to 90% with the default 1% once training has settled; the
final validation loss is the same (+0.009%). The 1% default meets the issue's
gradient-error target, and ``--activation-compression-error-budget`` raises it
for configurations where the activation memory matters more.
