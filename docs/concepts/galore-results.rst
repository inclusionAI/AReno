GaLore results
==============

Measurements for :doc:`galore`. Every run used one GPU. Raw logs, run
configurations and the summary tables are under ``examples/galore/results``;
``examples/galore/README.md`` lists the commands.

Environment
-----------

.. list-table::
   :widths: 30 70

   * - GPU
     - NVIDIA RTX PRO 5000 72GB Blackwell (one per run), driver 580.126.20
   * - Software
     - Python 3.12.11, PyTorch 2.8.0+cu128 (CUDA 12.8), Transformers 5.22.0,
       flash-attn 2.8.3
   * - Code
     - ``areno/`` tree ``118d31d9`` for every measurement (identical to this PR)
   * - Models
     - ModelScope ``Qwen/Qwen3-0.6B`` and ``Qwen/Qwen3-1.7B`` (``master``),
       identified by safetensors SHA-256 in the SFT ``runs.json`` files (0.6B:
       ``f47f7117...996874b``); BF16 weights, FP32 master weights and optimizer math
   * - SFT data
     - ``yahma/alpaca-cleaned`` ``alpaca_data_cleaned.json`` (SHA-256
       ``bd844b82...fe878b``, 51,760 rows)
   * - RL data
     - GSM8K ``main`` train split, math-verify 0/1 reward

Numerical correctness
---------------------

``tests/test_galore_optimizer_cpu.py`` checks the optimizer against an
independent float64 GaLore AdamW that uses the same projection rule but computes
its basis with a full SVD instead of the optimizer's Gram eigendecomposition:

* FP32 fused and square weights over ten steps with refreshes at steps 1, 4, 7
  and 10, both projection sides and weight decay: every step matches within
  ``atol=1e-6, rtol=1e-4``. Ineligible bias and embedding tensors match
  ``torch.optim.AdamW`` within the same tolerance.
* Resume: saving after steps 2, 3 or 4 (refresh interval 3, so the next step is
  mid-interval, a refresh, or just after one), reloading into an optimizer with
  different initial weights through ``torch.save``/``torch.load(weights_only=True)``
  and finishing eight steps reproduces the uninterrupted weights, bases, moments
  and FP32 masters bit for bit, for FP32 and BF16 weights.
* A fused ``[12; 6] x 10`` weight updates exactly like two separately optimized
  weights, which is the per-projection semantics of unfused modules.
* ``tests/test_galore_optimizer_cuda.py`` runs the CUDA optimizer with TF32 matrix
  multiplication enabled and matches the CPU optimizer within the same tolerance.

Save and resume through the public ``Trainer`` (GSPO with Qwen3-0.6B, same
settings as the RL runs below, seed 7, ``examples/galore/train_seeded.py``):
an uninterrupted 12-step run saved weights and optimizer state after steps 4, 8
and 12, and a second process resumed from the step-4 checkpoint with
``Trainer.load_optimizer_state`` for steps 5 to 12. GaLore used refresh interval
4, so the first resumed step is a projection refresh.

* Round trip: loading the step-4 checkpoint into a new ``Trainer`` and saving
  the optimizer state again reproduces all 764 GaLore state tensors (FP32
  masters, low-rank and AdamW moments, bases, step counters) and all 400 AdamW
  state tensors bit for bit (``check_state_roundtrip.py``).
* The first resumed step matches the uninterrupted step 5 exactly in everything
  computed before backward: mean reward, loss and trainer-native
  ``train/logp_abs_diff_mean``.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Step 5, GaLore
     - Uninterrupted
     - Resumed
     - Second uninterrupted run
   * - ``rollout/rewards_mean``
     - 0.632812
     - 0.632812
     - 0.578125
   * - ``train/loss``
     - -2.50060e-07
     - -2.50060e-07
     - 9.96515e-08
   * - ``train/logp_abs_diff_mean``
     - 0.0117793
     - 0.0117793
     - 0.0119708
   * - ``train/grad_norm``
     - 0.945204
     - 0.945090
     - 0.938938

The CUDA backward pass is not bitwise deterministic: two uninterrupted GaLore
runs with the same seed share their first rollout and loss exactly, yet their
first gradient norms differ (0.656622 and 0.656578), and by step 5 their
trajectories have drifted apart. Over steps 5 to 12 the resumed run differs from
the uninterrupted one by about as much as two uninterrupted runs differ from each
other:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Largest difference, steps 5-12
     - Resumed vs uninterrupted
     - Two uninterrupted runs
   * - GaLore ``rollout/rewards_mean``
     - 0.047
     - 0.055
   * - GaLore ``train/logp_abs_diff_mean``
     - 0.00065
     - 0.00031
   * - GaLore ``train/grad_norm``
     - 0.138
     - 0.104
   * - AdamW ``rollout/rewards_mean``
     - 0.055
     - 0.047
   * - AdamW ``train/logp_abs_diff_mean``
     - 0.00040
     - 0.00050
   * - AdamW ``train/grad_norm``
     - 0.123
     - 0.257

Optimizer state
---------------

For a 1024 x 1024 weight at rank 128 (measured by
``examples/galore/microbench.py``):

.. list-table::
   :header-rows: 1
   :widths: 22 26 30 22

   * - Weights
     - AdamW state
     - GaLore state
     - GaLore / AdamW
   * - FP32
     - 8.00 MiB (two moments)
     - 1.50 MiB (basis 0.50 + moments 1.00)
     - 0.188
   * - BF16
     - 10.13 MiB (+2.13 MiB master bits)
     - 3.63 MiB (+2.13 MiB master bits)
     - 0.358

A steady GaLore step on this matrix took 0.09 ms (AdamW: 0.03 ms) and a
projection-refresh step 8.6 ms, with at most 28.6 MiB (FP32) or 35.0 MiB (BF16)
of transient workspace.

Persistent optimizer state of the whole model (bytes reported by the
optimizers at the end of the benchmark runs):

.. list-table::
   :header-rows: 1
   :widths: 18 16 16 16 16 18

   * - Model (trainable)
     - AdamW
     - GaLore r128
     - AdamW8bit
     - AdamW4bit
     - GaLore detail
   * - Qwen3-0.6B (751.6M)
     - 7.09 GiB
     - 4.31 GiB
     - 2.30 GiB
     - 0.38 GiB
     - 440.4M params in 196 blocks: bases 0.10, moments 0.41 (3.28 full rank),
       master bits 0.87; AdamW tensors 2.94
   * - Qwen3-1.7B (2,031.7M)
     - 19.16 GiB
     - 9.53 GiB
     - 5.62 GiB
     - 1.01 GiB
     - 1,409.3M params in 196 blocks: bases 0.16, moments 0.71 (10.50 full
       rank), master bits 2.79; AdamW tensors 5.87

The AdamW column is computed as FP32 moments plus FP32-master low bits for every
parameter; the measured gap in allocated memory after a step between AdamW and
GaLore (below) matches the computed difference within 0.04 GiB. AdamW8bit and
AdamW4bit keep no FP32 master. The AReno Qwen3 LM head is a
separate parameter from the token embedding (311.2M and 622.5M AdamW parameters
in total), and both keep FP32 AdamW under GaLore.

Memory and step time
--------------------

Fixed-budget SFT steps (32 rows, at most 768 tokens each, eight-row
microbatches) with one process per GPU. Each optimizer ran 22 steps; the first
two are warm-up, and GaLore refreshed its projections at steps 11 and 21
(``--galore-update-proj-gap 10``). Allocated and reserved peaks come from the
PyTorch caching allocator; ``train`` covers forward/backward of a step and
``optimizer`` the update alone.

.. list-table::
   :header-rows: 1
   :widths: 16 13 13 15 15 14 14

   * - Optimizer
     - Allocated after step
     - Train peak alloc / reserved
     - Optimizer peak alloc / reserved
     - Step time
     - Optimizer step
     - Peak vs AdamW
   * - **Qwen3-0.6B**
     -
     -
     -
     -
     -
     -
   * - AdamW
     - 9.61
     - 14.41 / 14.63
     - 12.42 / 14.63
     - 0.679 s
     - 0.028 s
     -
   * - GaLore, steady
     - 6.87
     - 11.67 / 12.38
     - 9.67 / 12.38
     - 0.710 s
     - 0.063 s
     - -2.74 (-19.0%)
   * - GaLore, refresh
     - 6.87
     - 11.66 / 11.88
     - 9.67 / 11.88
     - 2.433 s
     - 1.805 s
     -
   * - AdamW8bit
     - 4.83
     - 9.63 / 10.41
     - 7.65 / 10.41
     - 0.665 s
     - 0.028 s
     - -4.78 (-33.2%)
   * - AdamW4bit
     - 2.92
     - 5.81 / 6.04
     - 4.33 / 6.04
     - 0.757 s
     - 0.121 s
     - -8.60 (-59.7%)
   * - **Qwen3-1.7B**
     -
     -
     -
     -
     -
     -
   * - AdamW
     - 24.08
     - 36.60 / 37.10
     - 31.65 / 37.10
     - 1.469 s
     - 0.074 s
     -
   * - GaLore, steady
     - 14.49
     - 27.01 / 27.55
     - 22.06 / 27.55
     - 1.659 s
     - 0.255 s
     - -9.59 (-26.2%)
   * - GaLore, refresh
     - 14.49
     - 27.01 / 27.45
     - 22.06 / 27.45
     - 5.424 s
     - 4.058 s
     -
   * - AdamW8bit
     - 10.52
     - 23.05 / 23.64
     - 18.10 / 23.64
     - 1.454 s
     - 0.076 s
     - -13.55 (-37.0%)
   * - AdamW4bit
     - 5.94
     - 13.60 / 14.21
     - 9.74 / 14.21
     - 1.753 s
     - 0.440 s
     - -23.00 (-62.8%)

Memory columns are GiB (maxima, or medians for allocated-after-step); times are
medians. Steady rows cover 18 (GaLore) or 20 measured steps, refresh rows the two
measured refresh steps.

* GaLore lowers the train peak by 2.74 GiB on Qwen3-0.6B and 9.59 GiB on
  Qwen3-1.7B, about the same amount as its persistent-state savings.
* Refresh steps reach the same allocated and reserved peaks as steady steps:
  the refresh workspace (at most 146 MiB per block, below) is smaller than the
  other transient buffers of an optimizer step.
* A refresh costs 1.8 s (0.6B) or 4.1 s (1.7B), 9 or 20 ms per step amortized
  over the default 200-step interval. A full SVD of every block would take
  about 10.8 s (0.6B) or 32.6 s (1.7B) per refresh by the per-block timings
  below.
* Between refreshes the GaLore update is slower than the fused AdamW kernel
  (0.063 s versus 0.028 s on 0.6B, 0.255 s versus 0.074 s on 1.7B). A CUDA
  profile of the update on 1.7B-shaped blocks spends about 90% of kernel time
  in full-size elementwise passes (FP32-master decode and encode with packed
  rounding bits, weight decay, copies) and about 7% in the projection GEMMs; a
  fused update kernel would remove most of it. Total step time rises by 5%
  (0.6B) and 13% (1.7B).
* AdamW8bit and AdamW4bit use less memory, but they keep no FP32 master; see
  the learning results below.

The full FP32 gradient is resident before every optimizer step (2.80 GiB for
the 752M trainable parameters of Qwen3-0.6B, 7.57 GiB for the 2,032M of
Qwen3-1.7B), so GaLore lowers the train peak by the moment savings only. A refresh replaces the
low-rank step with one eigendecomposition per block. Its workspace, measured per
block shape against a full SVD of the same gradient:

.. list-table::
   :header-rows: 1
   :widths: 30 20 25 25

   * - Block
     - Shape (side)
     - GaLore basis refresh
     - Full ``torch.linalg.svd``
   * - 0.6B ``q_proj``
     - 2048 x 1024 (right)
     - 8.5 ms, 32.6 MiB
     - 54.3 ms, 36.3 MiB
   * - 0.6B ``k_proj``/``v_proj``
     - 1024 x 1024 (right)
     - 8.5 ms, 28.6 MiB
     - 54.2 ms, 20.3 MiB
   * - 0.6B ``o_proj``
     - 1024 x 2048 (left)
     - 8.5 ms, 32.6 MiB
     - 54.6 ms, 44.3 MiB
   * - 0.6B ``gate_proj``/``up_proj``
     - 3072 x 1024 (right)
     - 8.6 ms, 36.6 MiB
     - 56.5 ms, 49.3 MiB
   * - 0.6B ``down_proj``
     - 1024 x 3072 (left)
     - 8.6 ms, 36.6 MiB
     - 56.6 ms, 62.0 MiB
   * - 1.7B ``q_proj``/``o_proj``
     - 2048 x 2048 (right)
     - 22.7 ms, 114.0 MiB
     - 207.6 ms, 80.6 MiB
   * - 1.7B ``k_proj``/``v_proj``
     - 1024 x 2048 (left)
     - 8.5 ms, 32.6 MiB
     - 54.8 ms, 44.3 MiB
   * - 1.7B ``gate_proj``/``up_proj``
     - 6144 x 2048 (right)
     - 23.4 ms, 146.0 MiB
     - 212.9 ms, 186.3 MiB
   * - 1.7B ``down_proj``
     - 2048 x 6144 (left)
     - 23.7 ms, 146.0 MiB
     - 212.7 ms, 234.3 MiB

Workspace is the allocator peak above the gradient that is already resident;
blocks are refreshed one at a time.

The benchmark GPUs have discrete memory, so no CPU offload or unified memory is
involved.

Supervised fine-tuning
----------------------

Each run trains 400 steps of 32 Alpaca rows (12,800 rows) with learning rate
1e-5 decaying to 1e-6 (cosine), no weight decay, and prompt/response limits of
256/512 tokens. Validation loss is the token-mean response NLL of 1,000 fixed
held-out rows, scored by the live actor through ``Trainer.score_logprobs``.
Seeds permute the training rows; every optimizer sees the same rows for a given
seed.

The GaLore scale was chosen on a separate split (``--split-seed 999``, seed 0)
before the acceptance runs:

.. list-table::
   :header-rows: 1
   :widths: 46 27 27

   * - Optimizer (dev split, seed 0)
     - Final validation loss
     - Change vs AdamW
   * - AdamW
     - 1.3578
     -
   * - GaLore rank 128, scale 1, refresh 100
     - 1.3783
     - +1.51%
   * - GaLore rank 128, scale 2, refresh 100
     - 1.3680
     - +0.75%
   * - GaLore rank 128, scale 2, refresh 200
     - 1.3674
     - +0.71%
   * - **GaLore rank 128, scale 4, refresh 100**
     - 1.3622
     - +0.32%
   * - GaLore rank 256, scale 2, refresh 100
     - 1.3623
     - +0.33%
   * - AdamW8bit
     - 1.4098
     - +3.83%
   * - AdamW4bit
     - 1.4099
     - +3.84%

All arms start from validation loss 1.8182 on this split. The refresh interval
(100 or 200) barely matters; the scale does, as expected from the
``sqrt(r / min(m, n)) = 0.35`` update-size ratio at rank 128.

Acceptance runs (``--split-seed 1234``, seeds 1 to 3), GaLore rank 128, refresh
interval 100, scale 4:

.. list-table::
   :header-rows: 1
   :widths: 16 12 12 12 18 18 12

   * - Optimizer
     - Seed 1
     - Seed 2
     - Seed 3
     - Mean +- std
     - Change vs AdamW
     - Train peak
   * - AdamW
     - 1.3570
     - 1.3539
     - 1.3564
     - 1.3557 +- 0.0016
     -
     - 14.43 GiB
   * - **GaLore**
     - 1.3617
     - 1.3594
     - 1.3610
     - 1.3607 +- 0.0012
     - **+0.37%** (+0.35, +0.41, +0.34% per seed)
     - 11.68 GiB
   * - AdamW8bit
     - 1.4109
     - 1.4108
     - 1.4106
     - 1.4107 +- 0.0001
     - +4.06%
     - 9.64 GiB
   * - AdamW4bit
     - 1.4108
     - 1.4106
     - 1.4104
     - 1.4106 +- 0.0002
     - +4.05%
     - 5.82 GiB

Validation loss starts at 1.8263. GaLore stays within the 2% acceptance bound
on every seed. Both low-bit AdamW variants plateau near 1.411: they update the
BF16 weights directly, and at this learning rate an Adam update (at most about
1e-5) is below half a BF16 unit in the last place of a weight near 0.02 (about
6e-5), so most updates round away. GaLore keeps the FP32 master and does not
have this problem.

The same comparison on Qwen3-1.7B (identical data, budget and GaLore settings;
one run per GPU):

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 12 18 18 14

   * - Optimizer
     - Seed 1
     - Seed 2
     - Seed 3
     - Mean +- std
     - Change vs AdamW
     - Train peak
   * - AdamW
     - 1.1260
     - 1.1232
     - 1.1252
     - 1.1248 +- 0.0014
     -
     - 36.62 GiB
   * - **GaLore**
     - 1.1296
     - 1.1278
     - 1.1290
     - 1.1288 +- 0.0009
     - **+0.36%** (+0.33, +0.41, +0.34% per seed)
     - 27.03 GiB

Validation loss starts at 1.9134. Median step time was 1.416 s for AdamW and
1.585 s for GaLore.

Reinforcement learning
----------------------

GSPO on GSM8K with Qwen3-0.6B (thinking disabled): 100 steps of 32 prompts with
eight samples each, at most 1,024 new tokens, learning rate 1e-6, rollout seed
``seed * 1_000_003 + step``. GaLore uses rank 128, refresh interval 50 (so every
run crosses a refresh at step 51) and scale 4. The final-window reward is the
mean of the last ten steps' mean 0/1 reward.

.. list-table::
   :header-rows: 1
   :widths: 14 12 12 12 18 16 16

   * - Optimizer
     - Seed 1
     - Seed 2
     - Seed 3
     - Mean +- std
     - Change vs AdamW
     - ``logp_abs_diff_mean``
   * - AdamW
     - 0.7148
     - 0.7184
     - 0.7164
     - 0.7165 +- 0.0018
     -
     - 0.0111
   * - **GaLore**
     - 0.7461
     - 0.7402
     - 0.7410
     - 0.7424 +- 0.0032
     - **+0.026**
     - 0.0109

Final-window reward per seed (steps 91 to 100). ``logp_abs_diff_mean`` is the
trainer-native ``train/logp_abs_diff_mean`` averaged over all steps (largest
single step: 0.0131 AdamW, 0.0130 GaLore). The rollout/training mismatch is the
same with either optimizer, so GaLore-updated weights reach the colocated
rollout engine like AdamW-updated ones.

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
   * - AdamW
     - 0.581
     - 0.622
     - 0.704
     - 0.666
     - 0.713
     - 0.687
     - 0.725
     - 0.717
     - 0.749
     - 0.717
   * - GaLore
     - 0.591
     - 0.641
     - 0.724
     - 0.689
     - 0.745
     - 0.712
     - 0.747
     - 0.745
     - 0.770
     - 0.742

The acceptance bound is a final-window reward no more than 0.05 below AdamW;
GaLore is 0.026 above it. With scale 4 its updates are slightly larger than
AdamW's at the same learning rate (``4 * sqrt(128 / 1024) = 1.4``), which is the
likely reason. Median step time was 19.5 to 20.3 s for AdamW and 19.5 to 22.4 s
for GaLore.
