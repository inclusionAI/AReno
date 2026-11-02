Selected GaLore scale1 confirmation
==================================

This follow-up replaces the scale2 configuration as the primary quality result.
The earlier ``galore-sft-qwen3`` report remains a historical experiment with its
own source, checkpoints, test results and timing controls. Do not combine its
test values with the new validation results below.

The selected configuration uses all eligible matrices, rank128, refresh gap200,
scale1, the recycled projector with finalQR, coordinate-moment transport,
lossless compact logical FP32 masters, tile16384 fused writeback and consumed
gradient release. Experimental model-owned masters were disabled in the measured
source (``ARENO_EXPERIMENT_MODEL_OWNED_MASTER=0``); the delivered optimizer
omits that private experimental path. This is an optimizer configuration
selected using development validation, not a new optimization algorithm.

Model, data and training conditions
----------------------------------

Qwen3-0.6B is pinned to ModelScope revision
``09b42cad3d112e832108974449ccb5e8e0f5b5d1`` and model weight SHA256
``f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b``.
All 596,049,920 unique tied parameters train; model matrices are BF16 and norms
are FP32. The dataset is ModelScope Dolly at revision
``40789acc12fc986e9145f87d159f19e2f1d75c6a``. Fixed splits contain
13,544/708/743 train/validation/test examples. Exact context/instruction-proxy
groups are disjoint; article, near-duplicate and pretraining separation are not
established. Validation/test contain 52,181/55,562 response targets.

The budget is 1,024 updates, seeds0/1/2, context512, batch4/microbatch2,
constant lr1e-5, betas(0.9,0.95), weight decay0.01 and global clip1.
Each seed's batches, sample indices and target counts are exactly paired across
all methods. Scoring and loss use the public Trainer; native
``train/logp_abs_diff_mean`` is zero in every recorded training update.
The existing AReno 8-bit AdamW is the low-bit control.

All nine new quality runs used measured Python runtime SHA256
``7da069e9aba272313b2e7b3f5325923b4779adf5afb50f9b98098aae9127faf2``.
The runtime digest is computed by concatenating sorted relative ``areno/*.py``
paths and contents. The evidence archive includes that exact experimental
Python source, frozen harnesses, actual launch vectors and per-run environment
metadata. This digest identifies the measured experimental tree; a later final
PR commit is a separate identity requiring its own equivalence/acceptance record.
PyTorch is2.6.0+cu124 with CUDA runtime12.4 on A100-SXM4-40GB.
Native extension SHA256 is
``f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226``.

Validation quality
------------------

.. csv-table::
   :header: "Method", "Seed0", "Seed1", "Seed2", "Mean", "Sample SD"

   "AdamW", "2.223922", "2.232003", "2.228214", "2.228046", "0.004044"
   "8-bit AdamW", "2.183684", "2.182602", "2.183787", "2.183357", "0.000656"
   "GaLore scale1", "2.176957", "2.177600", "2.177371", "2.177309", "0.000326"

The relative mean change is -2.2772% against full AdamW and -0.2770% against
8-bit. The issue's quality limit is <=+2% versus full AdamW. All three GaLore
validation endpoints are below their paired 8-bit endpoints. This modest result
is specific to this model, split and budget. Validation was used for development
selection; three seeds do not establish general or statistical superiority.
The fixed test split has historical results from previous configurations. The
frozen final evaluation below is descriptive; it is not an untouched held-out
discovery and was not used to revise this selection.

Frozen final test evaluation
----------------------------

Nine existing model-only endpoints were evaluated using the final runtime
``65b3dd64133770266a5a3f1f05f5bef64db359ead45810f0ae4a557719e0710c``
and the same native binary and frozen encoder/scorer. All324 model/config files
were freshly hashed against their training inventories before evaluation.
Every endpoint scores743 examples and55,562 response targets with the same
encoded-example fingerprint. No retraining, checkpoint selection or setting
changes followed test feedback. Sampled GPU occupancy contained no foreign PID.

.. csv-table::
   :header: "Method", "Seed0", "Seed1", "Seed2", "Mean", "Sample SD"

   "AdamW", "2.210181", "2.209285", "2.208179", "2.209215", "0.001003"
   "8-bit AdamW", "2.160923", "2.159379", "2.160790", "2.160364", "0.000856"
   "GaLore scale1", "2.156023", "2.152665", "2.155768", "2.154819", "0.001869"

GaLore's mean is2.4622% lower than AdamW and0.2567% lower than8-bit.
All three paired seed differences are negative. This remains one historically
used data split and a modest quality difference, without a broad superiority
or statistical-significance claim.

Same-card performance
---------------------

These are fresh-model processes on physical GPU0, UUID
``GPU-550e9997-ba0a-b916-63f8-ea9284318c37``. Each path has2 warm-up and400
measured updates. GaLore refreshes at200/400. No checkpoint or final validation/
test I/O is included. This is one timing replicate, not a confidence interval.

.. csv-table::
   :header: "Method", "Allocated peak GiB", "Reserved peak GiB", "Score+train ms/update", "Response targets/s"

   "AdamW", "12.083508", "13.140625", "590.078", "543.98"
   "8-bit AdamW", "8.453028", "9.667969", "554.504", "578.88"
   "GaLore scale1", "8.271771", "9.367188", "622.147", "515.94"

GaLore allocated peak is2.1443% lower than8-bit, while wall step time is12.1989%
higher and throughput10.8726% lower. Its398 ordinary updates average610.662ms;
the two refresh updates average2907.776ms. Thus refresh overhead alone does not
explain the remaining speed gap. Memory savings do not imply a throughput win. Normal allocated/reserved peaks
are8.271771/9.367188GiB; refresh allocated/reserved peaks are
8.143436/8.955078GiB. In particular, refresh time and peak allocation are separate
observations and the refresh peak need not be the largest allocator peak.

The GaLore entry reuses the baseline measurements by exact SHA256 from the
``performance-model-owned-s2`` directory's AdamW/8-bit controls. Those controls
did not use model-owned GaLore. The entries are different: their shared common
module is byte-identical, and training/scoring/timing AST equivalence is recorded
after five explicit guard/metadata normalizations. Original entries and the AST
proof are included. Score+train wall time includes actor scoring and the public
training call; optimizer CUDA events are read at existing profile synchronization,
without a newly inserted optimizer sync. NVML one-second samples are lower bounds.

Audit qualifications
--------------------

Full-state and independent model-only validation agree exactly at all nine
endpoints. The audit rehashed324 model/config files and all three retained
GaLore full optimizer states. The latter are kept externally, not embedded in
this small raw-log archive. Optimizer-retention inventories and model indices
identify those external artifacts.

One original GaLore seed0 full-state evaluation was flagged with a foreign GPU
PID at exit. Its training and model-only evaluation were clean. The original
flag is retained. A complete full-state evaluation on idle GPU0 returned the
same validation dictionary; it is separately archived. Its initial idle check
is not a claim of continuous NVML monitoring. The flagged evaluation contributes
no timing measurement to the performance table.

For339 stored FP32 bases, one CPU FP32 Gram accumulation diagnostic read
1.19209e-6. Recomputing dot products in FP64 on the same unchanged stored FP32
matrix gave3.66462e-7; the maximum across all339 was4.56939e-7 against the same
1e-6 geometry threshold. The original FP32 observation remains recorded. This
is a geometry diagnostic, not a relaxation of FP32 update-reference tolerances;
the original FP32 Gram calculation did not pass339/339.

A separate model-owned native-precision GaLore experiment reduces memory further
(7.0785GiB) but intentionally drops BF16 hidden FP32 low bits. It is not the
selected lossless-master candidate. Its single-seed4096-update validation is
2.180777 versus8-bit2.165304, and its timing still trails8-bit. These figures
cannot be attributed to the strict-master scale1 candidate. Negative structural
experiments remain available locally and are not promoted into the primary path.

Final-source correctness and delivery checks
--------------------------------------------

The delivered runtime SHA256 is
``65b3dd64133770266a5a3f1f05f5bef64db359ead45810f0ae4a557719e0710c``.
It keeps the original strict-master implementation, excludes the private native
master experiment, and adds early rejection of an explicitly incompatible
checkpoint backend policy even if relabeled as a legacy schema. A legitimate
legacy checkpoint without such a field remains supported. This rejects invalid
input without changing valid updates or the accepted checkpoint precision rule.

The final and measured runtimes were compared on two CPU and four CUDA cases,
including mixed BF16/FP32 owners, rank128 and multi-tile reconstruction. Across
1,152 tensor comparisons, updates and canonical states were bit-identical;
serialized checkpoints cross-loaded in both directions before/after refresh.
Malformed late-owner states were rejected without partial publication. This is
a concrete finite-case equivalence check, not a universal formal proof.

An independent same-rule FP32 reference passes six CPU and six CUDA cases at
the original atol1e-6/rtol1e-4, with refreshes0/3/6/9 and exact continuous/resumed
states after saving immediately before refresh. Maximum weight absolute error
is2.98e-8 on CPU and5.96e-8 on CUDA. The reference uses separate QR/Ritz code,
PyTorch AdamW and explicit dense covariance transport, not production update or
projector helpers. The small FP32 cases exercise ordinary FP32 masters; separate
BF16 CUDA integration checks cover compact storage and fused writeback.

For1024-by-1024/rank128, final-source measured optimizer tensors including the
FP32 master, two low-rank moments, basis and private CPU RNG total5,772,224B,
versus8,388,612B for measured PyTorch FP32 AdamW optimizer tensors. Moments plus
basis alone occupy1,572,864B; private RNG adds5,056B. Model weights, gradients,
activations and decomposition transients are excluded from this state check.

The public Trainer separately ran200 prefix updates on the actual fixed Dolly
training fold, context512/batch4/microbatch2 and the selected rank128/gap200/scale1
configuration. It saved, closed and created a new Trainer from that checkpoint.
Before scoring/backward, all906 canonical tensors (3,040,529,296B), optimizer
metadata,226 parameter owner names, RNG, schedule and worker clock matched the
saved sidecar exactly. Endpoint score probes and the next batch were identical.
Resumed updates200/201 correctly refreshed then used the ordinary path, with
one optimizer update/public call and native logp_abs_diff_mean0 throughout.
These202 engineering updates do not extend any formal1024 quality budget.
Immediate restore equality is not a promise of bitwise identical future native
BF16 CUDA backward trajectories. An initial observer-installation failure and
its logs are retained; only the successful independently rerun attempt is counted.

The full-model gradient/decomposition inventory remains the separately labeled
historical scale2 engineering diagnostic in :doc:`galore-sft-qwen3`:2,384,199,680B
of FP32 gradients through global clipping; returned outputs and allocator-visible
transients are distinguished from exclusive solver workspace. It is a
context64/batch2 probe, not the Dolly peak or a final-scale1 rerun. Its model,
projector, storage and gradient-consumer code is unchanged; the final strict
runtime differs from that diagnostic runtime only by the checkpoint rejection
guard. Primary selected steady/refresh allocator peaks are reported above.

Local final-source ``pytest tests/ -k cpu`` passed1,815 tests and12 subtests,
with28 skips and2,222 deselected. Related CUDA tests passed59 cases. Independent
offline replay passed18 overlapping tests and reproduced nine data/ordering
files byte for byte. The checkpoint-policy regression failed eight malformed
cases before the fix and passes all10 cases after it. These are local Python3.12
Linux/CUDA results, not every remote CI matrix entry or every GPU test. Pinned Ruff0.15.17 lint and formatting pass over all450 Python files. Changed-file
EOF, whitespace, private-key and1000KB size checks pass. The17 archive partitions
verify all640 members; an independently corrupted copy is rejected. Final
source/commit and exact commands are bound by the readiness record after packaging.

Portable evidence
-----------------

``galore-frontier-evidence/manifest.json`` binds independently verified archive
partitions, each below950,000B. Each archive has an internal SHA256/size member
manifest. From any working directory, run::

    python /path/to/repo/docs/reports/galore-frontier-evidence/verify.py /path/to/repo/docs/reports/galore-frontier-evidence

The bundle contains original complete trajectories, configs, commands, measured
source and both successful and qualified/failed audit records. Final numerical,
semantic, Trainer restore, descriptive test and repository-check records retain
their distinct provenance. Original absolute paths inside raw records remain
unchanged; member names and verification are portable. Large tensor artifacts,
model assets, environments and native binaries are excluded and remain external.

Use ``scripts/galore_sft/config.scale1-all.example.json`` and its README for
portable reproduction. The historical scale2 examples and report remain clearly
labeled. Delivery targets ``oss-camp2026``. The official
`issue593 <https://github.com/inclusionAI/AReno/issues/593>`_ requires baseline
and low-bit comparisons, not simultaneous quality, whole-model memory and speed
wins over8-bit. This result meets the measured validation limit, while its
reported speed deficit remains a limitation.
