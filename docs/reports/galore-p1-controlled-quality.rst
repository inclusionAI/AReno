GaLore common-runtime paired quality
====================================

This supplement compares the complete optimizer recipes of deferred647 and a
controlled transplant of the #640 optimizer on one common runtime. It does not
run untouched PR #640 end-to-end or assume checkpoint compatibility. Historical
final-source AdamW/8-bit quality results remain separately attributed in
:doc:`galore-final-quality-qwen3`.

Protocol and source
-------------------

The fixed configuration is Qwen3-1.7B revision
``4855588ea1a12789f2e965e5f52a9e4a24c94b2a``, BF16 with FP32 normalization
parameters, context limit512, batch4/micro4, rank128, refresh gap200, scale1,
constant learning rate1e-5, betas(0.9,0.95), weight decay0.01 and global clipping1.
Each method uses all three registered seeds,4096 updates and the original Dolly
training schedule. Each seed uses the same physical A100-SXM4-40GB card for both
methods. Paired training indices, token/mask hashes and target counts match.

The native-qualified Python snapshot is ``230fd975…`` at local source commit
``799070f6…``: P0 core ``47c75ef8…`` plus the three controlled-port files. This
commit is provenance metadata, not an assumed public commit link. All225 Python
files are retained and hashed in the evidence archive. The selected native binary
is ``f2d49f7b…``, directly observed in each owning worker after its first update.
This source is not relabeled as the PR's earlier ``337c7612…`` runtime.

The #640 mathematical implementation from ``539f6c4…`` is byte-identical,
SHA256 ``9234126f…``. Its adapter provides linear-row annotations from actual
common-runtime modules and a read-only refresh metric. The methods intentionally
retain their own projection policies: #640 projects112 matrices/196 logical
blocks with Gram/eigh, retains moments at refresh and uses a full vocabulary
fallback; deferred647 projects113 eligible whole matrices with recycled
projection, final QR, moment transport, vocabulary projection and deferred/fused
writeback. Both use strict logical FP32 masters. No causal isolation of the
gradient-lifetime change is claimed by this comparison.

Three-seed endpoint
-------------------

Each job makes one final original development evaluation after update4096:
708 examples,52,181 response targets. All actual returned response logprobs are
retained. No interim validation, test read, endpoint checkpoint, seed selection
or recipe retuning is performed. The candidate projector seed equals its
training seed, preserving the historical strict recipe's seed policy.

.. list-table:: Response-token-weighted validation NLL; lower is better
   :header-rows: 1

   * - Seed / GPU
     - #640 controlled port
     - Deferred647
     - Relative change
   * - 0 / 1
     - 1.937321577681
     - 1.911313006083
     - -1.3425%
   * - 1 / 2
     - 1.933694229617
     - 1.908372911030
     - -1.3095%
   * - 2 / 3
     - 1.932742129107
     - 1.907211818736
     - -1.3209%
   * - Three-seed mean
     - 1.934585978801
     - 1.908965911950
     - -1.3243%

All three paired seeds favor deferred647. This is evidence for these two
registered recipes on one model/dataset; it is not a broad quality superiority
claim, a test-set result, or a fresh comparison with8-bit/full AdamW. In
particular, it does not replace or reverse the earlier result that8-bit had
slightly lower NLL than our candidate in the historical final-source study.

Resource interruption and completion
------------------------------------

The original round completed #640 seeds0/2 and deferred647 seed1. Another user's
CUDA processes then appeared on the cards, interrupting the other three jobs at
2607,3142 and2672 completed updates respectively. No incomplete job contributed
a validation endpoint. The three completed jobs were frozen and reused without
changing any of their25,143 text files. The original failed status, partial
updates and resource monitors remain available.

An independently bounded queue waited for each card to be idle, then restarted
only deferred647 seeds0/2 and #640 seed1 from base weights at update0. Scientific
protocol and all eight harness files stayed byte-identical; all three fresh jobs
completed on their first admitted attempt. Final analysis combines exactly those
three new endpoints with the original three, and requires4096 updates per job,
matching paired inputs, native/owner/refresh/lifecycle checks and raw NLLs.

Training observations
----------------------

These instrumented quality jobs are not repeated timing qualification. Their
complete score+train sums are2031.4--2065.6 seconds for deferred647 and
3143.2--3177.3 seconds for the controlled port, corresponding to34.32--36.07%
less time in same-card seed pairs. Maximum allocated score+train peaks are
11.40--11.66 GiB versus17.91--18.00 GiB, or35.25--36.32% lower. Reserved peaks
are retained separately in the raw summaries and must not be substituted for
allocated peaks. The arms ran at different times, including the overnight queue
gap, so these observations do not establish stable speed dominance. No8-bit
performance measurement is added by this quality supplement.

Offline verification
---------------------

The standard-library verifier recomputes all24,576 updates and all six final raw
NLLs, checks the independent owning-worker score/train records, reproduces the
stored analysis exactly and separately recomputes each NLL using ``math.fsum``.
It checks source snapshots and the unchanged original completed-case inventory.
The existing bounded capacity container reader is reused with a distinct schema;
no archived Python is executed or model/data payload loaded.

.. code-block:: bash

   python -I -S docs/reports/galore-p1-quality-evidence/verify.py \
     docs/reports/galore-p1-quality-evidence
   python -m pytest tests/test_galore_p1_quality_evidence_cpu.py -q

The `package README <https://github.com/ymping666/AReno/blob/feat/galore-593-strict-deferred/docs/reports/galore-p1-quality-evidence/README.md>`_ describes identities,
the sole static-schedule portability adaptation, deterministic repacking,
exclusions and replay limits. Token payloads are omitted, so the offline audit
checks recorded identities rather than regenerating tokens or model outputs.
