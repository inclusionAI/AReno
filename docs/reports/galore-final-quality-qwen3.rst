Final-source GaLore quality on Qwen3-1.7B
=======================================

The published final runtime directly completes nine fresh response-only SFT
jobs: full AdamW, native 8-bit AdamW and strict deferred GaLore with rounded
scale/writeback fusion, each at seeds 0, 1 and 2 for 4096 updates. Candidate
validation NLL is 7.18% below full AdamW and 0.4603% above 8-bit on average.
All three paired seeds pass the predefined full-AdamW+2% quality gate; all
three are slightly worse than 8-bit. This is a memory/quality tradeoff, not
three-metric dominance.

Unlike the historical endpoints in :doc:`galore-deferred-qwen3`, these jobs
use the final repaired runtime directly and retain every response-target
validation logprob. The unchanged public reproduction runner performs the
training and original final evaluation; a private observation adapter records
already-returned CPU scalars without adding a score/evaluation call.

Fixed protocol and source
-------------------------

* Runtime SHA256
  ``337c7612282e8fc5254d890ac7659c88c2f38a8e8430d5b3cdc84f6dc96b28b8``;
  native extension SHA256
  ``f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226``.
  The measured checkout is commit ``7c4806bfd1c7632598de7c3946bd3a51f572ae0b``.
  This supplement changes no runtime source, dependency or native binary.
* ModelScope Qwen3-1.7B revision
  ``4855588ea1a12789f2e965e5f52a9e4a24c94b2a``; fixed Dolly train/validation
  folds, 13,544 encoded training records, context limit 512, batch 4 and one
  microbatch of 4. The original helper preserves response targets; this is
  not the exact-length packed workload in :doc:`galore-capacity-qwen3`.
* LR = minimum LR = 1e-5, betas (0.9, 0.95), weight decay 0.01, global clip 1.
  GaLore rank 128, refresh gap 200, scale 1.0, recycled projector with
  oversampling 16, one iteration and final QR, moment transport, compact
  lossless logical FP32 masters and tile 16384. Deferred projection and
  fused scale/writeback are enabled only for the candidate; model-owned
  BF16 masters are disabled. No recipe change follows the memory diagnosis.
* Three A100-SXM4-40GB cards, PyTorch 2.6.0+cu124. Seed 0 uses card 1 in
  AdamW/8-bit/GaLore order; seed 1 uses card 2 in 8-bit/GaLore/AdamW order;
  seed 2 uses card 3 in GaLore/AdamW/8-bit order. Each job is a fresh,
  independent single-GPU process. Method inputs match exactly within seed.
* Exactly 4096 optimizer updates, no extra warmup, one final evaluation over
  708 examples and 52,181 response targets. No interim validation, early
  stopping, endpoint checkpoint, test read or selection within this round.
  These are validation results, not untouched-test significance claims.

Quality results
---------------

.. list-table:: Final response NLL; lower is better
   :header-rows: 1

   * - Seed
     - Full AdamW
     - Native 8-bit AdamW
     - Strict deferred + fused
   * - 0
     - 2.055977397487
     - 1.900481950213
     - 1.911266654379
   * - 1
     - 2.058256235702
     - 1.898675096878
     - 1.907688123708
   * - 2
     - 2.054077299272
     - 1.900028704729
     - 1.906465148933
   * - Mean
     - 2.056103644154
     - 1.899728583940
     - 1.908473309006
   * - Sample standard deviation
     - 0.002092326716
     - 0.000940071332
     - 0.002495195504

The primary gate requires each paired candidate NLL and the candidate mean
to be at most 1.02 times its full-AdamW control. Every check passes: paired
relative changes are -7.0385%, -7.3153% and -7.1863%; the ratio of three-seed
means gives -7.1801%. Against 8-bit, paired increases are 0.5675%, 0.4747%
and 0.3388%; the ratio of means gives +0.4603%. The latter comparison is
reported separately and does not substitute for the primary baseline.

The historical strict-runtime mean was 1.908353 versus 8-bit 1.899689.
The new final-runtime result is consistent with the same qualitative tradeoff.
It establishes direct final-source learning evidence, not a causal precision
or fusion improvement over the older implementation.

Memory and time during the quality jobs
--------------------------------------

.. list-table:: Same-card observations over all 4096 updates
   :header-rows: 1

   * - Seed / card
     - 8-bit allocated GiB
     - GaLore allocated GiB
     - Reduction
     - Complete score+train time overhead
   * - 0 / 1
     - 16.364878
     - 11.657529
     - 28.7649%
     - +0.6313%
   * - 1 / 2
     - 16.282166
     - 11.474457
     - 29.5275%
     - +1.7575%
   * - 2 / 3
     - 16.266156
     - 11.403262
     - 29.8958%
     - +1.6634%

These are native training-microbatch allocator peaks and complete public
score-plus-train wall times; cold initialization, encoding and final validation
are excluded from per-update time. All 4096 training updates are included,
including 21 actual candidate refreshes at steps 0, 200, ..., 4000. Reserved
peaks and monitor samples remain separate raw records. This counter scope
differs from the capacity study's full score+train peak instrumentation.

The longer budget encounters different batches and peaks than the historical
400-measured-update experiment; these measurements retain approximately
three-tenths memory savings without claiming an identical peak. The method
order is rotated across cards, but there is only one arm per method/seed/card:
the timing range does not replace repeated ABBA qualification. At larger
packed inputs the previously measured savings remain about 1.9%, and the
confirmed capacity remains equal to 8-bit. No universal 29% saving is claimed.

Reproduction and audit
----------------------

The public entry is ``scripts/galore_deferred/run.py``. For each seed 0, 1 and
2, use separate fresh outputs for ``--method adamw``, ``--method adam8`` and
``--method deferred --fused-scale-writeback`` with ``--mode quality --steps
4096`` and the pinned model/data. See ``scripts/galore_deferred/README.md``
for the full environment, recipe and command. The observation adapter source
is preserved in the evidence; ordinary public runs produce aggregate NLL,
while this round additionally retains the original returned scalar stream.

The :download:`tensor-free evidence package
<galore-final-quality-evidence/README.md>` includes all nine raw step files,
original scalar streams, metadata, protocols, capture sources, queue records
and independent review. No model/checkpoint/data payload is committed. The
portable verifier uses only the Python standard library::

    python -I -S docs/reports/galore-final-quality-evidence/verify.py \
      docs/reports/galore-final-quality-evidence --repo .
    python -m pytest tests/test_galore_final_quality_evidence_cpu.py -q

It checks lossless archive identities, all 36,864 update rows, native logprob,
clock, normalization, once-only owner, barrier, gradient-release and cache
guards; exactly 21 refreshes per candidate; paired training and common
validation identities; all per-response scalar/count arithmetic; and the
saved means/comparisons/gate against independent recomputation. With
``--repo``, current runtime and public runner/helper/manifests must match.
Original host paths are informational after relocation; archived code is
never executed. Input hashes cannot regenerate omitted token payloads, and
this round saved no endpoint checkpoint. Historical save/resume evidence
remains separate.
