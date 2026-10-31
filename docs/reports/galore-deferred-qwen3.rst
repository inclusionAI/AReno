GaLore with deferred projected-gradient lifetime on Qwen3-1.7B
============================================================

This opt-in, single-GPU recipe reduces observed training allocated peak by
30.3% relative to native 8-bit AdamW on the matched Dolly workload, at an observed
2.4--3.5% complete score-plus-training time cost. Its 4096-update, three-seed
mean validation NLL is 0.46% higher than 8-bit AdamW and 7.20% lower than full
AdamW. It is a memory/quality/time tradeoff, not three-metric dominance.

The historical three-card and quality results retain the original frozen runtime identity
``80a70ff0f855fd5ec742ee985a507bc8adcbfd0ee3dafc23cc51da9708981c79``.
The initial strict integration (commit ``a63289e``) preserves all 221 Python
files byte for byte. The final optional fusion adds changes in exactly two
runtime files; ``FINAL_SOURCE_BRIDGE.json`` records their identities and the
separate numerical/performance qualification. Historical endpoints remain
attributed to their original sources, not relabeled as new final-source runs.
The original 0.6B report remains historical and describes a different workload.

Method and supported scope
--------------------------

Every model parameter remains trainable. Eligible actual two-dimensional
parameter matrices keep low-rank FP32 moments and projection bases; vectors
and ineligible matrices use full-rank AdamW. Fused QKV and gate/up parameters
are projected as whole parameter matrices, not separate logical blocks.
The measured recipe includes eligible vocabulary matrices.

Logical master weights remain lossless FP32, using the existing BF16 plus
residual-bit storage codec and bounded tiled writeback. This is not the
experimental BF16-authority approximation. Recycled subspace refresh with
final QR and statistical moment transport are explicit algorithm policies.

The private opt-in ``ARENO_EXPERIMENT_DEFERRED_PROJECTION=1`` computes projected
coordinates and norm contributions in post-accumulate leaf hooks and releases
full leaf gradients during backward. Persistent state and model tensors are
not mutated until the whole-backward barrier and global clipping finish.
Projection bases are computed from preclip gradients; this declared policy
is not claimed bitwise equivalent to the ordinary postclip optimizer path.

The deferred implementation rejects multiple microbatches/gradient accumulation,
reentrant backward, multi-GPU, compilation, active multimodal groups, offload
and non-dense-Qwen3 models. It stays disabled by default. The measured
configuration is context512, batch4 and microbatch4, with nonreentrant activation
checkpointing and native eager attention. These restrictions must not be hidden
by advertising a general default optimizer replacement.

Frozen experiment configuration
--------------------------------

* Model: ModelScope ``Qwen/Qwen3-1.7B``, revision
  ``4855588ea1a12789f2e965e5f52a9e4a24c94b2a``; 1,720,574,976 unique parameters,
  226 owners, 113 eligible projected matrices.
* Dolly revision ``40789acc12fc986e9145f87d159f19e2f1d75c6a``; fixed train SHA256
  ``41765eb786b76354f4a7630d5101d46ba1ae08a2eabc35724db462f2d6d93447``.
  Fixed 13,544 training and 708 validation examples. Validation has 52,181
  response targets at context512. The test split was not read in these studies.
* Seeds 0, 1, 2 for quality; seed0 for timing. Rank128, refresh gap200, scale1,
  recycled projector oversampling16/one iteration/final QR, moment transport,
  strict compact masters, tile_rows16384, fused writeback, gradient release.
* Learning rate and minimum rate 1e-5, betas (0.9, 0.95), weight decay0.01,
  global clipping1.0. The same original model and absolute-cursor batches are
  used for each paired method.
* Native extension SHA256
  ``f2d49f7b8ffd2fccbfce0f6eeb3108088be7834627b88c9ad71a0eb486f4d226``.
  Model manifests, software versions, hardware identities, commands and raw
  records are included in the tensor-free evidence package.

Complete-flow performance
-------------------------

Each arm is a fresh process with two warmup updates and 400 measured public
score-plus-training updates. The measured window includes refreshes200 and400;
cold refresh0 and warm step1 are retained separately. Each physical card rotates
the three method orders. Cards are timing repeats, not quality seeds.

.. list-table:: Context512 matched complete-flow results
   :header-rows: 1

   * - GPU
     - 8-bit AdamW ms
     - Original GaLore ms
     - Deferred GaLore ms
     - Deferred / 8-bit time
   * - 0
     - 504.040
     - 548.993
     - 518.944
     - +2.9570%
   * - 1
     - 502.129
     - 559.336
     - 514.130
     - +2.3901%
   * - 2
     - 500.556
     - 561.493
     - 517.936
     - +3.4722%

All three cards have the same observed allocated/reserved peaks: 8-bit AdamW
16.285506/17.306641 GiB, original GaLore15.788415/16.603516 GiB, deferred
GaLore11.347551/14.462891 GiB. Deferred allocated peak decreases30.3212%
versus 8-bit and28.1274% versus original GaLore. Full wall decreases5.47--8.08%
versus original GaLore. Complete raw ordinary/refresh/cold peaks and times
remain in the archived performance analysis and records.

The preregistered primary research criterion required every card to save at
least10% allocated with at most3% wall overhead versus 8-bit. It **fails** on
GPU2. The separately preregistered20% memory /10% overhead tradeoff passes all
cards. These thresholds are research decisions, not official issue criteria.
An optimizer-only CUDA event excludes projection moved into backward and
cannot establish complete-flow acceleration.

At context1024, a separate fixed-order single-card study observes16.413743 to
11.939960 GiB allocated (-27.2563%) and526.648 to527.946 ms (+0.2464%). It has
two warmup plus400 measured updates and real refreshes. This is a second
workload check, not robust speed superiority or uniform1024-length training.

Quality across both registered budgets
--------------------------------------

.. list-table:: Final response-token-weighted validation NLL (lower is better)
   :header-rows: 1

   * - Budget
     - Full AdamW mean
     - 8-bit AdamW mean
     - Strict deferred mean
     - Deferred / 8-bit
   * - 1024 updates
     - 1.936826
     - 1.917774
     - 1.904438
     - -0.6954%
   * - 4096 updates
     - 2.056386
     - 1.899689
     - 1.908353
     - +0.4561%

Every budget includes three seeds for each of the three methods, no extra
training warmup, matched batch indices/hashes/target counts, and one final full
validation endpoint. At4096, candidate NLL values are1.911059/1.907678/1.906322;
8-bit values are1.900555/1.898690/1.899823. All three seeds are worse than8-bit
at the longer budget. The short-budget advantage did not persist and is not
the headline claim. Every paired seed meets the full-AdamW+2% quality criterion.

The1024 study saved/reloaded endpoints; the4096 study evaluated live endpoints
and did not save fresh checkpoints. Do not describe4096 as a serialized endpoint
or a fresh-reload result. These are development-split measurements, not an
untouched test or a broad statistical significance claim. Original validation
scalar streams were not retained; archived loss totals/counts and recorded
digests cannot independently reconstruct those missing model outputs.

Optional exact scale/writeback fusion
-------------------------------------

The final bounded engineering experiment moves the separately rounded FP32
projected-delta multiplication into compact writeback before the original FMA.
It does not fold scale into learning rate or change moments/projection/clipping.
The private ``ARENO_EXPERIMENT_FUSED_SCALE_WRITEBACK=1`` opt-in stays disabled
by default and requires deferred mode. The portable runner exposes the same
explicit ``--fused-scale-writeback`` opt-in. Ineligible scalar types use the
original multiply/writeback path, selected once during construction.

One assigned A100 ran four fresh engines in strict/candidate/candidate/strict
order, each with two warmup plus400 measured updates, including refresh200/400.
The predeclared gate requires at least1% complete-flow improvement in **both**
opposite-order pairs, with no allocated peak increase.

.. list-table:: Prospective ABBA result, registered Python float scale1.0
   :header-rows: 1

   * - Pair
     - Strict ms
     - Fused ms
     - Complete-flow time change
   * - Strict before candidate
     - 514.357902
     - 503.281983
     - -2.15335%
   * - Candidate before strict
     - 514.755350
     - 508.270552
     - -1.25978%

Both pairs pass. All four arms have identical training allocated/reserved peaks
11.347551/14.462891 GiB. All1608 raw updates,402 paired positions, real refresh
indices and850 assigned-device monitor samples pass independent and parent
review. There is no new8-bit arm or quality endpoint. This one-card ABBA is
evidence of improvement against its contemporaneous strict controls, not an
independent three-card replication or a stable8-bit speed win.

The frozen timing source is ``5d96474a``. Independent review subsequently
reproduced an integer-scale typing bug outside its registered normal-float
domain. The final source ``337c7612`` explicitly converts the PTX operand to
FP32 and selects the original path for unsupported scalar objects/oversized
integers. The original failure, repaired qualification and initial debug-inclusive
PTX mismatch are retained. Six normal-float target specializations have identical
executable PTX/declarations after removing only source debug records/comments
and whitespace; one signature matches the aligned production tile size.
The additional eligibility check runs only in the constructor. This is an
explicit source/evidence bridge, **not a rerun** of native timing or quality
on the repaired source. Arbitrary NaN payload equivalence is not claimed.

The repaired source passes same-input normal and scalar-extreme full-optimizer
checks, original-tolerance independent CUDA reference, cross-flag serialized
continuation, finite edge/invalid-flag checks, fallback/error equivalence and
default-off snapshots. CPU tests exercise the original fallback; actual fused
arithmetic is checked on CUDA. The original96-case edge helper incorrectly
labels576 comparisons; its five assertions per case mean480, transparently
corrected without modifying the frozen helper/results.

Correctness and public restore
------------------------------

Independent same-policy FP32 reference checks use the original
``atol=1e-6, rtol=1e-4``. They include recycled projection, refresh, transport
and exact serialized optimizer continuation. The default-off path is separately
compared with the immutable delivery floor. Current final-check commands/results
are recorded alongside source identity in the evidence directory; historical
test results are not relabeled as freshly executed checks.

The actual public Trainer restore study loaded the retained1024-step strict
checkpoint into a new engine and completed178 updates at absolute indices
1024--1201, crossing production refresh1200. Immediately after load and before
the first score/backward, all906 canonical optimizer/RNG tensors and226 model
owners/310 native split weights matched the saved state exactly. Original
43 checkpoint files (11,320,251,530 bytes) were rehashed unchanged. This separate
engineering continuation is not added to the quality budget. Native BF16
backward can be nondeterministic; saved-state equality does not promise a
bitwise identical future native trajectory.

Evidence and reproduction
--------------------------

``galore-deferred-evidence/`` contains unchanged raw per-step JSONL, summaries,
encoding/inventory records, frozen protocols, controllers, independent audits
and the exact measured Python runtime. Checkpoint/model binaries and full data
are external. Original forensic harness paths remain unchanged; they are not
portable launch commands. See ``scripts/galore_deferred/README.md`` for a new
portable entrypoint, explicitly distinguished from the original measured one.

Verify every archive and logical source/log file without CUDA or model assets::

   python docs/reports/galore-deferred-evidence/verify.py \
       docs/reports/galore-deferred-evidence

The evidence directory additionally contains supplementary archives for the
registered ABBA, repaired-source qualification and full target PTX. The verifier
checks every supplementary member as well as the unchanged historical package.
``summarize.py --repo .`` verifies either the original runtime identity or the
explicit two-file final-source bridge; it does not claim byte identity for the
changed source.

Publication scope
------------------

`Issue593 <https://github.com/inclusionAI/AReno/issues/593>`_ also has
`PR640 <https://github.com/inclusionAI/AReno/pull/640>`_, inspected at
``539f6c4fa08ae3fcc6a4f5ebd712dda2841c34c4``. Its logical-block geometry,
projector and checkpoint interfaces differ. Its reported Alpaca/GSM8K workload
and hardware are not directly comparable to these Dolly numbers. The shared
lossless master codec is existing infrastructure, not a new contribution here.
The separable systems contribution is deferred projected-gradient lifetime and
bounded strict writeback. A port preserving640's geometry/API needs its own
qualification. The issue requests agreement on overlapping scopes before
separate PRs. This local package records evidence; it does not imply maintainer
approval, remote CI success, a resolved API choice or an award outcome.
