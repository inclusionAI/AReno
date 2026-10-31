# Independent read-only review of final writeback candidate

Reviewed on 2026-10-31. No candidate/delivery source was changed and no GPU work
was launched for this review.

## Decision

**Do not promote this candidate as a generic numerically equivalent replacement
yet.** The registered normal-range Python-float scale cases have substantial
passing evidence, and the frozen scale=1.0 ABBA remains a valid performance
experiment. There is an uncovered scalar operand type mismatch for other inputs
already accepted by `GaLoreAdamW`.

## Blocking scalar type hole

`_writeback` passes `SCALE` directly into `tl.inline_asm_elementwise`, with
`constraints="=f,f,f"` and `mul.rn.f32`. The optimizer preserves its constructor
`scale` object rather than normalizing it. Its constructor accepts positive
finite Python integers and Python floats outside the normal FP32 range.

Installed Triton 3.2 `language/semantic.py:111` converts Python integer 1 to
`tl.int32`. It converts Python floating values smaller than 2^-126 in magnitude
(including representable FP32 subnormals such as 1e-40), or larger than the
FP32 maximum, to `tl.float64`. `language/core.py:inline_asm_elementwise` converts
each operand with `semantic.to_tensor`, then broadcasts with
`binary_op_type_checking_impl(... arithmetic_check=False)`. The latter explicitly
skips numeric promotion. Thus these operands do not acquire FP32 type from the
FP32 delta or the declared result dtype.

The f-register PTX contract therefore lacks the required numeric FP32 conversion
for valid `scale=1`, `scale=2`, `scale=1e-40`, `scale=1e-300`, and `scale=1e300`
inputs. This is a source-level type-contract defect; no GPU compile failure or
particular wrong output is claimed because this review did not execute it.
CPU PyTorch reference inspection confirms that `mul_` accepts those Python
scalars, producing FP32 integer scaling/subnormal/zero/overflow respectively.

All current numerical cases use normal-range Python float literals
(.5, 1., 2., 4., 1.3, .1), so those passing tests do not close this hole.

For a subsequent independently versioned candidate, either explicitly convert
the inline operand to FP32 with verified PyTorch scalar rounding, or retain the
old multiply/writeback fallback for unsupported scalar types/ranges. Do not edit
the frozen source while its ABBA is running. An eventual corrected source would
need its own final numerical qualification; the old runtime byte-identity claim
would cease to apply to that changed source.

## Other reviewed semantics

- The scale multiplication is separately rounded `mul.rn.f32` without `.ftz`,
  before the original separately rounded decay and original FMA. Scale is not
  folded into learning rate. `enable_fp_fusion=False` is retained. This is the
  correct intended operation order for the tested FP32 operands. I did not
  inspect generated PTX for every specialization.
- Only projected CUDA owners with BF16MasterStorage and fused-writeback enabled
  use the new route. CPU, ordinary FP32 master, nonfused and flag-off paths retain
  the original explicit multiply. Fallback tensor updates remain unscaled.
- Compact storage is initialized only for contiguous BF16 owners, so passing
  noncontiguous model storage to the kernel is excluded by the original
  initialization invariant. Noncontiguous updates are copied contiguously by
  the original helper; both update layouts are explicitly tested.
- Tile offsets and terminal tails retain the same byte-alignment check and carry
  packing code. Low-level tests exercise offset 8 and terminal 10/19/1025/2049
  element counts. Those are concrete cases, not a proof of all index geometries.
- The invalid-flag reduction, finite-result rejection, compact residual encoding
  and late step-level asynchronous assertion are unchanged. Overflow and
  alpha-zero nonfinite cases occur in the low-level test grid. Existing writes
  happen before the final invalid assertion; this path does not promise atomic
  rollback on nonfinite failure, and the candidate does not add such a promise.
- The new flag is default-off and gated by deferred mode. It is an implementation
  choice absent from checkpoint policy. Shared checkpoint policy is defensible
  only while the supported numerical contract remains identical. Cross-flag
  serialized continuation is tested, but the scalar type hole currently limits
  that conclusion to the tested scalar types/ranges.
- Pending cache, hooks, barrier publication, refresh/transport, model/state/RNG
  comparisons and cross-flag resume are covered by the exact full-optimizer
  checks. Those finite toy cases do not establish arbitrary native future
  trajectory equality or arbitrary NaN payload equivalence.

## Evidence freshly verified

Independently rehashed all 238 PREPARED files, 15 assets, and 10 numerical proofs.
The FREEZE prepared digest matches. Candidate runtime is
`5d96474a50da9ce90ddad4e52e14725b76c2fa78a536ca7911216cb7e273e4c7`.
The source comparison against strict80a changes only compact_writeback.py and
deferred_projection.py.

FINAL_CPU_EXACT and FINAL_CUDA_EXACT each report 24 same-input cases and 12
updates per case, including skipped/zero gradients, refresh/transport and
serialized continuation. Default-off comparisons report 24 cases/288 snapshots
for each device against unchanged80a. FINAL_CUDA_EDGES contains 96 cases. Its
576 tensor-comparison label is incorrect: four raw byte comparisons plus one
decoded FP32 comparison means **480 actual assertions**. The separate existing
EDGE_COUNT_CORRECTION.json records this transparently and preserves the frozen
script/result hashes. No assertion was removed or relaxed.

No inference about ABBA success is made here; that experiment is still running
at review time. No quality endpoint is added or borrowed from a different source.

## Proposed follow-up test

`candidate_tests/test_fused_scale_writeback_cuda.py` is a syntax-checked proposal
outside both repositories. It includes integer scales and Python scalar extremes
alongside ordinary fractional scales, and compares original separately scaled
compact bytes/invalid flags against fused writes. It has not been run and is
expected to expose the uncovered operand-type path. Integration requires a
separate authorized correction and successful final numerical/performance gates.
