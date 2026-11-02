# Full-length packed-text capacity evidence

Run from the repository root with Python 3.10 or newer:

```bash
python docs/reports/galore-capacity-evidence/verify.py --repo . --json-output capacity-verification.json
```

Only the Python standard library is required. No GPU, PyTorch, native extension,
model weights, network access, installation or archive extraction is needed.
Omit `--repo` when auditing a standalone copy; that skips comparison with the
current checkout. An optional positional argument selects another evidence
directory. Archived harness sources are inert `.py.txt` forensic records and
are never imported or executed.

The verifier checks lossless archive/chunk/file identities and independently
recomputes the schedule, raw update guards, timing means, allocator peaks,
three-card comparisons and capacity gates. The result is 69 valid cases:
47 complete and 22 typed CUDA OOM, with 4,128 completed update rows and 26,448
resource-monitor samples. Exactly 49 valid cases are reused unchanged from the
first round. Its two interrupted confirmations, containing 370 partial updates,
are retained separately and excluded from successful-case claims; their
replacements start fresh for all 203 updates.

The candidate's confirmed grid capacities are batch 16 at context 1,024 and
context 4,096 at batch 4. Full AdamW confirms batch 8 and context 2,048. At both
candidate cells, all three cards complete 203 candidate updates, including the
real projection refresh at step 200; full AdamW encounters an actual typed CUDA
allocation failure on all three cards. Both full-AdamW capacity gates pass at a
2x confirmed-grid ratio. Native 8-bit AdamW also completes these cells, so the
stronger 8-bit capacity gate **does not pass**. Timings and memory comparisons
with 8-bit are recomputed per card in the JSON output. Whole score-plus-train
allocator peaks are distinct from native training-microbatch peaks and NVML
device observations.

This is a finite-grid capacity workload on train-only packed text, not a
model-quality evaluation, exact global capacity maximum, or attribution of OOM
to a particular kernel. The measured runtime hash is
`337c7612282e8fc5254d890ac7659c88c2f38a8e8430d5b3cdc84f6dc96b28b8`.
`--repo` verifies the current 221 Python runtime files against that hash without
importing the package.

**Offline limitation:** token payloads are deliberately omitted. The verifier
can check the deterministic sampler, token counts, full sequence lengths and
cross-case batch digests, but cannot regenerate token values from a digest.
The original independent full replay proofs and pinned train/data/model
manifests remain available as unchanged forensic records. Recreating token
payloads requires the separately licensed, pinned source assets and the saved
data recipe. No validation/test payload, raw training text, weights, native
binaries, event binaries, caches or credentials are included.

`manifest.json` describes 73 deterministic logical tar units: two protocol
bundles and 71 case units, including the two excluded interrupted cases. Each
unit contains an internal per-file hash
inventory. The logical bytes are split into 512 KiB chunks and stored in 17
bounded `part-*.tar.xz` files (about 2.05 MB total; largest 200,360 bytes).
Individual archive parts are below 900,000 bytes; the manifest stays compact
because thousands of worker-score files are inventoried inside their case
units. The reader checks every layer in memory and never writes archive
members to disk. Absolute paths occur only in unchanged historical records;
verification routes them through portable round labels.

To reproduce packaging from complete, immutable source rounds:

```bash
python docs/reports/galore-capacity-evidence/pack_evidence.py \
  --screen-root ORIGINAL_ROUND --repair-root REPAIR_ROUND \
  --output EMPTY_OUTPUT_DIRECTORY
```

The packer refuses an output directory that already contains archive parts or
a manifest. It allowlists text evidence, preserves each source file's exact
bytes and uses deterministic archive metadata. Source protocols, preparation
and freeze records, final analyses/reviews, raw case logs/profiles, train-data
licensing/recipes and independent replay proofs are included.
