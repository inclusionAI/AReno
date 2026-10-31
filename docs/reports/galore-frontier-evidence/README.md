# Selected GaLore evidence

This directory accompanies the selected scale1 report. `manifest.json` binds each original raw file to a compressed partition by relative path, byte length and SHA256. Every `*.tar.xz` is smaller than950000 bytes and contains an internal `MANIFEST.json`. The raw evidence has two explicitly separate source identities: the three-seed quality/performance experiment used runtime7da069e9…, while final acceptance used65b3dd64…. Exact digests, the measured Python source, final freeze record and numerical equivalence audit are included. Measurements are not relabeled as byte-identical final-commit runs.

Verify all partitions without CUDA, the original workspace or extraction:

```bash
python docs/reports/galore-frontier-evidence/verify.py docs/reports/galore-frontier-evidence
```

For verification after copying this directory elsewhere, use `python /path/to/verify.py /path/to/evidence-directory`. The helper imports adjacent `pack_evidence.py`; both files must stay together. Verification checks partition identity and size, all member hashes, global membership and duplicate/missing members. The manifest and helpers themselves are versioned by the repository commit.

Included primary evidence:

- Complete27 train/full-state-validation/model-only-validation output trees from the nine new1024-update comparisons, including raw step trajectories, logs and TensorBoard events; original launch vectors, frozen harness and selection/control records.
- All selected400-step GaLore performance outputs, both matched baseline output trees, original performance entries and explicit AST-equivalence proof.
- Original flagged validation record, separate clean full-state re-evaluation, the FP32/FP64 geometry diagnostic and completed quality audit.
- Measured experimental Python source, pinned model/data/environment identities and tensor-free checkpoint inventories/configs/indices.
- Final runtime semantic review, independent CPU/CUDA reference tests, state-byte accounting and policy rejection regression, including original failed harness observations.
- Final public Trainer prefix200 save/restore, immediate906-tensor equality and updates200(refresh)/201(ordinary); the failed first observer attempt and raw steps remain included.
- Nine frozen descriptive test evaluations, encoded-example fingerprint, per-seed outcomes, input hashes, queue provenance and GPU-process samples.
- Actual local CPU-suite, lint/format and portable offline-reproduction logs, runtime freeze and initial check summary. Final commit, packaging hooks and final local delivery status are recorded separately in `FINAL_PR_READY.json`; this package does not claim the entire publishedCI matrix.

Large optimizer/model tensors, native libraries, caches and all `acceptance/prefix200*` checkpoint directories are excluded. Deliberate extra-directory exclusions are listed in the global manifest. The separate checkpoint-retention record identifies external retained full states and any duplicate cleanup. Original logs/configs retain historical absolute paths as immutable provenance; archive member paths and verification are portable. This is a raw-evidence package, not a self-contained model/environment distribution.

Use `scripts/galore_sft` for portable training/evaluation and data preparation, with explicit model/data/output/checkpoint paths and the selected scale1 configuration. Reproduction requires obtaining the pinned model/dataset and building the native extension for the target CUDA environment. The frozen fixed test split has historical usage; its new evaluation is descriptive, with no post-test selection. Read `ACCEPTANCE_MATRIX.md` and the report for limits, numerical tolerances and memory/timing definitions.
