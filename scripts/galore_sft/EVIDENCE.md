# Shareable raw-evidence layout

Keep all paths relative to the repository or a separately archived artifact root. Each run gets a fresh directory; never reuse an endpoint as an output path.

```text
artifacts/
  raw/dolly/                      # pinned JSONL + original card; obtained separately
  data/dolly/                     # nine checked deterministic files + summary/source/license
  checkpoints/seed0/galore/final/ # weights + optimizer/RNG sidecar; obtained separately
  galore-sft/native/seed0/galore/
    command.sh                    # actual executable command and environment
    input-manifest.json           # resolved Git/model/data/source identity
    gpu-occupancy.jsonl            # timestamp/PID ownership before/during run
    train/
      metadata.json               # wrapper/source/binary/data hashes, versions, flags
      encoding.json               # mask/truncation/token counts
      steps.jsonl                 # unmodified per-update raw metrics and batch hashes
      curve.jsonl                 # fixed validation monitor, no selection
      summary.json
      train.log
      launcher.log
    eval/
      metadata.json
      summary.json                # full validation/test counts and token-weighted losses
      eval.log
      launcher.log
```

Repeat seed0/1/2 × adamw/adam8/galore with the same contract. A separately named arithmetic/runtime intervention must have its own directory and matched baselines. Label interrupted attempts, foreign-PID contamination and exclusion reasons; retain the original attempt logs rather than editing them into an accepted run. Report quality across the registered seeds, and distinguish allocated/reserved framework peaks from process/NVML measurements.

The small reproduction bundle includes code, pinned data hashes, original mathematical AST hashes, CPU test output, an offline byte-replay verification, and manifest templates. It omits raw Dolly examples, model weights, large optimizer sidecars and full formal GPU logs. Those can be distributed as a separately hashed artifact archive with the dataset attribution. Do not claim those omitted artifacts are present in this bundle.

`run-manifest.example.json` is explicitly a template, not a result. Filled manifests must point to actual unedited files. Existing historical GPU results belong to the frozen source/wrapper hashes listed in `frozen-ast-manifest.json`; the portable entrypoints have verified CPU equivalence only. Preserve source-wrapper lineage when including historical raw results alongside a later reproduction.
