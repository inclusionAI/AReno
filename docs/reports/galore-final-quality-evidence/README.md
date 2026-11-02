# Final-source three-seed SFT evidence

This package retains the nine final-runtime jobs (native full AdamW, native 8-bit AdamW, strict deferred + fused GaLore; seeds 0/1/2; 4096 updates each). It contains lossless raw per-step records, per-response-target validation logprobs, metadata, input digests, queue/resource records, frozen protocols and the original observation/audit sources. Weights, checkpoints, native binaries, dataset payloads and credentials are excluded.

From the repository root:

```bash
python -I -S docs/reports/galore-final-quality-evidence/verify.py \
  docs/reports/galore-final-quality-evidence --repo .
python -m pytest tests/test_galore_final_quality_evidence_cpu.py -q
```

`verify.py` uses only the standard library and the existing sibling capacity container reader. Its explicit schema differs from the capacity study; only the archive format is shared. It checks archive/chunk/file identities, all 36,864 update rows, exact paired training identities, response counts and identities, actual native consistency/lifecycle/refresh guards, source/model/data/wrapper identities and each NLL from 52,181 response logprobs. It recomputes the three-seed means, full-AdamW+2% gate and separate 8-bit comparison. `--repo` also checks all 221 runtime Python files and the public runner/helper/manifests against the measured sources. No archived harness is imported or executed. A temporary data tree is removed afterwards.

The trusted, repository-reviewed `audit.py` copies the frozen public-case and quality auditors with one portability adaptation: original host `source_root` is retained as metadata but is not required to equal the temporary data tree. Its scientific checks and numerical tolerances are retained. The original auditors remain separately archived as `.py.txt`. Python `-O` is explicitly rejected so assertions cannot silently disappear. Portable checks verify recorded input identities; they cannot regenerate omitted token values or rerun a model from logprob scalars.

`stage3/quality_capture.py.txt` records already-returned CPU response logprobs around the sole original final evaluation, returns the original score object and does not add scoring/training calls. Raw capture metadata binds its source hash and the unmodified public runner/helper. This closes the historical missing-scalar limitation for these nine new endpoints; it does not restore missing historical outputs.

The primary scientific gate is candidate loss <= full AdamW * 1.02 for each paired seed and for the ratio of three-seed means. Execution/audit success is distinct from that gate. Comparisons against 8-bit remain separate; the candidate is slightly worse on all three seeds. Training allocator peaks and complete score+train times are observations from the quality jobs, not repeated ABBA timing qualification or packed capacity measurements. No endpoint checkpoint was saved.

To rebuild deterministically from the original experiment root:

```bash
python docs/reports/galore-final-quality-evidence/pack_evidence.py \
  --root /path/to/original-final-validation-round --output /path/to/new-empty-package
```

The allowlist deliberately packages only this final quality study and its provenance. Earlier public-entry performance, save/resume repairs and profiling are separate studies; no new claim about those studies is inferred from this package. The top-level plan includes their context, while the quality decision freezes the unchanged recipe. Original absolute paths in archived forensic records are informational and are never used to locate private data by the portable verifier.
