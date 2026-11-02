# Strict deferred evidence

This package preserves 748 unchanged historical logical files (119,062,386 raw bytes) from the strict1.7B performance,1024/4096 quality, context1024 and native-resume studies, plus the exact measured221-file runtime. Binary model/checkpoint/data assets are external; their manifests remain in the records. Logs that duplicate JSONL and binary event streams are omitted. No original study was edited to build this package.

Fifteen tar.xz archives store lossless chunks; `manifest.json` lists every original logical path, complete-file length/SHA256, ordered chunks, archive membership and compressed identities. No extraction is necessary. Archives are below900,000bytes; the manifest is below the repository1000KB limit. This is packaging, not new training evidence.

From the repository root:

```bash
python docs/reports/galore-deferred-evidence/verify.py docs/reports/galore-deferred-evidence
python docs/reports/galore-deferred-evidence/summarize.py docs/reports/galore-deferred-evidence --repo .
```

The second command freshly checks49,698 raw training rows, all paired batch identities/response-target counts, native update/clock/finite-loss guards, deferred lifecycle barriers/release, real refresh indices and source identity, then recomputes both three-seed quality tables and three-card performance gates. It checks validation totals/count arithmetic; original validation scalar streams were not retained, so their digests cannot replay the missing model outputs. The old primary performance gate remains failed onGPU2, and all4096 candidate seeds remain worse than8-bit.

The final optional fused runtime differs from historical80a in exactly two files. `final-checks/FINAL_SOURCE_BRIDGE.json` pins those changes and final337c7612, frozen5d ABBA timing and normal-float executable-PTX identity. The verifier checks this explicit bridge instead of claiming final-source byte identity. Five `writeback-*.tar.xz` supplementary archives preserve236 additional unchanged files:1608 raw ABBA updates, per-arm monitors, frozen protocols, numerical checks, the integer typing failure and repair, full captured PTX and independent reviews. They are bound member-by-member by `FINAL_CHECKS_MANIFEST.json`. No extra final-source native timing/quality run is implied.

The `resume/` records describe exact load plus a separate178-update engineering continuation; they are not a4096 serialized endpoint or a quality budget extension. The short1024 quality study includes checkpoint/reload; the4096 one is live final validation only. Study source/harness manifests and absolute forensic paths remain unchanged. For new portable commands see `scripts/galore_deferred/README.md`; those wrappers were not used for the historical measurements.

`final-checks/` contains this round's source identity, runtime acceptance, commands and original success/failure logs; `FINAL_CHECKS_MANIFEST.json` binds them separately from the historical archives. Logs are losslessly compressed as `.log.xz`, preserving original diagnostic whitespace without treating it as source formatting. Original frozen harness `.py` files are stored as `.py.txt` without changing their bytes; the manifest maps their original names and records compressed plus original identities. They are forensic originals, while runnable reference tests live in `tests/`. Compare the exact commands and scopes rather than adding overlapping reference cases as independent replications.

To rebuild historical archives, `pack_evidence.py --project-root PATH --output NEW_DIRECTORY` expects the original study layout under that project root. It refuses an existing output directory. Original source files are tensor-free; full model/checkpoint payloads are deliberately excluded.
