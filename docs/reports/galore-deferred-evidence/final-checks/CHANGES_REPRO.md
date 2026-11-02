# Portable reproduction runner supplement

Implemented only `delivery/repo/scripts/galore_deferred/{run.py,README.md,model-manifest.json,input-prefix.json}` and `delivery/repo/tests/test_galore_deferred_repro_cpu.py`. No shared runtime/helper/public API was changed and no training was launched.

The runner supports fresh-process AdamW/8-bit/strict/deferred controls, fixed seed0/1/2, two warmup plus400 measured performance updates, or1024/4096 quality updates without extra warmup and one final validation pass. It saves no checkpoint and does not open/hash test data. It records real source/native/model/used-data identities and raw inputs/native guards. Pinned model file hashes and fixed data hashes are checked before training; output directory reuse is rejected.

The existing `galore_sft/common.py` functions `encode`, `batch_indices`, `evaluate` and `trainer` are reused directly. This session also compared their ASTs with the formal strict80a study helper: all four are equal. The new main loop and lightweight worker observer remain distinct from the original formal measurement harness, so the README and metadata explicitly prevent provenance confusion.

Executed with CUDA hidden: thirteen CPU mock contract tests passed, CLI help smoke passed, and pinned Ruff0.15.17 lint/format checks passed. Tests cover exact step budgets, warmup exclusion/refresh coverage, final-only response NLL, original batch-index fixtures, formal batch hash serialization, private-flag rejection and native guard failure cleanup. These are CPU contracts, not GPU or learning/performance qualification. `REPRO_QUALIFICATION.json` records commands, exact file hashes and AST comparison results.

CPU verification: source the final-round environment, then `CUDA_VISIBLE_DEVICES='' taskset -c 60-63 python -m pytest delivery/repo/tests/test_galore_deferred_repro_cpu.py -q -p no:cacheprovider`.
