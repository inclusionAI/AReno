# P1 common-runtime, three-seed quality evidence

This is a controlled optimizer-recipe comparison, not an untouched end-to-end
run of PR #640. Six fixed endpoints use Qwen3-1.7B, the original Dolly split,
context512, batch4/micro4, rank128/gap200/scale1 and 4096 updates. All three
paired seeds favor deferred647; mean response-token NLL is 1.908965911950 versus
1.934585978801, a 1.3243% reduction. No new 8-bit/full-AdamW quality baseline,
test evaluation, interim validation or checkpoint selection is performed.

From the repository root, with only Python's standard library:

```bash
python -I -S docs/reports/galore-p1-quality-evidence/verify.py \
  docs/reports/galore-p1-quality-evidence
python -m pytest tests/test_galore_p1_quality_evidence_cpu.py -q
```

The 43 bounded, lossless archives contain 50,630 text files, totaling 13,044,420
compressed bytes. Each part is below the repository's 1,000 KiB limit. The
existing capacity container reader verifies every archive, chunk, logical tar
and original-file digest. The P1 verifier then checks all 24,576 update rows,
their independent owning-worker score/train records, native hash observations,
226 unique model/optimizer owners, tied embedding/head identity, original seeded
training schedule, paired input identities, refresh/clipping/deferred barriers,
and six raw final logprob streams. Each stream contains 708 examples and 52,181
response targets. Both the original summation and a separate `math.fsum` reproduce
the NLLs. Altered logprobs are rejected even after recomputing stream hashes.

Three jobs completed before another user began using the GPUs. Their 25,143
original text files were frozen and reused byte-for-byte. The other three
interrupted jobs had no endpoint/checkpoint and restarted from base weights at
update0 after the cards became free. Their original partial update rows and
resource monitors are retained, and never counted as endpoints. Redundant
per-update worker mirrors from those interrupted attempts remain external; all
worker records from the six accepted complete jobs are included. None of the
three queued jobs needed another retry.

The measured common runtime is `230fd975…` (225 Python files), comprising P0 core
`47c75ef8…` plus three private controlled-port files. All 225 exact source files
are archived as inert `.py.txt` data and checked against the original inventory;
they are not installed into this PR's runtime. The original #640 optimizer from
commit `539f6c4…` remains byte-identical (`9234126f…`); the adapter supplies actual
logical linear-row annotations and a read-only refresh metric. The common native
binary hash is `f2d49f7b…`. Offline checks verify the recorded native identity;
the binary itself is excluded. This is separate evidence from the historical
`337c7612…` final-source study already in the PR.

`audit.py` retains the original scientific `audit_case`, `audit_group`, raw-NLL,
protocol and metric checks. Its sole path-related adaptation replaces the old
absolute-path dynamic schedule loader with the identical, statically reviewed
`batch_indices` function. The original protocol, all eight scientific harness
files, helper sources, both sessions, queue selection and SHA inventory remain
separately archived. No archived Python is imported or executed. Historical host
paths are informational and never resolved by the portable verifier. Python
`-O` is rejected. An AST regression checks the static schedule against the frozen
source without executing the archived source.

The current PR runtime is not claimed to equal the separately attributed P0/P1
snapshot, so this verifier deliberately has no misleading `--repo` option. A full
native rerun requires local model/data assets and the measured source/native
environment. The package supports independent offline result verification and
retains the exact execution sources and registered recipe, but does not rebuild
the binary, rerun tokenization/model inference or recover omitted dataset tokens.

Rebuild this package deterministically from the original experiment directories:

```bash
python docs/reports/galore-p1-quality-evidence/pack_evidence.py \
  --root /path/to/quality-queue-round \
  --old-root /path/to/original-quality-round \
  --runtime-repo /path/to/measured-common-runtime \
  --output /path/to/new-empty-package
```

Weights, checkpoints, native binaries, dataset/token payloads, caches and
credentials are excluded. Timing and allocated/reserved peaks retained in these
quality jobs are observations, not repeated performance qualification. Recipe
differences include projection granularity, projector, moment transport and
vocabulary handling as well as gradient lifetime; the observed quality advantage
is not attributed to deferred projection alone.
