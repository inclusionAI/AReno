# Releasing AReno

This is the release-hygiene checklist for cutting a tagged AReno release
(for example `v0.0.2`). It exists so that package metadata, git tags, the
GitHub milestone, and the docs do not drift apart.

## Conventions

| Concern | Convention |
| --- | --- |
| Git tag format | `vMAJOR.MINOR.PATCH`, e.g. `v0.0.2`. The leading `v` is required; the publish workflow rejects a tag without it. Pre-release suffixes are allowed (`v1.2.0rc1`). |
| Package version | PEP 440, without the `v`, e.g. `0.0.2`. Setuptools reads it dynamically from `VERSION`. |
| `VERSION` file | The committed source of truth for local package metadata. Keep it in sync with the release being cut; the publish workflow rewrites it from the tag at build time. |
| GitHub milestone | `vMAJOR.MINOR.PATCH`, matching the tag (e.g. milestone `v0.0.2` <-> tag `v0.0.2`). |
| Release artifacts | Source distribution (sdist) only. No wheels are built or published — the CUDA extension is compiled at install time on the user's machine. |
| Release notes | Written from the closed issues in the matching GitHub milestone. There is no committed `CHANGELOG.md`; the milestone is the source of truth. |

## Pre-release checklist

1. **All milestone issues closed.** Every issue in the `vX.Y.Z` milestone is
   closed (or explicitly moved to a later milestone).
2. **Version string matches.** `VERSION` contains `X.Y.Z`, equal to the release
   you are about to tag (without the `v`).
3. **CI is green on `main`.** The required checks pass on the commit you will
   tag:
   - `cpu_unit_tests` (`pytest tests/ -k cpu`)
   - `pre-commit`
   - `pr-style`
4. **Docs build cleanly** and do not reference a stale version.
5. **Release notes drafted** from the milestone's closed issues (see below).

## Cutting the release

1. Ensure `main` is at the commit you want to release and CI is green.
2. Confirm `VERSION` contains `X.Y.Z`.
3. Tag the release commit and push the tag:

   ```bash
   git tag vX.Y.Z <commit>
   git push origin vX.Y.Z
   ```

   The tag must be reachable from `origin/main` — the publish workflow refuses
   tags that are not ancestors of `main`.
4. Run the **Publish PyPI sdist** workflow (`workflow_dispatch`) with the tag
   input set to `vX.Y.Z`. The workflow:
   - validates the tag format (leading `v` required);
   - checks out the tagged commit and confirms it is reachable from `main`;
   - validates the tag-derived version as PEP 440 and writes it to `VERSION`;
   - builds an sdist only (asserts no wheel is produced);
   - runs `twine check` and uploads to PyPI.
5. Publish the GitHub release for `vX.Y.Z` with notes generated from the
   milestone (below), then close the milestone.

## Release notes from the milestone

Notes are written from the closed issues in the matching milestone. To list
them:

```bash
gh issue list --milestone vX.Y.Z --state closed \
  --json number,title,labels \
  --jq '.[] | "- \(.title) (#\(.number))"'
```

Group the resulting lines by label (e.g. `kind/feature`, `kind/fix`,
`kind/cleanup`) into the release body.

## Notes

- The committed `VERSION` file is the source of truth for local installs and
  source inspection. The publish-time rewrite guarantees the published artifact
  matches the tag without changing the tagged commit.
- Wheels are deliberately not shipped. AReno builds a CUDA extension at install
  time, so a prebuilt wheel would be environment-specific; the sdist lets each
  user build against their own CUDA/PyTorch.
