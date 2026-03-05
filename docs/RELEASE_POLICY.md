# Release and Support Policy

## Versioning

Use semantic tags: `vMAJOR.MINOR.PATCH`.

## Release Preconditions

1. All release blocker conformance IDs pass.
2. Release CI gate passes (`make ci-release`).
3. Conformance catalog and schema versions are unchanged or explicitly migrated.

## Required Release Artifacts

1. Schema versions (`manifest/lockfile/run_bundle/verify_report`).
2. Conformance catalog version.
3. CI run references for release gate.
4. Change notes with date + commit hash.

## Support Window

1. Current minor: full support.
2. Previous minor: critical/security fixes only.

## Breaking Changes

Breaking contract changes require:

1. Major version bump.
2. Migration document.
3. Explicit release note section: compatibility impact.
