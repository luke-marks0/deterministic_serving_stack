# CI Gates

This repository defines four CI gate tiers aligned with `plan/BUILD_PLAN.md`.

## PR Gate

Runs on pull requests:

1. `lint`
2. `schema`
3. `test-fast` (unit + integration)

## Main Gate

Runs on pushes to `main`:

1. `lint`
2. `schema`
3. `test-full` (unit + integration + e2e + determinism)

## Nightly Gate

Runs on schedule:

1. `lint`
2. `schema`
3. `test-nightly` (full + chaos + long-run determinism coverage)

## Release Gate

Runs on `v*` tags:

1. `lint`
2. `schema`
3. `test-release` (D0-D5 executable matrix)
4. Release blocker check from `docs/conformance/RELEASE_BLOCKERS.json`

## Notes

D0-D5 execute real resolver/builder/runner/verifier/dispatcher flows and emit conformance markers consumed by release blocker enforcement.

Conformance IDs are maintained in `docs/conformance/spec_requirements.v1.json` and validated in `scripts/ci/check_conformance_catalog.py` as part of the schema gate.
