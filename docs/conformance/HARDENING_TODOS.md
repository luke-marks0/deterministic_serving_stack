# Hardening TODOs (Completed)

This checklist turns the requested hardening items into explicit TODOs and records completion.

- [x] TODO 1: Harden manifest schema with required runtime/hardware/network/comparison sections.
  - Completed in: `schemas/manifest.v1.schema.json`
  - Includes: strict enums/patterns, cross-field constraints for TLS egress behavior, remote code policy, and engine-trace requirements.

- [x] TODO 2: Harden lockfile schema and integrity model.
  - Completed in: `schemas/lockfile.v1.schema.json`, `pkg/common/deterministic.py`, `scripts/ci/lockfile_validate.py`
  - Includes: typed artifacts, required pinned artifact classes, digest format enforcement, canonical lockfile digest computation/verification, and attestations.

- [x] TODO 3: Harden run-bundle and verify-report contracts.
  - Completed in: `schemas/run_bundle.v1.schema.json`, `schemas/verify_report.v1.schema.json`, `cmd/runner/main.py`, `cmd/verifier/main.py`
  - Includes: required provenance/environment/network metadata, machine-readable report fields, and verify summary text output.

- [x] TODO 4: Harden validation with full JSON Schema + negative fixtures + canonical JSON enforcement.
  - Completed in: `scripts/ci/schema_validate.py`, `scripts/ci/fixture_validate.py`, `scripts/ci/check_canonical_json.py`, `scripts/ci/schema_gate.sh`, `tests/fixtures/positive/`, `tests/fixtures/negative/`
  - Includes: Draft 2020-12 schema checks, positive/negative fixture validation, and canonical JSON gate checks.

- [x] TODO 5: Replace placeholder D0-D5 with real component tests and release blockers tied to conformance IDs.
  - Completed in: `cmd/resolver/main.py`, `cmd/builder/main.py`, `cmd/runner/main.py`, `cmd/runner/dispatcher.py`, `cmd/verifier/main.py`, `scripts/ci/d0_*.sh` ... `d5_*.sh`, `scripts/ci/check_release_blockers.py`, `docs/conformance/RELEASE_BLOCKERS.json`
  - Includes: resolver/builder/runner/verifier execution in CI scripts, conformance marker emission, release blocker enforcement, and long-run/topology determinism tests.
