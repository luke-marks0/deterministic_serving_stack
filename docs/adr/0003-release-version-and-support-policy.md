# ADR-0003: Release Versioning and Support Policy

- Status: Accepted
- Date: 2026-03-05
- Owners: platform
- Reviewers: inference, security
- Supersedes: None
- Superseded by: None

## Context

Even as an internal tool, deterministic behavior and reproducibility require controlled releases and traceable support windows.

## Decision

1. Releases are tag-based (`vMAJOR.MINOR.PATCH`).
2. Release gate MUST pass D0-D5 and release blockers before tagging.
3. Support policy:
   1. Current minor release: full support.
   2. Previous minor release: security and critical fixes only.
4. Breaking contract changes require major version bump and migration notes.
5. Every release MUST publish:
   1. schema versions,
   2. conformance catalog version,
   3. CI run references.

## Consequences

- Higher release discipline and traceability.
- Additional release overhead.

## Alternatives Considered

- Ad-hoc internal snapshots: rejected due to audit and rollback gaps.
- Date-only versions: rejected due to weak semantic meaning.

## Rollout and Rollback Plan

- Rollout: enforce in release CI policy and docs.
- Rollback: emergency hotfix tags allowed with post-incident ADR.

## Conformance Impact

- Conformance IDs: release blocker and report integrity IDs.
- CI gates affected: release.
