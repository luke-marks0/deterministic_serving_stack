# ADR-0001: Contract Versioning and Compatibility Policy

- Status: Accepted
- Date: 2026-03-05
- Owners: platform
- Reviewers: inference, security
- Supersedes: None
- Superseded by: None

## Context

Manifest, lockfile, run-bundle, and verify-report contracts must evolve safely while preserving deterministic reproducibility and auditability.

## Decision

1. All machine contracts are versioned (`*.vN.*`).
2. Backward incompatible changes require a major version increment (`v1` -> `v2`).
3. Within a major version:
   1. Required fields MUST NOT be removed.
   2. Enum values MAY be added only if existing behavior remains unchanged.
   3. Canonicalization rules MUST remain stable.
4. CI enforces schema compatibility and canonical formatting on every gate.
5. Release gate blocks if conformance blocker IDs are not satisfied.

## Consequences

- Contract changes become explicit and reviewable.
- Short-term friction increases for schema edits.
- Long-term regression risk and audit ambiguity decrease.

## Alternatives Considered

- Unversioned schemas: rejected due to migration ambiguity.
- Date-based schema names: rejected due to weak compatibility semantics.

## Rollout and Rollback Plan

- Rollout: enforced immediately via CI scripts.
- Rollback: revert schema changes and conformance blocker changes in one commit.

## Conformance Impact

- Conformance IDs: contract and lockfile determinism IDs.
- CI gates affected: PR, main, nightly, release.
