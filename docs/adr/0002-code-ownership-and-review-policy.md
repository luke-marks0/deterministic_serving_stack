# ADR-0002: Code Ownership and Review Policy

- Status: Accepted
- Date: 2026-03-05
- Owners: platform
- Reviewers: inference, security
- Supersedes: None
- Superseded by: None

## Context

The project spans supply-chain integrity, deterministic runtime behavior, and verification logic. Review responsibilities must be explicit.

## Decision

1. Repository ownership is defined in `.github/CODEOWNERS`.
2. Changes to schemas, conformance policies, and CI gates require platform review.
3. Changes to resolver/runner/verifier logic require inference review.
4. Changes touching security/attestation artifacts require security review.
5. Large changes MUST include plan notes with date and commit hash.

## Consequences

- Clear approval routing and accountability.
- Slightly slower merges for cross-cutting changes.

## Alternatives Considered

- Single-owner model: rejected due to domain complexity.
- Optional reviews: rejected due to auditability risk.

## Rollout and Rollback Plan

- Rollout: add CODEOWNERS and enforce PR reviews in repo settings.
- Rollback: remove CODEOWNERS entries only via explicit governance ADR.

## Conformance Impact

- Conformance IDs: governance and release-blocker integrity IDs.
- CI gates affected: PR and release.
