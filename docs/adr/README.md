# Architecture Decision Records (ADR)

This directory contains immutable architecture decisions for the deterministic serving stack.

## ADR Workflow

1. Create a new ADR from `0000-adr-template.md`.
2. Use the next sequential number (`0001`, `0002`, ...).
3. Set status to `Proposed` when opened.
4. Reference impacted conformance IDs and CI gates.
5. Move to `Accepted` only after required reviewers approve.
6. If superseded, mark old ADR as `Superseded by <ADR-ID>`.

## Required Sections

- Context
- Decision
- Consequences
- Alternatives considered
- Rollout and rollback plan
- Conformance impact

## Ownership

- Platform owners: runtime/CI/conformance governance
- Security owners: supply-chain and attestation controls
- Inference owners: resolver/runner/verifier behavior

See `.github/CODEOWNERS` for reviewer groups.
