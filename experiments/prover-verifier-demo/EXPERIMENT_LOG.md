# Experiment log: prover-verifier-demo

Started 2026-05-04. See docs/plans/prover-verifier-demo.md.

- 2026-05-04: Task 0.1 — scaffolded experiment directory.
- 2026-05-04: Task 0.2 — confirmed baseline green (321 unit tests, schema gate, 11 fixtures). pydantic missing from system python; created `.venv` via `uv venv .venv` and installed pydantic+jsonschema there. All canonical schema files are single-line sorted-keys (validated by scripts/ci/check_canonical_json.py).
- 2026-05-04: Task 0.3 — wired ruff + pyright + hypothesis (scoped tooling). `make lint-proverdet`, `make typecheck-proverdet`, `make test-proverdet` short-circuit cleanly when the proverdet code/test paths don't exist yet. Engineer's pre-commit habit: run all three before each commit, alongside `make test-fast` and `make schema`.
- 2026-05-04: Task 1.1 — added prover_graph.v1 schema (placeholder). 10 schema tests cover positive/negative cases including unknown fields, missing run_id, sha256-prefixed commitments, and rejecting string-typed claimed_flops.
- 2026-05-04: Task 1.2 — added replay_request.v1 + replay_evidence.v1 schemas. ReplayRequest uses oneOf for {kind: task | artifact} target; ProofOfWork dtype restricted to bf16/fp16/int8. ReplayEvidence carries pow_stream entries with matmul_dim+rounds+dtype duplicated from the request so the verdict engine can compute observed FLOPs from the transcript alone.
