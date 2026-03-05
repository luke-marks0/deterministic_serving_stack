# Deterministic Serving Stack

This project implements reproducible vLLM inference with Nix and Kubernetes, with long-term goals for multi-node and multi-rack deterministic operation.

## Start Here

Project planning and spec documents are in the `plan/` directory:

- `plan/spec.md`
- `plan/BUILD_PLAN.md`
- `plan/README.md`
- `plan/notes/README.md`
- `plan/notes/INDEX.md`

Read `plan/spec.md` first, then follow `plan/BUILD_PLAN.md` for implementation sequencing and testing expectations.

## Developer Workflow Requirement

After making any large change, always add a dated summary note under `plan/notes/` in the correct category.
Each note must include both:

1. Date (in the title).
2. `Commit:` hash (current `HEAD`, or `NO_COMMIT_YET` if no commit exists yet).

Category placement:

1. `plan/notes/features/<feature-name>/` for feature work.
2. `plan/notes/bugs/<bug-name-or-id>/` for bug work.
3. `plan/notes/issues/<issue-name-or-id>/` for broader issues or investigations.

Each summary must include:

1. What changed.
2. Why it changed.
3. Risks or tradeoffs.
4. Validation and test follow-up.

This is required to keep decisions auditable and help future contributors understand intent.

If existing note files do not fit the work, create a new subdirectory and note file.

## CI Gate Scaffolding

CI workflows are defined in `.github/workflows/` and documented in `docs/conformance/CI_GATES.md`.

Local equivalents:

1. `make ci-pr`
2. `make ci-main`
3. `make ci-nightly`
4. `make ci-release`
