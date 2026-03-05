# Deterministic Serving Stack

Deterministic inference stack for vLLM-style workloads, designed for reproducibility and auditability from single-node H100 baseline to future multi-node and multi-rack operation.

## Project Status

This repository currently contains:

1. Versioned schemas and deterministic reference implementations for resolver, builder, runner, and verifier.
2. CI gates and determinism matrix scripts (`D0` to `D5`).
3. Conformance tracking tied to spec requirements.
4. Planning and governance artifacts for long-term delivery.

## Repository Structure

```text
.
├── cmd/                    # CLI entrypoints: resolver, builder, runner, verifier
├── pkg/                    # Shared library code
├── schemas/                # JSON schemas (manifest, lockfile, run_bundle, verify_report)
├── scripts/
│   ├── ci/                 # CI gate scripts and conformance checks
│   └── notes/              # Note helpers (e.g., current commit helper)
├── tests/                  # unit, integration, determinism, e2e, fixtures
├── docs/
│   ├── adr/                # Architecture Decision Records
│   └── conformance/        # Conformance IDs, release blockers, CI gate docs
├── plan/                   # Specification, build plan, and development notes
├── deploy/                 # Kubernetes/Helm deployment scaffolding
├── nix/                    # Nix packaging/image scaffolding
└── .github/workflows/      # PR/main/nightly/release workflows
```

## Start Here (Plan Directory)

Project planning and spec documents are in `plan/`:

1. `plan/spec.md` (authoritative technical spec)
2. `plan/BUILD_PLAN.md` (phase-by-phase implementation plan)
3. `plan/README.md` (planning conventions)
4. `plan/notes/README.md` (note structure and rules)
5. `plan/notes/INDEX.md` (active note directories)

Recommended reading order:

1. `plan/spec.md`
2. `plan/BUILD_PLAN.md`
3. `docs/conformance/spec_requirements.v1.json`

## How To Use This Repo

### Prerequisites

1. `bash`
2. `python3`
3. `rg` (ripgrep) for lint checks

### Typical local workflow

1. Validate fast gate:
   `make ci-pr`
2. Run full gate before major merges:
   `make ci-main`
3. Validate determinism release matrix:
   `make ci-release`

### Run a full local pipeline example

```bash
tmp="$(mktemp -d)"
python3 cmd/resolver/main.py \
  --manifest tests/fixtures/positive/manifest.v1.example.json \
  --lockfile-out "$tmp/resolved.lock.json"
python3 cmd/builder/main.py \
  --lockfile "$tmp/resolved.lock.json" \
  --lockfile-out "$tmp/built.lock.json"
python3 cmd/runner/main.py \
  --manifest tests/fixtures/positive/manifest.v1.example.json \
  --lockfile "$tmp/built.lock.json" \
  --out-dir "$tmp/run"
python3 cmd/verifier/main.py \
  --baseline "$tmp/run/run_bundle.v1.json" \
  --candidate "$tmp/run/run_bundle.v1.json" \
  --report-out "$tmp/verify_report.json" \
  --summary-out "$tmp/verify_summary.txt"
```

Generated artifacts to inspect:

1. `$tmp/run/manifest.json`
2. `$tmp/run/lockfile.json`
3. `$tmp/run/run_bundle.v1.json`
4. `$tmp/verify_report.json`

## Developer Workflow Requirement

After making any large change, always add a dated summary note under `plan/notes/` in the correct category.

Each note must include:

1. Date in the note title.
2. `Commit:` hash (current `HEAD`, or `NO_COMMIT_YET` if no commit exists yet).
3. What changed.
4. Why it changed.
5. Risks or tradeoffs.
6. Validation and test follow-up.

Category placement:

1. `plan/notes/features/<feature-name>/` for feature work.
2. `plan/notes/bugs/<bug-name-or-id>/` for bug work.
3. `plan/notes/issues/<issue-name-or-id>/` for broader issues or investigations.

If existing note files do not fit the work, create a new subdirectory and note file.

## CI Gate Scaffolding

GitHub workflow definitions:

1. `.github/workflows/pr-gate.yml`
2. `.github/workflows/main-gate.yml`
3. `.github/workflows/nightly.yml`
4. `.github/workflows/release-gate.yml`

CI gate behavior and conformance mapping are documented in:

1. `docs/conformance/CI_GATES.md`
2. `docs/conformance/spec_requirements.v1.json`
3. `docs/conformance/RELEASE_BLOCKERS.json`

Local equivalents:

1. `make ci-pr`
2. `make ci-main`
3. `make ci-nightly`
4. `make ci-release`
