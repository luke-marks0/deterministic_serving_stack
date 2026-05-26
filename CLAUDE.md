# Deterministic Serving Stack

## Project overview

Bitwise-deterministic LLM inference. Given the same model weights, prompts, and config flags, two independent servers produce identical token outputs. Proven across millions of tokens on H100s.

## Key commands

```bash
# Run unit tests (no GPU required)
python3 -m unittest discover -s tests/unit -v

# Run a specific test module
python3 -m unittest tests.unit.test_schema_files -v

# Run determinism tests (requires GPU)
python3 -m unittest discover -s tests/determinism -v

# Validate schemas
bash scripts/ci/schema_gate.sh

# Run the mock runner (no GPU — wiring only, not a determinism proof)
python3 modules/inference/runner/main.py --manifest modules/inference/manifests/qwen3-1.7b.manifest.json --lockfile <lockfile> --out-dir /tmp/run --mode mock
```

## Code layout

```
modules/                — Capability layer; each module physically owns its code + shared core/ + Pipeline
  build/                — Hermetic Nix runtime: builder/, lockfiles/, nix/ (flake.nix + flake.lock live at root)
  inference/            — Deterministic vLLM: server/, runner/, resolver/, capture/, manifest/ (model), manifests/ (data)
  network/              — networkdet/ (sim TCP/IP) + native/libnetdet/ (DPDK transmit)
  attestation/          — freivalds/, e2e/, proverdet/ + verifier/ (+ verifier_cli/server) + prover/
  memory/               — PoSE memory wipe + erasure attestation (pose/ sub-package + api.py)
  utils/                — provisioning/replay helpers (re-exports core/common)
  core/                 — Shared: common/ (canonical JSON, digests, schema validation, HF) + schemas/ (JSON Schema defs)
workflows/              — Recipe book: runnable compositions of modules (e.g. deterministic_inference_server.py)
tests/                  — unit/, integration/, e2e/, determinism/, modules/, fixtures/
scripts/ci/             — CI scripts (schema gates, conformance checks, determinism gates d0–d6)
scripts/                — General utilities (reproduce.sh); scripts/lambda/ (lambda CLI)
scripts/deploy/         — Lambda/vast/warden provisioning (utils-owned)
demos/                  — Runnable end-to-end scenarios: e2e-audit (used by scripts/demo.sh), prover-verifier (CPU demo of the protocol)
tests/conformance/      — Machine-readable spec catalog + release blockers (read by CI gates)
flake.nix, flake.lock   — Hermetic build entrypoint + pin (at root: the flake's src=self packages repo-wide code and callers invoke `.#`)
```

## Capability modules and workflows

The repo is organized **by function**. Each `modules/<capability>/` **physically
owns its code** — the former `pkg/` and `cmd/` top-level trees were consolidated
into the modules, and `core/` holds the shared `common` helpers plus the JSON
Schema `schemas`. A capability need not be a Python package (build is nix +
shell); the contract is a documented `README.md`, and for Python ones a small
`api.py`. `workflows/` composes modules via `modules.Pipeline` into runnable
recipes. New modules: add a `README.md` (Purpose · Interface · Artifacts ·
Requirements · Example) and a smoke test in `tests/modules/`. Design and
implementation plans live on the `experiments` branch.

## Experiments vs demos

**On `main` there is no `experiments/` folder.** Research artifacts live on the
**`experiments` branch** (`git checkout experiments` to work on them, or browse
the branch on GitHub). The branch holds all the experimental work — including
memory_wipe, multinode-determinism, freivalds-attestation, overhead-benchmark,
multi-gpu-determinism, single-node-determinism, network-determinism,
deterministic-cuda-graphs, task-graph-prototype, timing_channel, and the
research portions of the demos.

**On `main`, runnable end-to-end scenarios live under `demos/`:**
- `demos/e2e-audit/` — the audit replay demo used by `scripts/demo.sh`
- `demos/prover-verifier/` — the prover↔verifier protocol demo (CPU; `./demo.sh --quick`)

Each experiment folder on the `experiments` branch should contain:
- `plan.md` — the experiment design and implementation plan
- `EXPERIMENT_LOG.md` — append-only log of commands, milestones, roadblocks, and results
- `scripts/`, `data/`, `reports/`, `figures/`

Do NOT scatter experiment artifacts across `scripts/`, `results/`, `docs/reports/`, or other top-level directories. If code is reusable across experiments, put it in the relevant `modules/<capability>/` (or `modules/core/` if shared) with tests in `tests/unit/`.

Use `/experiment <idea>` to start a new experiment — it walks through design, planning, critique, and implementation.

## Determinism flags (the "c3" config)

The full deterministic stack requires all three:
1. `enforce_eager=True` — disables CUDA Graphs and torch.compile
2. `CUBLAS_WORKSPACE_CONFIG=:4096:8` — deterministic cuBLAS kernels
3. `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` — batch-order invariance

Env vars MUST be set before `import vllm` or `import torch`.

## Testing patterns

- Tests use `unittest.TestCase` (not pytest)
- Test files: `tests/unit/test_*.py`, `tests/e2e/test_*.py`, etc.
- Helper utilities: `tests/helpers.py` (read_json, write_json, run_cmd)
- Repo root path in tests: `Path(__file__).resolve().parents[2]`
- Repo root path in scripts at `scripts/*.py`: `Path(__file__).resolve().parents[1]`
- Repo root path in scripts at `demos/*/scripts/*.py`: `Path(__file__).resolve().parents[2]`

## Style

- Python, no framework — just stdlib + vllm + torch on GPU machines
- Canonical JSON: `json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"`
- SHA256 digests prefixed: `sha256:<hex>`
- Use `uv` for Python tooling, never pip/pipx/apt
