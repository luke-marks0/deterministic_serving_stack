#!/usr/bin/env bash
# prover-verifier demo entry point (Task 10.1).
#
# Spawns prover + verifier locally (or talks to a remote pair, see --remote),
# runs three scenarios — benign, mixed_lora, lora_loading — and exits 0 iff
# the actual verdicts match the expected outcomes.
#
# Usage:
#   ./demo.sh                # default: 5s per scenario
#   ./demo.sh --quick        # 2s per scenario (CI/dev)
#   ./demo.sh --long         # 15s per scenario (deep view)
#   PROVER_HOST=10.0.0.1 PROVER_PORT=8000 \
#   VERIFIER_HOST=10.0.0.2 VERIFIER_PORT=9000 ./demo.sh --remote

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PER_SCENARIO=5.0
REMOTE=""
for arg in "$@"; do
  case "$arg" in
    --quick)  PER_SCENARIO=2.0 ;;
    --long)   PER_SCENARIO=15.0 ;;
    --remote) REMOTE="--remote" ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}"
      exit 0
      ;;
    *)
      echo "Unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# Pick a Python. Order of preference:
#   1. ${REPO_ROOT}/.venv/bin/python3            (already provisioned)
#   2. uv venv + uv pip install (provision now)  (per CLAUDE.md, never pip)
#   3. system python3 (best-effort fallback; will fail if pydantic+jsonschema
#      aren't on the path, but worth trying for users who manage deps elsewhere)
PY=""
if [[ -x "${REPO_ROOT}/.venv/bin/python3" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python3"
elif command -v uv >/dev/null 2>&1; then
  echo "demo.sh: provisioning ${REPO_ROOT}/.venv via uv (one-time)" >&2
  uv venv --quiet "${REPO_ROOT}/.venv"
  # Demo runtime deps only; matplotlib stays out (the demo doesn't plot).
  uv pip install --quiet --python "${REPO_ROOT}/.venv/bin/python3" \
    pydantic jsonschema
  PY="${REPO_ROOT}/.venv/bin/python3"
else
  echo "demo.sh: no .venv and no uv on PATH — falling back to system python3." >&2
  echo "         If imports fail, install uv (https://github.com/astral-sh/uv)" >&2
  echo "         and re-run; we'll provision the venv automatically." >&2
  PY="python3"
fi

cd "${REPO_ROOT}"
exec "${PY}" experiments/prover-verifier-demo/scripts/demo_driver.py \
  --per-scenario "${PER_SCENARIO}" ${REMOTE}
