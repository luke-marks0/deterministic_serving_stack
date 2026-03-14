#!/usr/bin/env bash
# Start the deterministic serving stack on a Lambda instance.
#
# Usage:
#   deploy/lambda/serve.sh [--manifest PATH] [--port PORT]
#
# Prerequisites: run setup.sh first, or have vLLM + model cached.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
VENV="/home/ubuntu/venv"
MANIFEST="${REPO_ROOT}/manifests/qwen3-1.7b.manifest.json"
OUT_BASE="/home/ubuntu/server-runs"
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --manifest) MANIFEST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

source "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_ID="serve-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="${OUT_BASE}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "=== Deterministic Serving Stack ==="
echo "Manifest: ${MANIFEST}"
echo "Run dir:  ${RUN_DIR}"
echo ""

# Step 1: Resolve manifest -> lockfile + resolved manifest
echo "--- Step 1: Resolver ---"
LOCKFILE="${RUN_DIR}/lockfile.v1.json"
RESOLVED_MANIFEST="${RUN_DIR}/manifest.resolved.json"

python3 "${REPO_ROOT}/cmd/resolver/main.py" \
    --manifest "${MANIFEST}" \
    --lockfile-out "${LOCKFILE}" \
    --manifest-out "${RESOLVED_MANIFEST}" \
    --resolve-hf \
    --hf-resolution-mode online

echo "Lockfile: ${LOCKFILE}"
echo "Resolved manifest: ${RESOLVED_MANIFEST}"

# Step 2: Build (reference descriptor)
echo "--- Step 2: Builder ---"
BUILT_LOCKFILE="${RUN_DIR}/lockfile.built.v1.json"

python3 "${REPO_ROOT}/cmd/builder/main.py" \
    --lockfile "${LOCKFILE}" \
    --lockfile-out "${BUILT_LOCKFILE}" \
    --builder-system equivalent

echo "Built lockfile: ${BUILT_LOCKFILE}"

# Step 3: Start server
echo "--- Step 3: Starting server ---"
exec python3 "${REPO_ROOT}/cmd/server/main.py" \
    --manifest "${RESOLVED_MANIFEST}" \
    --lockfile "${BUILT_LOCKFILE}" \
    --out-dir "${RUN_DIR}" \
    --host 0.0.0.0 \
    --port "${PORT}"
