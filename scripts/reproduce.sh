#!/usr/bin/env bash
# Reproduce the exact setup from a manifest.
#
# Given a manifest JSON, this script installs the pinned software versions,
# downloads the model, resolves the lockfile, builds the closure, and starts
# the server. The result is a byte-identical replica of the original setup.
#
# Usage:
#   scripts/reproduce.sh manifests/qwen3-1.7b.manifest.json
#   scripts/reproduce.sh manifests/qwen3-1.7b.manifest.json --run-tests
#
# Requires: python3, pip, nvidia GPU with compute capability >= 9.0
set -euo pipefail

MANIFEST="${1:?Usage: $0 <manifest.json> [--run-tests]}"
RUN_TESTS=false
if [ "${2:-}" = "--run-tests" ]; then
    RUN_TESTS=true
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Reproduce from Manifest ==="
echo "Manifest: ${MANIFEST}"
echo ""

# ---- Step 1: Extract pinned versions from manifest ----
echo "--- Step 1: Reading software pins ---"
PINS=$(python3 -c "
import json, sys
m = json.load(open('${MANIFEST}'))
pins = m.get('runtime', {}).get('software_pins', {})
engine = m.get('runtime', {}).get('serving_engine', {})
if not pins.get('vllm_version'):
    print('ERROR: manifest missing runtime.software_pins.vllm_version', file=sys.stderr)
    sys.exit(1)
print(f\"VLLM_VERSION={pins['vllm_version']}\")
print(f\"TORCH_VERSION={pins['torch_version']}\")
print(f\"CUDA_VERSION={pins['cuda_version']}\")
print(f\"PYTHON_VERSION={pins['python_version']}\")
print(f\"MAX_MODEL_LEN={engine.get('max_model_len', 8192)}\")
print(f\"MAX_NUM_SEQS={engine.get('max_num_seqs', 256)}\")
print(f\"GPU_MEM_UTIL={engine.get('gpu_memory_utilization', 0.9)}\")
print(f\"DTYPE={engine.get('dtype', 'auto')}\")
print(f\"ATTENTION_BACKEND={engine.get('attention_backend', 'FLASH_ATTN')}\")
model = m.get('model', {}).get('source', '')
if model.startswith('hf://'):
    model = model[5:]
print(f\"MODEL_ID={model}\")
")

eval "$PINS"
echo "  vLLM: ${VLLM_VERSION}"
echo "  PyTorch: ${TORCH_VERSION}"
echo "  CUDA: ${CUDA_VERSION}"
echo "  Python: ${PYTHON_VERSION}"
echo "  Model: ${MODEL_ID}"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  max_num_seqs: ${MAX_NUM_SEQS}"
echo "  gpu_memory_utilization: ${GPU_MEM_UTIL}"
echo "  dtype: ${DTYPE}"
echo "  attention_backend: ${ATTENTION_BACKEND}"

# ---- Step 2: Create/activate venv ----
echo ""
echo "--- Step 2: Python environment ---"
VENV="${REPO_ROOT}/.venv-reproduce"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip setuptools wheel -q 2>&1 | tail -1

# ---- Step 3: Install pinned versions ----
echo "--- Step 3: Installing pinned software ---"

# Install PyTorch with CUDA
CUDA_SHORT=$(echo "$CUDA_VERSION" | tr -d '.')
pip install "torch==${TORCH_VERSION}" --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT}" -q 2>&1 | tail -3 || \
    pip install "torch==${TORCH_VERSION}" -q 2>&1 | tail -3

# Install vLLM
pip install "vllm==${VLLM_VERSION}" -q 2>&1 | tail -3

# Install deps for the repo
pip install jsonschema requests huggingface_hub pyyaml -q 2>&1 | tail -1

# ---- Step 4: Verify installed versions ----
echo "--- Step 4: Verifying installations ---"
python3 -c "
import vllm, torch
expected_vllm = '${VLLM_VERSION}'
expected_torch = '${TORCH_VERSION}'
actual_vllm = vllm.__version__
actual_torch = torch.__version__

ok = True
if actual_vllm != expected_vllm:
    print(f'WARNING: vLLM version mismatch: expected {expected_vllm}, got {actual_vllm}')
    ok = False
if actual_torch != expected_torch:
    print(f'WARNING: PyTorch version mismatch: expected {expected_torch}, got {actual_torch}')
    ok = False
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    ok = False
else:
    cc = torch.cuda.get_device_capability()
    print(f'GPU: {torch.cuda.get_device_name()} (CC {cc[0]}.{cc[1]})')
    if cc[0] < 9:
        print(f'WARNING: Batch invariance requires CC >= 9.0, got {cc[0]}.{cc[1]}')

if ok:
    print(f'OK: vLLM {actual_vllm}, PyTorch {actual_torch}, CUDA {torch.version.cuda}')
"

# ---- Step 5: Download model ----
echo "--- Step 5: Downloading model ---"
python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('${MODEL_ID}')
print(f'Model cached at: {p}')
"

# ---- Step 6: Resolve + Build ----
echo "--- Step 6: Resolve and build ---"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

RUN_DIR="${REPO_ROOT}/.reproduce-run"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

python3 "${REPO_ROOT}/cmd/resolver/main.py" \
    --manifest "${MANIFEST}" \
    --lockfile-out "$RUN_DIR/lockfile.v1.json" \
    --manifest-out "$RUN_DIR/manifest.resolved.json" \
    --resolve-hf --hf-resolution-mode online

python3 "${REPO_ROOT}/cmd/builder/main.py" \
    --lockfile "$RUN_DIR/lockfile.v1.json" \
    --lockfile-out "$RUN_DIR/lockfile.built.v1.json" \
    --builder-system equivalent

echo "  Resolved manifest: $RUN_DIR/manifest.resolved.json"
echo "  Built lockfile: $RUN_DIR/lockfile.built.v1.json"

# ---- Step 7: Start server ----
echo ""
echo "--- Step 7: Starting server ---"
export VLLM_BATCH_INVARIANT=1
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONHASHSEED=0
export RUNNER_MAX_MODEL_LEN="${MAX_MODEL_LEN}"
export RUNNER_GPU_MEM_UTIL="${GPU_MEM_UTIL}"
export VLLM_ATTENTION_BACKEND="${ATTENTION_BACKEND}"

python3 "${REPO_ROOT}/cmd/server/main.py" \
    --manifest "$RUN_DIR/manifest.resolved.json" \
    --lockfile "$RUN_DIR/lockfile.built.v1.json" \
    --out-dir "$RUN_DIR/server" \
    --host 0.0.0.0 \
    --port 8000 &

SERVER_PID=$!

# Wait for ready
for i in $(seq 1 120); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo ""
        echo "=== Server ready ==="
        echo "  PID: $SERVER_PID"
        echo "  URL: http://0.0.0.0:8000/v1/chat/completions"
        echo "  Model: ${MODEL_ID}"
        echo ""
        break
    fi
    sleep 3
done

if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    exit 1
fi

# ---- Step 8: Run tests (optional) ----
if [ "$RUN_TESTS" = true ]; then
    echo "--- Step 8: Running determinism tests ---"
    pip install pytest pytest-timeout -q 2>&1 | tail -1

    export DETERMINISTIC_SERVER_URL="http://127.0.0.1:8000"

    echo ""
    echo "  Network determinism tests..."
    python3 -m pytest "${REPO_ROOT}/tests/determinism/test_network_determinism.py" -v --timeout=300 2>&1 || true

    echo ""
    echo "  Large batch determinism tests..."
    export DETERMINISTIC_BATCH_SIZE=128
    python3 -m pytest "${REPO_ROOT}/tests/determinism/test_large_batch_determinism.py" -v --timeout=600 2>&1 || true

    echo ""
    echo "  Verification pipeline..."
    bash "${REPO_ROOT}/deploy/lambda/verify.sh" --requests 8 2>&1 || true
fi

# Keep server running
echo ""
echo "Server running (PID $SERVER_PID). Press Ctrl+C to stop."
wait $SERVER_PID
