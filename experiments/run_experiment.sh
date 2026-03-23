#!/usr/bin/env bash
# Run a single model's experiment set on this server.
# Usage: run_experiment.sh <model> <server-id>
set -euo pipefail

MODEL="$1"
SERVER_ID="$2"
REPO="/home/ubuntu/deterministic_serving_stack"
OUT="/tmp/determinism-experiment"
VENV_ACTIVATE="/home/ubuntu/venv/bin/activate"

# Activate venv if it exists
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
fi

export VLLM_BATCH_INVARIANT=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
export PYTHONHASHSEED=0

echo "=== Starting $MODEL on $SERVER_ID ==="

# Kill any existing vllm (but not this script's python)
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
sleep 3
# Clear GPU memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 2

# Start server
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --seed 42 --port 8000 \
    --max-model-len 32768 --attention-backend FLASH_ATTN \
    > /tmp/vllm-experiment.log 2>&1 &

# Wait for ready
echo "  Waiting for server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"'; then
        echo "  Server ready (${i}0s)"
        break
    fi
    sleep 10
done

# Run repeated test
echo ""
echo "=== Repeated test (20 runs × 30K tokens) ==="
python3 "$REPO/experiments/million_token_determinism.py" \
    --model "$MODEL" --mode repeated --server-id "$SERVER_ID" \
    --out-dir "$OUT" --n-runs 20

# Run diverse test
echo ""
echo "=== Diverse test (34 prompts × 30K tokens) ==="
python3 "$REPO/experiments/million_token_determinism.py" \
    --model "$MODEL" --mode diverse --server-id "$SERVER_ID" \
    --out-dir "$OUT"

echo ""
echo "=== Done: $MODEL on $SERVER_ID ==="
