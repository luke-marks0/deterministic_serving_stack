#!/usr/bin/env bash
# Run all 3 models sequentially in a single tmux-friendly script.
# Usage: tmux new -s experiment 'bash ~/deterministic_serving_stack/experiments/run_all_models.sh <server-id>'
set -euo pipefail

SERVER_ID="${1:?Usage: $0 <server-id>}"
REPO="/home/ubuntu/deterministic_serving_stack"
OUT="/tmp/determinism-experiment"
VENV_ACTIVATE="/home/ubuntu/venv/bin/activate"

if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
fi

export VLLM_BATCH_INVARIANT=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_DETERMINISTIC=1
export PYTHONHASHSEED=0

MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-30B-A3B"
)

start_server() {
    local MODEL="$1"
    echo "  Killing existing vllm..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    sleep 3
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 2

    echo "  Starting $MODEL..."
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" --seed 42 --port 8000 \
        --max-model-len 32768 --attention-backend FLASH_ATTN \
        > /tmp/vllm-experiment.log 2>&1 &

    echo -n "  Waiting for server"
    for i in $(seq 1 60); do
        if curl -s http://localhost:8000/v1/models 2>/dev/null | grep -q '"id"'; then
            echo " ready (${i}0s)"
            return 0
        fi
        echo -n "."
        sleep 10
    done
    echo " TIMEOUT"
    tail -10 /tmp/vllm-experiment.log
    return 1
}

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')

    echo ""
    echo "================================================================"
    echo "  MODEL: $MODEL"
    echo "  SERVER: $SERVER_ID"
    echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "================================================================"

    start_server "$MODEL" || continue

    echo ""
    echo "--- Repeated test (20 runs × 30K tokens) ---"
    python3 "$REPO/experiments/million_token_determinism.py" \
        --model "$MODEL" --mode repeated --server-id "$SERVER_ID" \
        --out-dir "$OUT" --n-runs 20

    echo ""
    echo "--- Diverse test (34 prompts × 30K tokens) ---"
    python3 "$REPO/experiments/million_token_determinism.py" \
        --model "$MODEL" --mode diverse --server-id "$SERVER_ID" \
        --out-dir "$OUT"

    echo ""
    echo "=== DONE: $MODEL_SHORT on $SERVER_ID at $(date -u '+%H:%M:%S UTC') ==="
done

echo ""
echo "================================================================"
echo "  ALL MODELS COMPLETE on $SERVER_ID"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
