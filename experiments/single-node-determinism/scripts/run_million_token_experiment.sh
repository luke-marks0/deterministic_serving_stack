#!/usr/bin/env bash
# Million-Token Determinism Experiment — Orchestrator
#
# Runs 1M token generation for each model on two servers,
# then compares results (same-server and cross-server).
#
# Usage: experiments/run_million_token_experiment.sh
#
set -euo pipefail

S1="ubuntu@192.222.50.183"
S2="ubuntu@192.222.56.186"
SSH_OPTS="-o ServerAliveInterval=15 -o ServerAliveCountMax=120 -o StrictHostKeyChecking=no"

MODELS=(
    "mistralai/Mistral-7B-Instruct-v0.3"
    "Qwen/Qwen3-8B"
    "Qwen/Qwen3-30B-A3B"
)

EXPERIMENT_DIR="/tmp/million-token-experiment"
REPO_DIR="/home/ubuntu/deterministic_serving_stack"

start_model_server() {
    local SERVER="$1"
    local MODEL="$2"
    echo "  Starting $MODEL on $SERVER..."
    ssh $SSH_OPTS "$SERVER" "
        killall -9 python3 2>/dev/null; sleep 3
        source /home/ubuntu/venv/bin/activate
        # Download model if needed
        python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('${MODEL}')\" 2>/dev/null
        VLLM_BATCH_INVARIANT=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 TORCH_DETERMINISTIC=1 PYTHONHASHSEED=0 \
        nohup python3 -m vllm.entrypoints.openai.api_server \
            --model ${MODEL} --seed 42 --port 8000 --max-model-len 32768 \
            --attention-backend FLASH_ATTN > /tmp/vllm-experiment.log 2>&1 &
    "
}

wait_for_server() {
    local SERVER="$1"
    echo -n "  Waiting for server on $SERVER"
    for i in $(seq 1 60); do
        if ssh $SSH_OPTS "$SERVER" "curl -s http://localhost:8000/v1/models" 2>/dev/null | grep -q '"id"'; then
            echo " ready (${i}0s)"
            return 0
        fi
        echo -n "."
        sleep 10
    done
    echo " TIMEOUT"
    ssh $SSH_OPTS "$SERVER" "tail -10 /tmp/vllm-experiment.log"
    return 1
}

run_generation() {
    local SERVER="$1"
    local MODEL="$2"
    local RUN_ID="$3"
    echo "  Running $RUN_ID on $SERVER ($MODEL)..."
    ssh $SSH_OPTS "$SERVER" "
        cd ${REPO_DIR}
        python3 experiments/million_token_determinism.py \
            --model '${MODEL}' \
            --run-id '${RUN_ID}' \
            --out-dir '${EXPERIMENT_DIR}'
    "
}

echo "================================================================"
echo "  Million-Token Determinism Experiment"
echo "  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
echo ""
echo "  Server 1: $S1"
echo "  Server 2: $S2"
echo "  Models: ${MODELS[*]}"
echo "  Target: 1,000,000 tokens per model per run"
echo ""

# Sync experiment script to both servers
rsync -az experiments/ "$S1:${REPO_DIR}/experiments/"
rsync -az experiments/ "$S2:${REPO_DIR}/experiments/"

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|.*/||')
    echo ""
    echo "================================================================"
    echo "  Model: $MODEL"
    echo "================================================================"

    # Start model on both servers
    start_model_server "$S1" "$MODEL" &
    start_model_server "$S2" "$MODEL" &
    wait

    wait_for_server "$S1" || continue
    wait_for_server "$S2" || continue

    # Run S1-Run1 and S2-Run1 in parallel
    echo ""
    echo "--- Phase 1: S1-Run1 + S2-Run1 (parallel) ---"
    run_generation "$S1" "$MODEL" "${MODEL_SHORT}-s1-run1" &
    PID_S1=$!
    run_generation "$S2" "$MODEL" "${MODEL_SHORT}-s2-run1" &
    PID_S2=$!
    wait $PID_S1
    wait $PID_S2

    # Run S1-Run2 (sequential, same server as Run1)
    echo ""
    echo "--- Phase 2: S1-Run2 (same-server repeat) ---"
    run_generation "$S1" "$MODEL" "${MODEL_SHORT}-s1-run2"

    # Pull results from both servers
    echo ""
    echo "--- Collecting results ---"
    mkdir -p "/tmp/million-token-results/${MODEL_SHORT}"
    rsync -az "$S1:${EXPERIMENT_DIR}/${MODEL_SHORT}-s1-run1/" "/tmp/million-token-results/${MODEL_SHORT}/s1-run1/"
    rsync -az "$S1:${EXPERIMENT_DIR}/${MODEL_SHORT}-s1-run2/" "/tmp/million-token-results/${MODEL_SHORT}/s1-run2/"
    rsync -az "$S2:${EXPERIMENT_DIR}/${MODEL_SHORT}-s2-run1/" "/tmp/million-token-results/${MODEL_SHORT}/s2-run1/"

    # Compare
    echo ""
    echo "--- Same-server comparison (S1-Run1 vs S1-Run2) ---"
    python3 experiments/million_token_determinism.py \
        --compare "/tmp/million-token-results/${MODEL_SHORT}/s1-run1" \
                  "/tmp/million-token-results/${MODEL_SHORT}/s1-run2"

    echo ""
    echo "--- Cross-server comparison (S1-Run1 vs S2-Run1) ---"
    python3 experiments/million_token_determinism.py \
        --compare "/tmp/million-token-results/${MODEL_SHORT}/s1-run1" \
                  "/tmp/million-token-results/${MODEL_SHORT}/s2-run1"
done

echo ""
echo "================================================================"
echo "  Experiment Complete — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "================================================================"
