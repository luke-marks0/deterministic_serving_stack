#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Vast.ai TP Determinism Test Runner
# ============================================================================
# Usage:
#   scripts/vastai_tp_test.sh <instance_id>
#
# Prerequisites:
#   pip install vastai && vastai set api-key <key>
#
# The script assumes the instance is already created and running.
# It SSHes in, syncs the repo, installs vLLM, resolves the manifest
# with real HF artifacts, and runs the TP determinism test.
# ============================================================================

INSTANCE_ID="${1:?Usage: $0 <instance_id>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Wait for instance to be ready
# ---------------------------------------------------------------------------
wait_for_instance() {
    local max_wait=600  # 10 minutes
    local elapsed=0
    echo "Waiting for instance $INSTANCE_ID to be ready..."
    while [ "$elapsed" -lt "$max_wait" ]; do
        local status
        status=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(data.get('actual_status', data.get('status_msg', '')))" 2>/dev/null || echo "unknown")
        if [ "$status" = "running" ]; then
            echo "Instance is running."
            return 0
        fi
        echo "  Status: $status (${elapsed}s elapsed)"
        sleep 15
        elapsed=$((elapsed + 15))
    done
    echo "ERROR: Instance did not become ready within ${max_wait}s"
    return 1
}

# ---------------------------------------------------------------------------
# Get SSH connection details
# ---------------------------------------------------------------------------
get_ssh_info() {
    vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
host = d.get('ssh_host', '')
port = d.get('ssh_port', '')
if not host or not port:
    # Fallback: parse from public_ipaddr / ports
    host = d.get('public_ipaddr', '')
    ports = d.get('ports', {})
    for k, v in ports.items():
        if '22/tcp' in k:
            port = str(v[0].get('HostPort', '')) if isinstance(v, list) else ''
            break
print(f'{host} {port}')
"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

wait_for_instance

read -r SSH_HOST SSH_PORT <<< "$(get_ssh_info)"
if [ -z "$SSH_HOST" ] || [ -z "$SSH_PORT" ]; then
    echo "ERROR: Could not determine SSH connection details"
    vastai show instance "$INSTANCE_ID"
    exit 1
fi

SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT root@$SSH_HOST"
SCP_CMD="scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $SSH_PORT"

echo "SSH: $SSH_CMD"

# ---------------------------------------------------------------------------
# Sync repo to instance
# ---------------------------------------------------------------------------
echo "=== Syncing repo to instance ==="
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.nix*' --exclude 'result' \
    -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $SSH_PORT" \
    "$REPO_ROOT/" "root@$SSH_HOST:/workspace/deterministic_serving_stack/"

# ---------------------------------------------------------------------------
# Install dependencies on instance
# ---------------------------------------------------------------------------
echo "=== Installing dependencies ==="
$SSH_CMD << 'REMOTE_SETUP'
set -euo pipefail
cd /workspace/deterministic_serving_stack

# Install vLLM and project deps
pip install -q vllm==0.8.5.post1 pydantic huggingface_hub 2>&1 | tail -5

# Verify GPUs
echo "=== GPU inventory ==="
nvidia-smi -L
python3 -c "import torch; print(f'PyTorch CUDA devices: {torch.cuda.device_count()}')"
python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
REMOTE_SETUP

# ---------------------------------------------------------------------------
# Resolve manifest with real HF artifacts (downloads model)
# ---------------------------------------------------------------------------
echo "=== Resolving manifest (downloading Qwen2.5-32B, ~64GB) ==="
$SSH_CMD << 'REMOTE_RESOLVE'
set -euo pipefail
cd /workspace/deterministic_serving_stack

# Resolve HF model artifacts (downloads + computes SHA-256 digests)
python3 cmd/resolver/main.py \
    --manifest manifests/qwen2.5-32b-tp4.manifest.json \
    --lockfile-out /tmp/tp4-resolved.lock.json \
    --manifest-out manifests/qwen2.5-32b-tp4.resolved.manifest.json \
    --resolve-hf

echo "=== Manifest resolved, building lockfile ==="
python3 cmd/builder/main.py \
    --lockfile /tmp/tp4-resolved.lock.json \
    --lockfile-out /tmp/tp4-built.lock.json

echo "Resolution complete."
REMOTE_RESOLVE

# ---------------------------------------------------------------------------
# Run TP determinism test
# ---------------------------------------------------------------------------
echo "=== Running TP determinism test ==="
$SSH_CMD << 'REMOTE_TEST'
set -euo pipefail
cd /workspace/deterministic_serving_stack

export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN

MANIFEST="manifests/qwen2.5-32b-tp4.resolved.manifest.json"
LOCKFILE="/tmp/tp4-built.lock.json"

echo "=== Run A ==="
python3 cmd/runner/main.py \
    --manifest "$MANIFEST" --lockfile "$LOCKFILE" \
    --out-dir /tmp/tp4-run-a --mode vllm --replica-id replica-0

echo "=== Run B ==="
python3 cmd/runner/main.py \
    --manifest "$MANIFEST" --lockfile "$LOCKFILE" \
    --out-dir /tmp/tp4-run-b --mode vllm --replica-id replica-0

echo "=== Verifying ==="
python3 cmd/verifier/main.py \
    --baseline /tmp/tp4-run-a/run_bundle.v1.json \
    --candidate /tmp/tp4-run-b/run_bundle.v1.json \
    --report-out /tmp/tp4-report.json \
    --summary-out /tmp/tp4-summary.txt

python3 -c "
import json
r = json.load(open('/tmp/tp4-report.json'))
print(f'Status: {r[\"status\"]}')
if r['status'] != 'conformant':
    print(f'First divergence: {r.get(\"first_divergence\", \"N/A\")}')
    if 'numeric_diff_stats' in r:
        print(f'Diff stats: {json.dumps(r[\"numeric_diff_stats\"], indent=2)}')
else:
    print('TP DETERMINISM: CONFORMANT')
"
REMOTE_TEST

# ---------------------------------------------------------------------------
# Copy results back
# ---------------------------------------------------------------------------
echo "=== Copying results ==="
RESULTS_DIR="$REPO_ROOT/experiments/tp_determinism/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RESULTS_DIR"

$SCP_CMD "root@$SSH_HOST:/tmp/tp4-report.json" "$RESULTS_DIR/"
$SCP_CMD "root@$SSH_HOST:/tmp/tp4-summary.txt" "$RESULTS_DIR/"
$SCP_CMD "root@$SSH_HOST:/tmp/tp4-run-a/run_bundle.v1.json" "$RESULTS_DIR/run_bundle_a.json"
$SCP_CMD "root@$SSH_HOST:/tmp/tp4-run-b/run_bundle.v1.json" "$RESULTS_DIR/run_bundle_b.json"

echo ""
echo "=== Results saved to $RESULTS_DIR ==="
echo ""
cat "$RESULTS_DIR/tp4-summary.txt" 2>/dev/null || true

echo ""
echo "Instance $INSTANCE_ID is still running. To destroy:"
echo "  vastai destroy instance $INSTANCE_ID"
