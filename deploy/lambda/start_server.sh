#!/usr/bin/env bash
# Start the deterministic server on a remote Lambda node.
# Usage: deploy/lambda/start_server.sh <ip>
set -euo pipefail

IP="${1:?Usage: $0 <ip>}"

echo "=== Starting server on ${IP} ==="

ssh -o StrictHostKeyChecking=no "ubuntu@${IP}" bash -s <<'REMOTE'
set -euo pipefail
source /home/ubuntu/venv/bin/activate
export PYTHONPATH="/home/ubuntu/deterministic_serving_stack"
cd /home/ubuntu/deterministic_serving_stack

# Kill existing server
pkill -f "cmd/server/main.py" 2>/dev/null || true
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

RUN_DIR="/home/ubuntu/server-runs/serve-live"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"

# Resolve + Build
python3 cmd/resolver/main.py \
    --manifest manifests/qwen3-1.7b.manifest.json \
    --lockfile-out "$RUN_DIR/lockfile.v1.json" \
    --manifest-out "$RUN_DIR/manifest.resolved.json" \
    --resolve-hf --hf-resolution-mode online 2>&1 | tail -2

python3 cmd/builder/main.py \
    --lockfile "$RUN_DIR/lockfile.v1.json" \
    --lockfile-out "$RUN_DIR/lockfile.built.v1.json" \
    --builder-system equivalent 2>&1 | tail -2

# Start server in background
nohup python3 cmd/server/main.py \
    --manifest "$RUN_DIR/manifest.resolved.json" \
    --lockfile "$RUN_DIR/lockfile.built.v1.json" \
    --out-dir "$RUN_DIR" \
    --host 0.0.0.0 \
    --port 8000 \
    > "$RUN_DIR/server.log" 2>&1 &

# Wait for ready
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8000/health >/dev/null 2>&1; then
        echo "Server ready"
        exit 0
    fi
    sleep 3
done
echo "Server failed to start"
tail -20 "$RUN_DIR/server.log"
exit 1
REMOTE
