#!/usr/bin/env bash
# End-to-end Mistral Large 2 TP=8 determinism run (A then B).
# Runs entirely on the remote under nohup so ssh dropouts don't kill it.
set -euo pipefail

cd /mnt/models/repo
source .venv/bin/activate

export HF_HOME=/mnt/models/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
: "${HF_TOKEN:?HF_TOKEN must be set in the caller env}"
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
export NCCL_DEBUG=WARN
export VLLM_BATCH_INVARIANT=1

LOG=/tmp/mistral_full.log
stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }

for variant in A B; do
    manifest="exp/mistral/manifest_${variant,,}.json"
    outdir="exp/mistral/$variant"
    mkdir -p "$outdir"

    log "=== Mistral Large 2 TP=8 Run $variant — resolve ==="
    python cmd/resolver/main.py \
        --manifest "$manifest" \
        --lockfile-out "$outdir/resolved.lock.json" \
        --manifest-out "$outdir/resolved.manifest.json" \
        --resolve-hf \
        --hf-token "$HF_TOKEN" \
        --hf-cache-dir /mnt/models/hf >> "$LOG" 2>&1

    log "=== Mistral Large 2 TP=8 Run $variant — build ==="
    python cmd/builder/main.py \
        --lockfile "$outdir/resolved.lock.json" \
        --lockfile-out "$outdir/built.lock.json" >> "$LOG" 2>&1

    log "=== Mistral Large 2 TP=8 Run $variant — infer ==="
    python cmd/runner/main.py \
        --manifest "$outdir/resolved.manifest.json" \
        --lockfile "$outdir/built.lock.json" \
        --out-dir "$outdir/run" \
        --mode vllm \
        --replica-id replica-0 >> "$LOG" 2>&1

    log "=== Mistral Run $variant complete ==="
done

log "=== Comparing A vs B ==="
python exp/compare_runs.py \
    exp/mistral/A/run \
    exp/mistral/B/run \
    mistral-large2-tp8 \
    /mnt/models/results/mistral_report.json >> "$LOG" 2>&1

log "=== DONE ==="
