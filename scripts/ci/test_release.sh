#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running release determinism matrix (D0-D5) and release contracts"
rm -rf .ci-results
bash scripts/ci/d0_schema_determinism.sh
bash scripts/ci/d1_build_determinism.sh
bash scripts/ci/d2_single_node_runtime_determinism.sh
bash scripts/ci/d3_replicated_node_determinism.sh
bash scripts/ci/d4_tp_pp_determinism.sh
bash scripts/ci/d5_network_determinism.sh
bash scripts/ci/release_contracts.sh
python3 scripts/ci/check_release_blockers.py
log "Release determinism matrix and contract proofs passed"
