#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running nightly suite (full + chaos + D0-D5 determinism)"
bash scripts/ci/test_full.sh
bash scripts/ci/run_unittests.sh tests/chaos
bash scripts/ci/d0_schema_determinism.sh
bash scripts/ci/d1_build_determinism.sh
bash scripts/ci/d2_single_node_runtime_determinism.sh
bash scripts/ci/d3_replicated_node_determinism.sh
bash scripts/ci/d4_tp_pp_determinism.sh
bash scripts/ci/d5_network_determinism.sh
bash scripts/ci/run_unittests.sh tests/determinism/test_long_run_determinism.py
log "Nightly suite passed"
