#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running nightly suite (full + chaos + extended determinism)"
bash scripts/ci/test_full.sh
bash scripts/ci/run_unittests.sh tests/chaos
bash scripts/ci/d0_schema_determinism.sh
bash scripts/ci/d5_network_determinism.sh
log "Nightly suite passed"
