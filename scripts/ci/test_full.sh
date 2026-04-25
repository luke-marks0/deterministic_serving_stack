#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Running full test suite"
bash scripts/ci/test_fast.sh
bash scripts/ci/run_unittests.sh tests/e2e
bash scripts/ci/run_unittests.sh tests/determinism
log "Full test suite passed"
