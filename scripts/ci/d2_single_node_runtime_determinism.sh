#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "D2: single-node runner determinism"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

MANIFEST="tests/fixtures/positive/manifest.v1.example.json"
RESOLVED="$TMP_DIR/resolved.lock.json"
BUILT="$TMP_DIR/built.lock.json"
RUN_A="$TMP_DIR/run-a"
RUN_B="$TMP_DIR/run-b"
REPORT="$TMP_DIR/verify_report.json"
SUMMARY="$TMP_DIR/verify_summary.txt"

python3 cmd/resolver/main.py --manifest "$MANIFEST" --lockfile-out "$RESOLVED"
python3 cmd/builder/main.py --lockfile "$RESOLVED" --lockfile-out "$BUILT"
python3 cmd/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN_A" --replica-id replica-0
python3 cmd/runner/main.py --manifest "$MANIFEST" --lockfile "$BUILT" --out-dir "$RUN_B" --replica-id replica-0
python3 cmd/verifier/main.py --baseline "$RUN_A/run_bundle.v1.json" --candidate "$RUN_B/run_bundle.v1.json" --report-out "$REPORT" --summary-out "$SUMMARY"

python3 - << 'PY' "$REPORT"
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "conformant":
    raise SystemExit(f"Expected conformant status, got: {report['status']}")
print("Verifier conformant status confirmed")
PY

python3 scripts/ci/mark_conformance.py --id SPEC-7.1-SINGLE-NODE-RUNNER

log "D2 passed"
