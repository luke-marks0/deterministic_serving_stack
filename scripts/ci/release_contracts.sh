#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

log "Release contracts: builder, resolver, deploy, provenance, and verifier"

python3 -m unittest tests.integration.test_builder_closure_profile -v
python3 scripts/ci/mark_conformance.py --id SPEC-3.2-NIX-BUILDER

python3 -m unittest tests.integration.test_resolver_hf_resolution -v
python3 scripts/ci/mark_conformance.py --id SPEC-4.4-REMOTE-CODE-PIN
python3 scripts/ci/mark_conformance.py --id SPEC-5.1-RESOLVE-HF-IMMUTABLE-REV
python3 scripts/ci/mark_conformance.py --id SPEC-5.1-ENUMERATE-HF-REQUIRED-FILES
python3 scripts/ci/mark_conformance.py --id SPEC-5.1-HF-PER-FILE-DIGESTS

python3 -m unittest \
  tests.integration.test_runner_context_provenance \
  tests.e2e.test_deploy_assets \
  tests.e2e.test_verifier_outputs \
  -v
python3 scripts/ci/mark_conformance.py --id SPEC-6.2-K8S-IMMUTABLE-IMAGE-DIGEST
python3 scripts/ci/mark_conformance.py --id SPEC-7.2-RUN-INPUTS-PRESENT
python3 scripts/ci/mark_conformance.py --id SPEC-7.2-K8S-MOUNT-AND-RECORD
python3 scripts/ci/mark_conformance.py --id SPEC-12-RUN-BUNDLE-CONTENT
python3 scripts/ci/mark_conformance.py --id SPEC-12-PROVENANCE-RERUN-SUFFICIENT
python3 scripts/ci/mark_conformance.py --id SPEC-13-VERIFY-REPORT

log "Release contracts passed"
