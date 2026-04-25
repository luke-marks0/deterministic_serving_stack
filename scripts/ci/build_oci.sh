#!/usr/bin/env bash
# Build the OCI image from the Nix flake and extract the runtime_closure_digest.
#
# This runs in CI (or locally with Nix installed). The output is:
# 1. An OCI image tarball pushed to a registry (pinned by digest)
# 2. A runtime_closure_digest that goes into the lockfile
#
# Usage:
#   scripts/ci/build_oci.sh [--registry REGISTRY] [--push]
set -euo pipefail

source "$(dirname "$0")/common.sh"
require_cmd nix

REGISTRY="${REGISTRY:-ghcr.io/anonymous/deterministic-serving}"
PUSH=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --registry) REGISTRY="$2"; shift 2 ;;
        --push) PUSH=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log "Building runtime closure via Nix flake"

# Build the closure and capture the store path
CLOSURE_PATH=$(nix build .#closure --print-out-paths --no-link 2>&1)
log "Closure path: ${CLOSURE_PATH}"

# Query closure info for the digest
CLOSURE_INFO=$(nix path-info --json --recursive "${CLOSURE_PATH}")
CLOSURE_DIGEST=$(echo "$CLOSURE_INFO" | python3 -c "
import json, sys, hashlib
data = json.load(sys.stdin)
# Normalize: dict-of-dicts or list-of-dicts
if isinstance(data, dict):
    entries = sorted(data.keys())
    canonical = json.dumps({k: data[k] for k in entries}, sort_keys=True, separators=(',',':'))
else:
    entries = sorted(data, key=lambda x: x.get('path',''))
    canonical = json.dumps(entries, sort_keys=True, separators=(',',':'))
digest = hashlib.sha256(canonical.encode()).hexdigest()
print(f'sha256:{digest}')
")

log "Runtime closure digest: ${CLOSURE_DIGEST}"

# Build the OCI image
log "Building OCI image"
OCI_PATH=$(nix build .#oci --print-out-paths --no-link 2>&1)
log "OCI image: ${OCI_PATH}"

# Extract image digest
IMAGE_DIGEST=$(sha256sum "${OCI_PATH}" | awk '{print "sha256:" $1}')
log "Image digest: ${IMAGE_DIGEST}"

# Write metadata
cat > /tmp/nix-build-metadata.json <<EOF
{
  "closure_path": "${CLOSURE_PATH}",
  "runtime_closure_digest": "${CLOSURE_DIGEST}",
  "oci_image_path": "${OCI_PATH}",
  "oci_image_digest": "${IMAGE_DIGEST}",
  "registry": "${REGISTRY}"
}
EOF
log "Metadata written to /tmp/nix-build-metadata.json"

if [ "$PUSH" = true ]; then
    require_cmd skopeo
    TAG="${REGISTRY}:$(git rev-parse --short HEAD)"
    log "Pushing to ${TAG}"
    skopeo copy \
        "docker-archive:${OCI_PATH}" \
        "docker://${TAG}" \
        --digestfile /tmp/oci-pushed-digest

    PUSHED_DIGEST=$(cat /tmp/oci-pushed-digest)
    log "Pushed: ${TAG}@${PUSHED_DIGEST}"

    # Also tag as latest
    skopeo copy \
        "docker://${TAG}" \
        "docker://${REGISTRY}:latest"
    log "Tagged ${REGISTRY}:latest"
fi

log "Build complete"
echo ""
echo "runtime_closure_digest=${CLOSURE_DIGEST}"
echo "oci_image_digest=${IMAGE_DIGEST}"
