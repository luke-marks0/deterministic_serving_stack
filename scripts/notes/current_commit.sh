#!/usr/bin/env bash
set -euo pipefail

if git rev-parse --short HEAD >/dev/null 2>&1; then
  git rev-parse --short HEAD
else
  echo "NO_COMMIT_YET"
fi
