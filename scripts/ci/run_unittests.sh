#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: run_unittests.sh <test-dir>\n' >&2
  exit 1
fi

TEST_DIR="$1"
python3 -m unittest discover -s "$TEST_DIR" -p 'test_*.py' -v
