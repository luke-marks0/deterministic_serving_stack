#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.common.contracts import ValidationError, validate_with_schema
from pkg.common.deterministic import (
    canonical_json_bytes,
    canonical_json_text,
    compute_lockfile_digest,
    sha256_prefixed,
    stable_sort_artifacts,
)


def build_runtime(lockfile: dict[str, Any]) -> dict[str, Any]:
    validate_with_schema("lockfile.v1.schema.json", lockfile)
    expected = compute_lockfile_digest(lockfile)
    actual = lockfile["canonicalization"]["lockfile_digest"]
    if expected != actual:
        raise ValidationError(
            f"Input lockfile canonicalization.lockfile_digest mismatch: expected={expected} actual={actual}"
        )

    artifacts = stable_sort_artifacts(lockfile["artifacts"])
    deterministic_timestamp = lockfile["generated_at"]
    closure_seed = {
        "artifacts": [{"id": a["artifact_id"], "digest": a["digest"]} for a in artifacts],
        "resolver": lockfile["resolver"],
    }

    lockfile["artifacts"] = artifacts
    lockfile["runtime_closure_digest"] = sha256_prefixed(canonical_json_bytes(closure_seed))
    lockfile["generated_at"] = deterministic_timestamp

    lockfile.setdefault("attestations", []).append(
        {
            "attestation_type": "build_provenance",
            "signer": "builder@deterministic-serving-stack",
            "statement_digest": sha256_prefixed(canonical_json_bytes(closure_seed)),
            "timestamp": deterministic_timestamp,
        }
    )

    lockfile["canonicalization"]["method"] = "json_canonical_v1"
    lockfile["canonicalization"]["lockfile_digest"] = compute_lockfile_digest(lockfile)
    validate_with_schema("lockfile.v1.schema.json", lockfile)
    return lockfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic runtime closure digest")
    parser.add_argument("--lockfile", required=True, help="Input lockfile")
    parser.add_argument("--lockfile-out", required=True, help="Output lockfile")
    args = parser.parse_args()

    lockfile_path = Path(args.lockfile)
    out_path = Path(args.lockfile_out)

    lockfile = json.loads(lockfile_path.read_text(encoding="utf-8"))
    built = build_runtime(lockfile)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json_text(built), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
