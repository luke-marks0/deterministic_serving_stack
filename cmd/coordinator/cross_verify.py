#!/usr/bin/env python3
"""Cross-pod verification for replicated serving (Phase 5).

Collects capture logs from all replicas and verifies that requests
dispatched to different replicas with the same input produce the
same output, proving cross-pod determinism.

Usage:
    python3 cmd/coordinator/cross_verify.py \
        --dispatch-log /path/to/dispatch.jsonl \
        --replica-dirs replica-0=/path/to/r0,replica-1=/path/to/r1 \
        --manifest manifest.resolved.json \
        --lockfile lockfile.built.v1.json \
        --out-dir /path/to/verify-output
"""
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
    sha256_prefixed,
    utc_now_iso,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def cross_verify(
    dispatch_log_path: Path,
    replica_dirs: dict[str, Path],
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Compare per-replica bundles for cross-pod determinism."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dispatch log
    dispatch_entries = []
    for line in dispatch_log_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            dispatch_entries.append(json.loads(line))

    # Group dispatches by replica
    by_replica: dict[str, list[dict[str, Any]]] = {}
    for entry in dispatch_entries:
        replica = entry.get("replica", "unknown")
        by_replica.setdefault(replica, []).append(entry)

    # Load per-replica run bundles
    replica_bundles: dict[str, dict[str, Any]] = {}
    for name, rdir in replica_dirs.items():
        bundle_path = rdir / "run_bundle.v1.json"
        if bundle_path.exists():
            replica_bundles[name] = _load_json(bundle_path)

    # Cross-replica checks
    checks = []

    # 1. Runtime closure digest must be identical across all replicas
    closure_digests = {
        name: bundle.get("runtime_closure_digest", "")
        for name, bundle in replica_bundles.items()
    }
    unique_digests = set(closure_digests.values())
    checks.append({
        "check": "runtime_closure_digest_uniform",
        "outcome": "pass" if len(unique_digests) == 1 else "fail",
        "detail": f"{len(unique_digests)} unique digest(s) across {len(replica_bundles)} replicas",
        "values": closure_digests,
    })

    # 2. Hardware fingerprint must be identical (for strict_hardware mode)
    hw_fingerprints = {
        name: bundle.get("environment_info", {}).get("hardware_fingerprint", "")
        for name, bundle in replica_bundles.items()
    }
    unique_hw = set(hw_fingerprints.values())
    checks.append({
        "check": "hardware_fingerprint_uniform",
        "outcome": "pass" if len(unique_hw) == 1 else "warn",
        "detail": f"{len(unique_hw)} unique fingerprint(s)",
        "values": hw_fingerprints,
    })

    # 3. Environment versions must match
    env_keys = ["vllm_version", "torch_version", "cuda_version", "driver_version"]
    env_mismatches = []
    for key in env_keys:
        vals = {
            name: bundle.get("environment_info", {}).get(key, "")
            for name, bundle in replica_bundles.items()
        }
        if len(set(vals.values())) > 1:
            env_mismatches.append({"key": key, "values": vals})

    checks.append({
        "check": "environment_versions_uniform",
        "outcome": "pass" if not env_mismatches else "fail",
        "detail": f"{len(env_mismatches)} version mismatch(es)",
        "mismatches": env_mismatches,
    })

    # 4. Compare observables across replicas (tokens, logits, activations)
    observable_names = ["tokens", "logits", "activations", "engine_trace", "network_egress"]
    replica_names = list(replica_bundles.keys())
    observable_mismatches = []

    if len(replica_names) >= 2:
        baseline_name = replica_names[0]
        baseline_bundle = replica_bundles[baseline_name]
        baseline_dir = replica_dirs[baseline_name]

        for obs_name in observable_names:
            baseline_obs_info = baseline_bundle.get("observables", {}).get(obs_name)
            if not baseline_obs_info:
                continue
            baseline_obs_path = baseline_dir / baseline_obs_info["path"]
            if not baseline_obs_path.exists():
                continue
            baseline_obs = _load_json(baseline_obs_path)

            for other_name in replica_names[1:]:
                other_bundle = replica_bundles[other_name]
                other_dir = replica_dirs[other_name]
                other_obs_info = other_bundle.get("observables", {}).get(obs_name)
                if not other_obs_info:
                    observable_mismatches.append({
                        "observable": obs_name,
                        "replica": other_name,
                        "reason": "observable missing from bundle",
                    })
                    continue
                other_obs_path = other_dir / other_obs_info["path"]
                if not other_obs_path.exists():
                    observable_mismatches.append({
                        "observable": obs_name,
                        "replica": other_name,
                        "reason": "observable file missing",
                    })
                    continue
                other_obs = _load_json(other_obs_path)
                if baseline_obs != other_obs:
                    observable_mismatches.append({
                        "observable": obs_name,
                        "replicas": [baseline_name, other_name],
                        "reason": "content mismatch",
                        "baseline_digest": baseline_obs_info.get("digest", ""),
                        "other_digest": other_obs_info.get("digest", ""),
                    })

    checks.append({
        "check": "observables_cross_replica",
        "outcome": "pass" if not observable_mismatches else "fail",
        "detail": f"{len(observable_mismatches)} observable mismatch(es) across replicas",
        "mismatches": observable_mismatches,
    })

    # Overall status
    all_pass = all(c["outcome"] == "pass" for c in checks)
    report = {
        "cross_verify_version": "v1",
        "generated_at": utc_now_iso(),
        "status": "conformant" if all_pass else "non_conformant",
        "replicas": list(replica_bundles.keys()),
        "dispatch_entries": len(dispatch_entries),
        "checks": checks,
    }

    report_path = out_dir / "cross_verify_report.json"
    report_path.write_text(canonical_json_text(report), encoding="utf-8")

    summary_lines = [
        "Cross-Pod Verification Summary",
        f"Status: {report['status']}",
        f"Replicas: {', '.join(report['replicas'])}",
        f"Dispatch entries: {report['dispatch_entries']}",
    ]
    for check in checks:
        summary_lines.append(f"  {check['check']}: {check['outcome']} — {check['detail']}")

    summary_path = out_dir / "cross_verify_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-pod verification")
    parser.add_argument("--dispatch-log", required=True)
    parser.add_argument("--replica-dirs", required=True, help="name=path,name=path")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lockfile", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    replica_dirs = {}
    for pair in args.replica_dirs.split(","):
        name, path = pair.split("=", 1)
        replica_dirs[name.strip()] = Path(path.strip())

    manifest = _load_json(Path(args.manifest))
    lockfile = _load_json(Path(args.lockfile))

    report = cross_verify(
        Path(args.dispatch_log), replica_dirs, manifest, lockfile, Path(args.out_dir),
    )
    print(f"Status: {report['status']}")
    return 0 if report["status"] == "conformant" else 1


if __name__ == "__main__":
    raise SystemExit(main())
