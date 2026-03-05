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
    compute_bundle_digest,
    compute_lockfile_digest,
    sha256_prefixed,
    utc_now_iso,
)


def _seed_for_request(run_id: str, req_id: str, prompt: str) -> int:
    digest = sha256_prefixed(canonical_json_bytes({"run_id": run_id, "id": req_id, "prompt": prompt}))
    return int(digest.split(":", 1)[1][:16], 16)


def _tokens(seed: int, count: int = 8) -> list[int]:
    vals = []
    value = seed
    for _ in range(count):
        value = (1103515245 * value + 12345) % (2**31)
        vals.append(value % 50000)
    return vals


def _logits(tokens: list[int]) -> list[float]:
    return [round((tok % 997) / 997.0, 8) for tok in tokens]


def _activations(tokens: list[int]) -> list[float]:
    return [round(((tok * 3) % 991) / 991.0, 8) for tok in tokens]


def _network_frame_hex(seed: int, req_id: str) -> str:
    payload = canonical_json_text({"req_id": req_id, "seed": seed}).encode("utf-8")
    return payload.hex()


def _write_json(path: Path, data: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json_text(data), encoding="utf-8")
    return sha256_prefixed(path.read_bytes())


def _hardware_fingerprint(hardware_profile: dict[str, Any]) -> str:
    return sha256_prefixed(canonical_json_bytes(hardware_profile))


def _value_repr(value: Any) -> str:
    return canonical_json_text(value).strip()


def _diff_hardware(expected: Any, actual: Any, path: str, out: list[dict[str, str]]) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            out.append({"path": path, "expected": _value_repr(expected), "actual": _value_repr(actual)})
            return
        keys = sorted(set(expected.keys()) | set(actual.keys()))
        for key in keys:
            key_path = f"{path}.{key}"
            if key not in expected:
                out.append({"path": key_path, "expected": "<absent>", "actual": _value_repr(actual[key])})
                continue
            if key not in actual:
                out.append({"path": key_path, "expected": _value_repr(expected[key]), "actual": "<absent>"})
                continue
            _diff_hardware(expected[key], actual[key], key_path, out)
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or expected != actual:
            out.append({"path": path, "expected": _value_repr(expected), "actual": _value_repr(actual)})
        return

    if expected != actual:
        out.append({"path": path, "expected": _value_repr(expected), "actual": _value_repr(actual)})


def _extract_observed_hardware(
    expected_hardware: dict[str, Any],
    runtime_hardware: dict[str, Any] | None,
) -> dict[str, Any]:
    if runtime_hardware is None:
        return expected_hardware
    if not isinstance(runtime_hardware, dict):
        raise ValidationError("runtime_hardware must be a JSON object when provided")
    return runtime_hardware


def _hardware_conformance_record(
    expected_hardware: dict[str, Any],
    observed_hardware: dict[str, Any],
    *,
    strict_hardware: bool,
) -> dict[str, Any]:
    diffs: list[dict[str, str]] = []
    _diff_hardware(expected_hardware, observed_hardware, "$.hardware_profile", diffs)
    status = "conformant" if len(diffs) == 0 else "non_conformant"
    record = {
        "status": status,
        "strict_hardware": strict_hardware,
        "expected_fingerprint": _hardware_fingerprint(expected_hardware),
        "actual_fingerprint": _hardware_fingerprint(observed_hardware),
        "diffs": diffs,
    }
    if strict_hardware and len(diffs) > 0:
        first = diffs[0]
        raise ValidationError(
            "Hardware conformance failed (strict_hardware=true): "
            f"{first['path']} expected={first['expected']} actual={first['actual']}"
        )
    return record


def run(
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
    out_dir: Path,
    replica_id: str,
    *,
    runtime_hardware: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_with_schema("manifest.v1.schema.json", manifest)
    validate_with_schema("lockfile.v1.schema.json", lockfile)

    manifest_digest = sha256_prefixed(canonical_json_bytes(manifest))
    if lockfile["manifest_digest"] != manifest_digest:
        raise ValidationError("Lockfile manifest_digest mismatch")

    expected_lockfile_digest = compute_lockfile_digest(lockfile)
    if lockfile["canonicalization"]["lockfile_digest"] != expected_lockfile_digest:
        raise ValidationError("Lockfile canonicalization.lockfile_digest mismatch")

    lock_artifacts_by_id = {item["artifact_id"]: item for item in lockfile["artifacts"]}
    for artifact_input in manifest["artifact_inputs"]:
        artifact_id = artifact_input["artifact_id"]
        if artifact_id not in lock_artifacts_by_id:
            raise ValidationError(f"Lockfile missing artifact required by manifest: {artifact_id}")
        expected_digest = artifact_input.get("expected_digest")
        if expected_digest is not None:
            actual_digest = lock_artifacts_by_id[artifact_id]["digest"]
            if actual_digest != expected_digest:
                raise ValidationError(
                    f"Artifact digest mismatch for {artifact_id}: expected={expected_digest} actual={actual_digest}"
                )

    expected_hardware = manifest["hardware_profile"]
    observed_hardware = _extract_observed_hardware(expected_hardware, runtime_hardware)
    hardware_conformance = _hardware_conformance_record(
        expected_hardware,
        observed_hardware,
        strict_hardware=bool(manifest["runtime"]["strict_hardware"]),
    )
    observed_gpu = observed_hardware.get("gpu", expected_hardware["gpu"])
    observed_nic = observed_hardware.get("nic", expected_hardware["nic"])

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_copy = out_dir / "manifest.json"
    lockfile_copy = out_dir / "lockfile.json"
    manifest_copy.write_text(canonical_json_text(manifest), encoding="utf-8")
    lockfile_copy.write_text(canonical_json_text(lockfile), encoding="utf-8")

    request_outputs: list[dict[str, Any]] = []
    engine_events: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []

    target_batch = manifest["runtime"]["batch_cardinality"]["target_batch_size"]
    for idx, req in enumerate(manifest["requests"]):
        seed = _seed_for_request(manifest["run_id"], req["id"], req["prompt"])
        toks = _tokens(seed)
        lgt = _logits(toks)
        act = _activations(toks)

        request_outputs.append(
            {
                "id": req["id"],
                "tokens": toks,
                "logits": lgt,
                "activations": act,
            }
        )

        engine_events.append(
            {
                "step": idx,
                "event": "batch_composition",
                "batch_size": target_batch,
                "request_id": req["id"],
                "replica_id": replica_id,
            }
        )

        if "request_reorder" in manifest["runtime"]["engine_trace"]["events"]:
            engine_events.append(
                {
                    "step": idx,
                    "event": "request_reorder",
                    "before": idx,
                    "after": idx,
                }
            )

        if "attention_backend_selection" in manifest["runtime"]["engine_trace"]["events"]:
            engine_events.append(
                {
                    "step": idx,
                    "event": "attention_backend_selection",
                    "backend": "flash_attention_2",
                }
            )

        if "collective_algorithm_selection" in manifest["runtime"]["engine_trace"]["events"]:
            topo = manifest["hardware_profile"]["topology"]["mode"]
            algorithm = "none" if topo == "single_node" else "ring_all_reduce"
            engine_events.append(
                {
                    "step": idx,
                    "event": "collective_algorithm_selection",
                    "algorithm": algorithm,
                }
            )

        frames.append(
            {
                "request_id": req["id"],
                "frame_hex": _network_frame_hex(seed, req["id"]),
            }
        )

    observables_dir = out_dir / "observables"
    tokens_path = observables_dir / "tokens.json"
    logits_path = observables_dir / "logits.json"
    activations_path = observables_dir / "activations.json"
    trace_path = observables_dir / "engine_trace.json"
    network_path = observables_dir / "network_egress.json"

    tokens_digest = _write_json(tokens_path, [{"id": r["id"], "tokens": r["tokens"]} for r in request_outputs])
    logits_digest = _write_json(logits_path, [{"id": r["id"], "logits": r["logits"]} for r in request_outputs])
    activations_digest = _write_json(activations_path, [{"id": r["id"], "activations": r["activations"]} for r in request_outputs])
    trace_digest = _write_json(trace_path, engine_events)
    network_digest = _write_json(network_path, frames)

    run_bundle: dict[str, Any] = {
        "run_bundle_version": "v1",
        "run_id": manifest["run_id"],
        "created_at": utc_now_iso(),
        "manifest_copy": {
            "path": str(manifest_copy.relative_to(out_dir)),
            "digest": sha256_prefixed(manifest_copy.read_bytes()),
        },
        "lockfile_copy": {
            "path": str(lockfile_copy.relative_to(out_dir)),
            "digest": sha256_prefixed(lockfile_copy.read_bytes()),
        },
        "runtime_closure_digest": lockfile["runtime_closure_digest"],
        "resolved_artifact_digests": [
            {
                "artifact_id": a["artifact_id"],
                "artifact_type": a["artifact_type"],
                "digest": a["digest"],
            }
            for a in lockfile["artifacts"]
        ],
        "environment_info": {
            "vllm_version": "0.1.0",
            "torch_version": "2.5.0",
            "cuda_version": "12.4",
            "driver_version": observed_gpu.get("driver_version", expected_hardware["gpu"]["driver_version"]),
            "gpu_inventory": [observed_gpu.get("model", expected_hardware["gpu"]["model"])],
            "hardware_fingerprint": hardware_conformance["actual_fingerprint"],
        },
        "hardware_conformance": hardware_conformance,
        "execution_trace_metadata": {
            "actual_batch_sizes": [target_batch for _ in manifest["requests"]],
            "resolved_args": {
                "batch_policy": manifest["runtime"]["batch_policy"],
                "strict_hardware": str(manifest["runtime"]["strict_hardware"]).lower(),
            },
            "resolved_env": {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_LAUNCH_BLOCKING": str(int(manifest["runtime"]["deterministic_knobs"]["cuda_launch_blocking"])),
            },
        },
        "network_provenance": {
            "capture_path": str(network_path.relative_to(out_dir)),
            "capture_digest": network_digest,
            "frame_count": len(frames),
            "capture_mode": "userspace_pre_enqueue",
            "nic_fingerprint": sha256_prefixed(canonical_json_bytes(observed_nic)),
            "security_mode": manifest["network"]["security_mode"],
        },
        "observables": {
            "tokens": {
                "path": str(tokens_path.relative_to(out_dir)),
                "digest": tokens_digest,
            },
            "logits": {
                "path": str(logits_path.relative_to(out_dir)),
                "digest": logits_digest,
            },
            "activations": {
                "path": str(activations_path.relative_to(out_dir)),
                "digest": activations_digest,
            },
            "engine_trace": {
                "path": str(trace_path.relative_to(out_dir)),
                "digest": trace_digest,
            },
            "network_egress": {
                "path": str(network_path.relative_to(out_dir)),
                "digest": network_digest,
            },
        },
        "attestations": [
            {
                "attestation_type": "run_provenance",
                "signer": "runner@deterministic-serving-stack",
                "statement_digest": sha256_prefixed(canonical_json_bytes({"run_id": manifest["run_id"], "replica_id": replica_id})),
                "timestamp": utc_now_iso(),
            }
        ],
        "bundle_digest": "sha256:" + ("0" * 64),
    }

    run_bundle["bundle_digest"] = compute_bundle_digest(run_bundle)
    validate_with_schema("run_bundle.v1.schema.json", run_bundle)
    _write_json(out_dir / "run_bundle.v1.json", run_bundle)
    return run_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic runner scaffold")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    parser.add_argument("--lockfile", required=True, help="Lockfile JSON path")
    parser.add_argument("--out-dir", required=True, help="Output bundle directory")
    parser.add_argument("--replica-id", default="replica-0", help="Replica identifier")
    parser.add_argument("--runtime-hardware", help="Optional observed runtime hardware profile JSON path")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    lockfile = json.loads(Path(args.lockfile).read_text(encoding="utf-8"))
    runtime_hardware = None
    if args.runtime_hardware:
        runtime_hardware = json.loads(Path(args.runtime_hardware).read_text(encoding="utf-8"))

    run(
        manifest,
        lockfile,
        Path(args.out_dir),
        args.replica_id,
        runtime_hardware=runtime_hardware,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
