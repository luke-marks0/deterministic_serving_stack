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
from pkg.common.hf_resolution import (
    HFResolutionError,
    HuggingFaceHubClient,
    resolve_hf_model,
)

MODEL_ARTIFACT_TYPES = {
    "model_weights",
    "model_config",
    "tokenizer",
    "generation_config",
    "chat_template",
    "prompt_formatter",
    "remote_code",
}


def _artifact_from_input(item: dict[str, Any], model_source: str) -> dict[str, Any]:
    source_bytes = canonical_json_bytes(item)
    digest = item.get("expected_digest") or sha256_prefixed(source_bytes)
    return {
        "artifact_id": item["artifact_id"],
        "artifact_type": item["artifact_type"],
        "name": item.get("name", item["artifact_id"]),
        "source_kind": item["source_kind"],
        "uri": item["source_uri"],
        "immutable_ref": item["immutable_ref"],
        "digest": digest,
        "size_bytes": int(item.get("size_bytes", max(1, len(source_bytes)))),
        "hash_algorithm": "sha256",
        "resolved_from": model_source,
        "build_output": item["artifact_type"] in {"compiled_extension", "kernel_library"},
    }


def _merge_model_artifacts(
    existing_artifact_inputs: list[dict[str, Any]],
    resolved_model_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    keep = [item for item in existing_artifact_inputs if item.get("artifact_type") not in MODEL_ARTIFACT_TYPES]
    merged = [*keep, *resolved_model_artifacts]
    merged = sorted(
        merged,
        key=lambda item: (
            str(item.get("artifact_type", "")),
            str(item.get("artifact_id", "")),
            str(item.get("immutable_ref", "")),
            str(item.get("expected_digest", "")),
        ),
    )
    return merged


def _resolve_manifest_hf_model(
    manifest: dict[str, Any],
    *,
    hf_cache_dir: Path | None,
    hf_token: str | None,
) -> dict[str, Any]:
    source = manifest["model"]["source"]
    if not str(source).startswith("hf://"):
        return manifest

    client = HuggingFaceHubClient(token=hf_token)
    resolved = resolve_hf_model(
        manifest["model"],
        bool(manifest["model"]["trust_remote_code"]),
        client=client,
        cache_dir=hf_cache_dir,
    )

    manifest["model"]["resolved_revision"] = resolved.resolved_revision
    manifest["model"]["weights_revision"] = resolved.resolved_revision
    manifest["model"]["tokenizer_revision"] = resolved.resolved_revision
    manifest["model"]["required_files"] = resolved.required_files
    if resolved.remote_code is not None:
        manifest["model"]["remote_code"] = resolved.remote_code
    elif "remote_code" in manifest["model"]:
        del manifest["model"]["remote_code"]

    manifest["artifact_inputs"] = _merge_model_artifacts(manifest["artifact_inputs"], resolved.model_artifacts)
    return manifest


def resolve_manifest_to_lockfile(
    manifest: dict[str, Any],
    *,
    resolve_hf: bool = False,
    hf_cache_dir: Path | None = None,
    hf_token: str | None = None,
) -> dict[str, Any]:
    validate_with_schema("manifest.v1.schema.json", manifest)
    if resolve_hf:
        manifest = _resolve_manifest_hf_model(manifest, hf_cache_dir=hf_cache_dir, hf_token=hf_token)
        validate_with_schema("manifest.v1.schema.json", manifest)
    deterministic_timestamp = manifest["created_at"]

    artifacts = [_artifact_from_input(item, manifest["model"]["source"]) for item in manifest["artifact_inputs"]]
    artifacts = stable_sort_artifacts(artifacts)

    runtime_seed = {
        "runtime": manifest["runtime"],
        "hardware": manifest["hardware_profile"],
        "network": manifest["network"],
    }

    lockfile = {
        "lockfile_version": "v1",
        "generated_at": deterministic_timestamp,
        "manifest_digest": sha256_prefixed(canonical_json_bytes(manifest)),
        "runtime_closure_digest": sha256_prefixed(canonical_json_bytes(runtime_seed)),
        "resolver": {
            "name": "deterministic-resolver",
            "version": "0.1.0"
        },
        "canonicalization": {
            "method": "json_canonical_v1",
            "lockfile_digest": "sha256:" + ("0" * 64)
        },
        "artifacts": artifacts,
        "attestations": [
            {
                "attestation_type": "resolver_provenance",
                "signer": "resolver@deterministic-serving-stack",
                "statement_digest": sha256_prefixed(canonical_json_bytes({"artifacts": artifacts})),
                "timestamp": deterministic_timestamp,
            }
        ],
    }
    lockfile["canonicalization"]["lockfile_digest"] = compute_lockfile_digest(lockfile)
    validate_with_schema("lockfile.v1.schema.json", lockfile)
    return lockfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve manifest into deterministic lockfile")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--lockfile-out", required=True, help="Path to lockfile JSON output")
    parser.add_argument("--resolve-hf", action="store_true", help="Resolve Hugging Face model files and digests")
    parser.add_argument("--hf-cache-dir", help="Optional HF cache directory")
    parser.add_argument("--hf-token", help="Optional HF token")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    lockfile_path = Path(args.lockfile_out)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lockfile = resolve_manifest_to_lockfile(
        manifest,
        resolve_hf=bool(args.resolve_hf),
        hf_cache_dir=Path(args.hf_cache_dir) if args.hf_cache_dir else None,
        hf_token=args.hf_token,
    )

    lockfile_path.parent.mkdir(parents=True, exist_ok=True)
    lockfile_path.write_text(canonical_json_text(lockfile), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HFResolutionError as exc:
        print(str(exc))
        raise SystemExit(1)
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
