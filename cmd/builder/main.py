#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
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


CLOSURE_COMPONENT_RULES: tuple[tuple[str, set[str]], ...] = (
    ("serving_stack", {"serving_stack"}),
    ("cuda_userspace_or_container", {"cuda_lib", "container_image"}),
    ("kernel_libraries", {"kernel_library", "compiled_extension"}),
    ("network_stack", {"network_stack_binary"}),
    ("pmd_driver", {"pmd_driver"}),
)

SHA256_RE = re.compile(r"^sha256:[a-f0-9]{64}$")


def _component_artifacts(
    artifacts: list[dict[str, Any]],
    allowed_types: set[str],
) -> list[dict[str, Any]]:
    selected = [item for item in artifacts if item["artifact_type"] in allowed_types]
    return sorted(
        selected,
        key=lambda item: (
            str(item["artifact_type"]),
            str(item["artifact_id"]),
            str(item["digest"]),
            str(item["immutable_ref"]),
        ),
    )


def _component_descriptor(name: str, selected: list[dict[str, Any]]) -> dict[str, Any]:
    digest_seed = [
        {
            "artifact_id": item["artifact_id"],
            "artifact_type": item["artifact_type"],
            "digest": item["digest"],
            "immutable_ref": item["immutable_ref"],
            "uri": item["uri"],
        }
        for item in selected
    ]
    return {
        "name": name,
        "artifact_types": sorted({item["artifact_type"] for item in selected}),
        "artifact_ids": [item["artifact_id"] for item in selected],
        "artifact_count": len(selected),
        "artifact_digest": sha256_prefixed(canonical_json_bytes(digest_seed)),
    }


def _collect_closure_components(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for name, allowed_types in CLOSURE_COMPONENT_RULES:
        selected = _component_artifacts(artifacts, allowed_types)
        if len(selected) == 0:
            allowed = ", ".join(sorted(allowed_types))
            raise ValidationError(f"Lockfile missing required closure component '{name}' ({allowed})")
        components.append(_component_descriptor(name, selected))
    return sorted(components, key=lambda item: str(item["name"]))


def _collect_oci_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in artifacts:
        if item["source_kind"] != "oci":
            continue
        immutable_ref = str(item.get("immutable_ref", ""))
        uri = str(item.get("uri", ""))
        if not (SHA256_RE.match(immutable_ref) or "@sha256:" in uri):
            continue
        out.append(
            {
                "artifact_id": item["artifact_id"],
                "artifact_type": item["artifact_type"],
                "uri": uri,
                "digest": item["digest"],
                "immutable_ref": immutable_ref,
            }
        )
    return sorted(out, key=lambda item: (str(item["artifact_type"]), str(item["artifact_id"]), str(item["digest"])))


def _build_seed(
    *,
    builder_system: str,
    resolver: dict[str, Any],
    components: list[dict[str, Any]],
    oci_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "builder_system": builder_system,
        "resolver": resolver,
        "components": components,
        "oci_artifacts": oci_artifacts,
    }


def _attestation_statement(
    *,
    builder_system: str,
    closure_uri: str,
    closure_inputs_digest: str,
    components: list[dict[str, Any]],
    oci_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "builder_system": builder_system,
        "closure_uri": closure_uri,
        "closure_inputs_digest": closure_inputs_digest,
        "components": [
            {
                "name": item["name"],
                "artifact_count": item["artifact_count"],
                "artifact_digest": item["artifact_digest"],
            }
            for item in components
        ],
        "oci_artifacts": [
            {
                "artifact_id": item["artifact_id"],
                "digest": item["digest"],
            }
            for item in oci_artifacts
        ],
    }


def build_runtime(lockfile: dict[str, Any], *, builder_system: str = "nix") -> dict[str, Any]:
    if builder_system not in {"nix", "equivalent"}:
        raise ValidationError(f"Unsupported builder_system: {builder_system}")

    validate_with_schema("lockfile.v1.schema.json", lockfile)
    expected = compute_lockfile_digest(lockfile)
    actual = lockfile["canonicalization"]["lockfile_digest"]
    if expected != actual:
        raise ValidationError(
            f"Input lockfile canonicalization.lockfile_digest mismatch: expected={expected} actual={actual}"
        )

    artifacts = stable_sort_artifacts(lockfile["artifacts"])
    deterministic_timestamp = lockfile["generated_at"]
    components = _collect_closure_components(artifacts)
    oci_artifacts = _collect_oci_artifacts(artifacts)
    closure_seed = _build_seed(
        builder_system=builder_system,
        resolver=lockfile["resolver"],
        components=components,
        oci_artifacts=oci_artifacts,
    )
    closure_inputs_digest = sha256_prefixed(canonical_json_bytes(closure_seed))
    closure_uri = f"{builder_system}://closure/{closure_inputs_digest.split(':', 1)[1]}"

    lockfile["artifacts"] = artifacts
    lockfile["runtime_closure_digest"] = closure_inputs_digest
    lockfile["generated_at"] = deterministic_timestamp
    lockfile["build"] = {
        "builder_system": builder_system,
        "closure_uri": closure_uri,
        "closure_inputs_digest": closure_inputs_digest,
        "components": components,
        "oci_artifacts": oci_artifacts,
    }

    statement = _attestation_statement(
        builder_system=builder_system,
        closure_uri=closure_uri,
        closure_inputs_digest=closure_inputs_digest,
        components=components,
        oci_artifacts=oci_artifacts,
    )
    attestations = [item for item in lockfile.get("attestations", []) if item.get("attestation_type") != "build_provenance"]
    attestations.append(
        {
            "attestation_type": "build_provenance",
            "signer": "builder@deterministic-serving-stack",
            "statement_digest": sha256_prefixed(canonical_json_bytes(statement)),
            "timestamp": deterministic_timestamp,
        }
    )
    lockfile["attestations"] = attestations

    lockfile["canonicalization"]["method"] = "json_canonical_v1"
    lockfile["canonicalization"]["lockfile_digest"] = compute_lockfile_digest(lockfile)
    validate_with_schema("lockfile.v1.schema.json", lockfile)
    return lockfile


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic runtime closure digest")
    parser.add_argument("--lockfile", required=True, help="Input lockfile")
    parser.add_argument("--lockfile-out", required=True, help="Output lockfile")
    parser.add_argument("--builder-system", default="nix", choices=["nix", "equivalent"], help="Hermetic builder system")
    args = parser.parse_args()

    lockfile_path = Path(args.lockfile)
    out_path = Path(args.lockfile_out)

    lockfile = json.loads(lockfile_path.read_text(encoding="utf-8"))
    built = build_runtime(lockfile, builder_system=args.builder_system)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_json_text(built), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(str(exc))
        raise SystemExit(1)
