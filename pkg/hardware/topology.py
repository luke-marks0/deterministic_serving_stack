"""Topology-aware hardware conformance for TP/PP and multi-rack (Phases 6-7).

Validates collective stack pinning, NVLink/NVSwitch topology,
and cross-rank hardware consistency.
"""
from __future__ import annotations

from typing import Any

from pkg.common.contracts import ValidationError
from pkg.common.deterministic import canonical_json_bytes, sha256_prefixed


def validate_collective_stack(lockfile: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate that the collective stack artifact is pinned for TP/PP topologies."""
    mode = manifest["hardware_profile"]["topology"]["mode"]
    if mode not in ("tensor_parallel", "pipeline_parallel"):
        return {"required": False, "status": "not_applicable"}

    collective_artifacts = [
        a for a in lockfile.get("artifacts", [])
        if a["artifact_type"] == "collective_stack"
    ]

    if not collective_artifacts:
        raise ValidationError(
            f"Topology mode '{mode}' requires a collective_stack artifact in the lockfile"
        )

    artifact = collective_artifacts[0]
    return {
        "required": True,
        "status": "pinned",
        "artifact_id": artifact["artifact_id"],
        "digest": artifact["digest"],
        "immutable_ref": artifact["immutable_ref"],
    }


def compute_topology_fingerprint(manifest: dict[str, Any]) -> str:
    """Compute a fingerprint of the topology configuration."""
    topo = manifest["hardware_profile"]["topology"]
    gpu = manifest["hardware_profile"]["gpu"]
    seed = {
        "mode": topo["mode"],
        "node_count": topo["node_count"],
        "rack_count": topo["rack_count"],
        "collective_fabric": topo["collective_fabric"],
        "gpu_model": gpu["model"],
        "gpu_count": gpu["count"],
    }
    return sha256_prefixed(canonical_json_bytes(seed))


def validate_cross_rank_consistency(
    rank_hardware_profiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate that all ranks have consistent hardware.

    For TP/PP, every rank must have the same GPU model, driver,
    CUDA version, and NIC configuration.
    """
    if not rank_hardware_profiles:
        return {"status": "no_ranks", "diffs": []}

    baseline = rank_hardware_profiles[0]
    baseline_fp = sha256_prefixed(canonical_json_bytes(baseline))
    diffs = []

    for rank_idx, profile in enumerate(rank_hardware_profiles[1:], start=1):
        rank_fp = sha256_prefixed(canonical_json_bytes(profile))
        if rank_fp != baseline_fp:
            diffs.append({
                "rank": rank_idx,
                "expected_fingerprint": baseline_fp,
                "actual_fingerprint": rank_fp,
            })

    return {
        "status": "conformant" if not diffs else "non_conformant",
        "baseline_fingerprint": baseline_fp,
        "ranks_checked": len(rank_hardware_profiles),
        "diffs": diffs,
    }


def compute_placement_constraints(manifest: dict[str, Any]) -> dict[str, Any]:
    """Compute K8s scheduling constraints from the manifest topology.

    Returns node selectors, affinities, and anti-affinities for
    topology-aware placement.
    """
    topo = manifest["hardware_profile"]["topology"]
    gpu = manifest["hardware_profile"]["gpu"]
    nic = manifest["hardware_profile"]["nic"]

    node_selector = {
        "hardware.gpu.vendor": gpu["vendor"],
        "hardware.gpu.model": gpu["model"],
        "hardware.nic.model": nic["model"],
    }

    constraints: dict[str, Any] = {
        "node_selector": node_selector,
        "topology_mode": topo["mode"],
        "node_count": topo["node_count"],
        "rack_count": topo["rack_count"],
    }

    if topo["mode"] in ("tensor_parallel", "pipeline_parallel"):
        # TP/PP needs all nodes in the same rack for NVLink
        if topo["collective_fabric"] == "intra_node":
            constraints["anti_affinity"] = None
            constraints["affinity"] = {"topology_key": "kubernetes.io/hostname"}
        elif topo["collective_fabric"] == "intra_rack":
            constraints["affinity"] = {"topology_key": "topology.kubernetes.io/zone"}
        else:
            constraints["affinity"] = None

    if topo["mode"] == "replicated":
        # Replicated prefers spreading across racks
        constraints["pod_anti_affinity"] = {
            "topology_key": "topology.kubernetes.io/zone",
            "max_skew": 1,
        }

    return constraints


def validate_rack_placement(
    pod_assignments: list[dict[str, Any]],
    rack_count: int,
) -> dict[str, Any]:
    """Validate that pod-to-rack assignments respect topology constraints."""
    rack_loads: dict[int, int] = {}
    for assignment in pod_assignments:
        rack = assignment.get("rack_id", 0)
        rack_loads[rack] = rack_loads.get(rack, 0) + 1

    max_load = max(rack_loads.values()) if rack_loads else 0
    min_load = min(rack_loads.values()) if rack_loads else 0
    skew = max_load - min_load

    return {
        "rack_count": rack_count,
        "rack_loads": rack_loads,
        "max_skew": skew,
        "balanced": skew <= 1,
        "racks_used": len(rack_loads),
    }
