"""Multi-rack production hardening: placement, failure domains, capacity (Phase 7).

Provides rack-aware dispatch policies, failure domain isolation,
deterministic retry semantics, and SLO tracking.
"""
from __future__ import annotations

from typing import Any

from pkg.common.deterministic import canonical_json_bytes, sha256_prefixed


class RackTopology:
    """Represents the physical rack layout for multi-rack deployments."""

    def __init__(self, rack_count: int, nodes_per_rack: int):
        self.rack_count = rack_count
        self.nodes_per_rack = nodes_per_rack
        self.total_nodes = rack_count * nodes_per_rack

    def rack_for_node(self, node_index: int) -> int:
        return node_index % self.rack_count

    def nodes_in_rack(self, rack_id: int) -> list[int]:
        return [
            i for i in range(self.total_nodes)
            if self.rack_for_node(i) == rack_id
        ]


class FailureDomain:
    """Tracks failure domains and enforces isolation constraints.

    A failure domain is a rack — if one rack goes down, requests
    should be rerouted to other racks without breaking determinism.
    """

    def __init__(self, topology: RackTopology):
        self.topology = topology
        self._failed_racks: set[int] = set()

    def mark_rack_failed(self, rack_id: int) -> None:
        self._failed_racks.add(rack_id)

    def mark_rack_healthy(self, rack_id: int) -> None:
        self._failed_racks.discard(rack_id)

    @property
    def healthy_racks(self) -> list[int]:
        return [r for r in range(self.topology.rack_count) if r not in self._failed_racks]

    @property
    def failed_racks(self) -> list[int]:
        return sorted(self._failed_racks)

    def can_serve(self) -> bool:
        return len(self.healthy_racks) > 0


class DeterministicRetry:
    """Deterministic retry semantics that preserve request ordering.

    When a replica fails, the request is retried on the next replica
    in the deterministic sequence — never randomly. This preserves
    the property that the same failure pattern produces the same
    retry routing.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    def retry_targets(
        self,
        failed_replica_index: int,
        total_replicas: int,
        failed_indices: set[int],
    ) -> list[int]:
        """Return ordered list of retry targets (deterministic)."""
        targets = []
        for offset in range(1, total_replicas):
            candidate = (failed_replica_index + offset) % total_replicas
            if candidate not in failed_indices:
                targets.append(candidate)
            if len(targets) >= self.max_retries:
                break
        return targets


class CapacityTracker:
    """Track per-rack capacity for artifact mirrors, queue isolation, and storage."""

    def __init__(self, topology: RackTopology):
        self.topology = topology
        self._rack_queues: dict[int, int] = {r: 0 for r in range(topology.rack_count)}
        self._rack_bundle_count: dict[int, int] = {r: 0 for r in range(topology.rack_count)}

    def enqueue(self, rack_id: int) -> None:
        self._rack_queues[rack_id] = self._rack_queues.get(rack_id, 0) + 1

    def dequeue(self, rack_id: int) -> None:
        self._rack_queues[rack_id] = max(0, self._rack_queues.get(rack_id, 0) - 1)

    def record_bundle(self, rack_id: int) -> None:
        self._rack_bundle_count[rack_id] = self._rack_bundle_count.get(rack_id, 0) + 1

    @property
    def queue_depths(self) -> dict[int, int]:
        return dict(self._rack_queues)

    @property
    def bundle_counts(self) -> dict[int, int]:
        return dict(self._rack_bundle_count)

    def max_queue_depth(self) -> int:
        return max(self._rack_queues.values()) if self._rack_queues else 0


class DeterminismSLO:
    """Track determinism SLO metrics for multi-rack deployments."""

    def __init__(self):
        self._total_verifications = 0
        self._conformant_count = 0
        self._non_conformant_count = 0
        self._mismatch_count = 0

    def record(self, status: str) -> None:
        self._total_verifications += 1
        if status == "conformant":
            self._conformant_count += 1
        elif status.startswith("non_conformant"):
            self._non_conformant_count += 1
        elif status == "mismatch_outputs":
            self._mismatch_count += 1

    @property
    def conformance_rate(self) -> float:
        if self._total_verifications == 0:
            return 1.0
        return self._conformant_count / self._total_verifications

    @property
    def mismatch_rate(self) -> float:
        if self._total_verifications == 0:
            return 0.0
        return self._mismatch_count / self._total_verifications

    def report(self) -> dict[str, Any]:
        return {
            "total_verifications": self._total_verifications,
            "conformant": self._conformant_count,
            "non_conformant": self._non_conformant_count,
            "mismatch_outputs": self._mismatch_count,
            "conformance_rate": round(self.conformance_rate, 6),
            "mismatch_rate": round(self.mismatch_rate, 6),
        }

    def meets_slo(self, min_conformance: float = 0.999) -> bool:
        return self.conformance_rate >= min_conformance and self.mismatch_rate == 0.0


def generate_compliance_export(
    slo: DeterminismSLO,
    topology: RackTopology,
    capacity: CapacityTracker,
    failure_domains: FailureDomain,
) -> dict[str, Any]:
    """Generate an audit-ready compliance report for external review."""
    return {
        "report_type": "determinism_compliance",
        "topology": {
            "rack_count": topology.rack_count,
            "nodes_per_rack": topology.nodes_per_rack,
            "total_nodes": topology.total_nodes,
        },
        "slo": slo.report(),
        "slo_met": slo.meets_slo(),
        "capacity": {
            "queue_depths": capacity.queue_depths,
            "bundle_counts": capacity.bundle_counts,
            "max_queue_depth": capacity.max_queue_depth(),
        },
        "failure_domains": {
            "healthy_racks": failure_domains.healthy_racks,
            "failed_racks": failure_domains.failed_racks,
            "can_serve": failure_domains.can_serve(),
        },
    }
