from __future__ import annotations

import unittest
from pathlib import Path

from tests.helpers import read_json


class TestTopologyModule(unittest.TestCase):
    def test_topology_fingerprint_is_stable(self) -> None:
        from pkg.hardware.topology import compute_topology_fingerprint

        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        fp1 = compute_topology_fingerprint(manifest)
        fp2 = compute_topology_fingerprint(manifest)
        self.assertEqual(fp1, fp2)
        self.assertTrue(fp1.startswith("sha256:"))

    def test_collective_stack_required_for_tp(self) -> None:
        from pkg.hardware.topology import validate_collective_stack

        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))

        # single_node should not require collective stack
        manifest["hardware_profile"]["topology"]["mode"] = "single_node"
        result = validate_collective_stack({"artifacts": []}, manifest)
        self.assertFalse(result["required"])

    def test_cross_rank_consistency_all_same(self) -> None:
        from pkg.hardware.topology import validate_cross_rank_consistency

        profiles = [{"gpu": "H100", "driver": "550"}, {"gpu": "H100", "driver": "550"}]
        result = validate_cross_rank_consistency(profiles)
        self.assertEqual(result["status"], "conformant")
        self.assertEqual(result["diffs"], [])

    def test_cross_rank_consistency_mismatch(self) -> None:
        from pkg.hardware.topology import validate_cross_rank_consistency

        profiles = [{"gpu": "H100", "driver": "550"}, {"gpu": "A100", "driver": "525"}]
        result = validate_cross_rank_consistency(profiles)
        self.assertEqual(result["status"], "non_conformant")
        self.assertEqual(len(result["diffs"]), 1)

    def test_placement_constraints_replicated(self) -> None:
        from pkg.hardware.topology import compute_placement_constraints

        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        constraints = compute_placement_constraints(manifest)
        self.assertEqual(constraints["node_selector"]["hardware.gpu.vendor"], "nvidia")

    def test_rack_placement_balance(self) -> None:
        from pkg.hardware.topology import validate_rack_placement

        assignments = [
            {"rack_id": 0}, {"rack_id": 1}, {"rack_id": 0}, {"rack_id": 1},
        ]
        result = validate_rack_placement(assignments, rack_count=2)
        self.assertTrue(result["balanced"])
        self.assertEqual(result["max_skew"], 0)


class TestRackPolicy(unittest.TestCase):
    def test_rack_topology(self) -> None:
        from pkg.hardware.rack_policy import RackTopology

        topo = RackTopology(rack_count=4, nodes_per_rack=8)
        self.assertEqual(topo.total_nodes, 32)
        self.assertEqual(topo.rack_for_node(0), 0)
        self.assertEqual(topo.rack_for_node(5), 1)
        self.assertEqual(len(topo.nodes_in_rack(0)), 8)

    def test_failure_domain(self) -> None:
        from pkg.hardware.rack_policy import FailureDomain, RackTopology

        topo = RackTopology(4, 8)
        fd = FailureDomain(topo)
        self.assertTrue(fd.can_serve())
        self.assertEqual(len(fd.healthy_racks), 4)

        fd.mark_rack_failed(2)
        self.assertEqual(len(fd.healthy_racks), 3)
        self.assertEqual(fd.failed_racks, [2])
        self.assertTrue(fd.can_serve())

        fd.mark_rack_healthy(2)
        self.assertEqual(len(fd.healthy_racks), 4)

    def test_deterministic_retry(self) -> None:
        from pkg.hardware.rack_policy import DeterministicRetry

        retry = DeterministicRetry(max_retries=2)
        targets = retry.retry_targets(
            failed_replica_index=1,
            total_replicas=4,
            failed_indices={1},
        )
        self.assertEqual(targets, [2, 3])

        # Same input always gives same output
        targets2 = retry.retry_targets(1, 4, {1})
        self.assertEqual(targets, targets2)

    def test_slo_tracking(self) -> None:
        from pkg.hardware.rack_policy import DeterminismSLO

        slo = DeterminismSLO()
        for _ in range(999):
            slo.record("conformant")
        slo.record("non_conformant_hardware")

        self.assertEqual(slo.conformance_rate, 0.999)
        self.assertTrue(slo.meets_slo(min_conformance=0.999))
        self.assertEqual(slo.mismatch_rate, 0.0)

    def test_compliance_export(self) -> None:
        from pkg.hardware.rack_policy import (
            CapacityTracker, DeterminismSLO, FailureDomain,
            RackTopology, generate_compliance_export,
        )

        topo = RackTopology(2, 4)
        slo = DeterminismSLO()
        slo.record("conformant")
        capacity = CapacityTracker(topo)
        capacity.enqueue(0)
        capacity.record_bundle(0)
        fd = FailureDomain(topo)

        report = generate_compliance_export(slo, topo, capacity, fd)
        self.assertTrue(report["slo_met"])
        self.assertTrue(report["failure_domains"]["can_serve"])
        self.assertEqual(report["topology"]["rack_count"], 2)


if __name__ == "__main__":
    unittest.main()
