from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd, write_json


class TestMultiRackTopology(unittest.TestCase):
    def test_multi_rack_dispatch_is_stable(self) -> None:
        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        manifest["run_id"] = "run-multi-rack"
        manifest["hardware_profile"]["topology"]["mode"] = "replicated"
        manifest["hardware_profile"]["topology"]["node_count"] = 8
        manifest["hardware_profile"]["topology"]["rack_count"] = 4
        manifest["hardware_profile"]["topology"]["collective_fabric"] = "cross_rack"

        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            manifest_path = tdir / "manifest.multi_rack.json"
            dispatch1 = tdir / "dispatch1.json"
            dispatch2 = tdir / "dispatch2.json"
            write_json(manifest_path, manifest)

            replicas = ",".join(f"replica-{idx}" for idx in range(8))
            run_cmd(["python3", "cmd/runner/dispatcher.py", "--manifest", str(manifest_path), "--replicas", replicas, "--out", str(dispatch1)])
            run_cmd(["python3", "cmd/runner/dispatcher.py", "--manifest", str(manifest_path), "--replicas", replicas, "--out", str(dispatch2)])

            self.assertEqual(dispatch1.read_bytes(), dispatch2.read_bytes())
            assignments = read_json(dispatch1)
            rack_ids = {entry["rack_id"] for entry in assignments}
            self.assertTrue(all(0 <= rack < 4 for rack in rack_ids))


if __name__ == "__main__":
    unittest.main()
