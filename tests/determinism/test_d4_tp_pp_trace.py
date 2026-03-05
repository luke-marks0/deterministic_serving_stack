from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd, write_json


class TestD4TpPpTrace(unittest.TestCase):
    def test_collective_trace_is_recorded_and_stable(self) -> None:
        base_manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        base_manifest["run_id"] = "run-d4"
        base_manifest["hardware_profile"]["topology"]["mode"] = "tensor_parallel"
        base_manifest["hardware_profile"]["topology"]["node_count"] = 4
        base_manifest["hardware_profile"]["topology"]["rack_count"] = 2
        base_manifest["hardware_profile"]["topology"]["collective_fabric"] = "cross_rack"
        base_manifest["deterministic_dispatcher"]["enabled"] = True
        base_manifest["deterministic_dispatcher"]["algorithm"] = "sequence_map"

        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            manifest = tdir / "manifest.tp.json"
            lock_resolved = tdir / "resolved.lock.json"
            lock_built = tdir / "built.lock.json"
            run1 = tdir / "run1"
            run2 = tdir / "run2"

            write_json(manifest, base_manifest)

            run_cmd(["python3", "cmd/resolver/main.py", "--manifest", str(manifest), "--lockfile-out", str(lock_resolved)])
            run_cmd(["python3", "cmd/builder/main.py", "--lockfile", str(lock_resolved), "--lockfile-out", str(lock_built)])
            run_cmd(["python3", "cmd/runner/main.py", "--manifest", str(manifest), "--lockfile", str(lock_built), "--out-dir", str(run1)])
            run_cmd(["python3", "cmd/runner/main.py", "--manifest", str(manifest), "--lockfile", str(lock_built), "--out-dir", str(run2)])

            trace1 = read_json(run1 / "observables/engine_trace.json")
            trace2 = read_json(run2 / "observables/engine_trace.json")
            self.assertEqual(trace1, trace2)
            self.assertTrue(any(evt.get("event") == "collective_algorithm_selection" for evt in trace1))


if __name__ == "__main__":
    unittest.main()
