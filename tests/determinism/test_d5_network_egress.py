from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, run_cmd


class TestD5NetworkEgress(unittest.TestCase):
    def test_network_egress_is_reproducible(self) -> None:
        manifest = "tests/fixtures/positive/manifest.v1.example.json"
        with tempfile.TemporaryDirectory() as td:
            tdir = Path(td)
            lock_resolved = tdir / "resolved.lock.json"
            lock_built = tdir / "built.lock.json"
            run1 = tdir / "run1"
            run2 = tdir / "run2"

            run_cmd(["python3", "cmd/resolver/main.py", "--manifest", manifest, "--lockfile-out", str(lock_resolved)])
            run_cmd(["python3", "cmd/builder/main.py", "--lockfile", str(lock_resolved), "--lockfile-out", str(lock_built)])
            run_cmd(["python3", "cmd/runner/main.py", "--manifest", manifest, "--lockfile", str(lock_built), "--out-dir", str(run1)])
            run_cmd(["python3", "cmd/runner/main.py", "--manifest", manifest, "--lockfile", str(lock_built), "--out-dir", str(run2)])

            net1 = read_json(run1 / "observables/network_egress.json")
            net2 = read_json(run2 / "observables/network_egress.json")
            self.assertEqual(net1, net2)


if __name__ == "__main__":
    unittest.main()
