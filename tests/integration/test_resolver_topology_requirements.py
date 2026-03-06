from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from tests.helpers import read_json, write_json

_RESOLVER_MODULE_PATH = Path(__file__).resolve().parents[2] / "cmd" / "resolver" / "main.py"
_RESOLVER_SPEC = importlib.util.spec_from_file_location("resolver_main_topology", _RESOLVER_MODULE_PATH)
if _RESOLVER_SPEC is None or _RESOLVER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load resolver module from {_RESOLVER_MODULE_PATH}")
resolver_main = importlib.util.module_from_spec(_RESOLVER_SPEC)
_RESOLVER_SPEC.loader.exec_module(resolver_main)
resolve_manifest_to_lockfile = resolver_main.resolve_manifest_to_lockfile
ValidationError = resolver_main.ValidationError


class TestResolverTopologyRequirements(unittest.TestCase):
    def test_tensor_parallel_manifest_requires_collective_stack_artifact(self) -> None:
        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        manifest["run_id"] = "run-tp-missing-collective"
        manifest["hardware_profile"]["topology"]["mode"] = "tensor_parallel"
        manifest["hardware_profile"]["topology"]["node_count"] = 4
        manifest["hardware_profile"]["topology"]["rack_count"] = 2
        manifest["hardware_profile"]["topology"]["collective_fabric"] = "cross_rack"

        with self.assertRaises(ValidationError) as ctx:
            resolve_manifest_to_lockfile(manifest)

        self.assertIn("$.artifact_inputs", str(ctx.exception))

    def test_tensor_parallel_manifest_with_collective_stack_resolves(self) -> None:
        manifest = read_json(Path("tests/fixtures/positive/manifest.v1.example.json"))
        manifest["run_id"] = "run-tp-with-collective"
        manifest["hardware_profile"]["topology"]["mode"] = "tensor_parallel"
        manifest["hardware_profile"]["topology"]["node_count"] = 4
        manifest["hardware_profile"]["topology"]["rack_count"] = 2
        manifest["hardware_profile"]["topology"]["collective_fabric"] = "cross_rack"
        manifest["artifact_inputs"].append(
            {
                "artifact_id": "collective-stack",
                "artifact_type": "collective_stack",
                "expected_digest": "sha256:" + ("c" * 64),
                "immutable_ref": "sha256:" + ("d" * 64),
                "name": "nccl-stack",
                "size_bytes": 512,
                "source_kind": "oci",
                "source_uri": "oci://registry.example/nccl@sha256:" + ("d" * 64),
            }
        )

        with tempfile.TemporaryDirectory() as td:
            manifest_path = Path(td) / "manifest.tp.json"
            write_json(manifest_path, manifest)
            lockfile = resolve_manifest_to_lockfile(read_json(manifest_path))

        self.assertTrue(any(item["artifact_type"] == "collective_stack" for item in lockfile["artifacts"]))


if __name__ == "__main__":
    unittest.main()
