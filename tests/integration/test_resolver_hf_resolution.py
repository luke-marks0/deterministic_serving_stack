from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pkg.common.deterministic import canonical_json_bytes
from pkg.common.hf_resolution import resolve_hf_model

_RESOLVER_MODULE_PATH = Path(__file__).resolve().parents[2] / "cmd" / "resolver" / "main.py"
_RESOLVER_SPEC = importlib.util.spec_from_file_location("resolver_main", _RESOLVER_MODULE_PATH)
if _RESOLVER_SPEC is None or _RESOLVER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load resolver module from {_RESOLVER_MODULE_PATH}")
resolver_main = importlib.util.module_from_spec(_RESOLVER_SPEC)
_RESOLVER_SPEC.loader.exec_module(resolver_main)
resolve_manifest_to_lockfile = resolver_main.resolve_manifest_to_lockfile


class FakeHFClient:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.commit = "1234567890abcdef1234567890abcdef12345678"

    def resolve_commit(self, repo_id: str, revision: str) -> str:
        return self.commit

    def list_files(self, repo_id: str, revision: str) -> list[str]:
        return sorted(
            [
                "config.json",
                "generation_config.json",
                "tokenizer.json",
                "chat_template.jinja",
                "prompt_formatter.py",
                "model-00001.safetensors",
                "model-00002.safetensors",
                "remote_logic.py",
            ]
        )

    def download_file(self, repo_id: str, revision: str, file_path: str, cache_dir: Path | None) -> Path:
        return self.root / file_path


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class TestHFResolution(unittest.TestCase):
    def _prepare_files(self, root: Path) -> None:
        files = {
            "config.json": b"{}\n",
            "generation_config.json": b"{\"max_new_tokens\": 8}\n",
            "tokenizer.json": b"{\"tokenizer\": true}\n",
            "chat_template.jinja": b"{{ messages }}\n",
            "prompt_formatter.py": b"def format_prompt(x):\n    return x\n",
            "model-00001.safetensors": b"WEIGHTS1",
            "model-00002.safetensors": b"WEIGHTS2",
            "remote_logic.py": b"print('remote')\n",
        }
        for name, content in files.items():
            (root / name).write_bytes(content)

    def test_resolve_hf_model_outputs_required_files_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._prepare_files(root)
            client = FakeHFClient(root)

            model = {
                "source": "hf://org/model",
                "requested_revision": "main",
                "resolved_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "tokenizer_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "weights_revision": "cccccccccccccccccccccccccccccccccccccccc",
                "trust_remote_code": False,
                "required_files": [],
            }
            resolved = resolve_hf_model(model, False, client=client, cache_dir=root)

            self.assertEqual(resolved.resolved_revision, client.commit)
            roles = [item["role"] for item in resolved.required_files]
            self.assertIn("weights_shard", roles)
            self.assertIn("config", roles)
            self.assertIn("tokenizer", roles)
            self.assertIn("generation_config", roles)
            self.assertIn("chat_template", roles)
            self.assertIn("prompt_formatter", roles)
            self.assertTrue(all(item["digest"].startswith("sha256:") for item in resolved.required_files))
            self.assertTrue(any(item["artifact_type"] == "model_weights" for item in resolved.model_artifacts))

    def test_resolve_hf_model_with_remote_code(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._prepare_files(root)
            client = FakeHFClient(root)

            model = {
                "source": "hf://org/model",
                "requested_revision": "main",
                "resolved_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "tokenizer_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "weights_revision": "cccccccccccccccccccccccccccccccccccccccc",
                "trust_remote_code": True,
                "required_files": [],
            }
            resolved = resolve_hf_model(model, True, client=client, cache_dir=root)

            self.assertIsNotNone(resolved.remote_code)
            self.assertEqual(resolved.remote_code["commit"], client.commit)
            self.assertTrue(any(item["artifact_type"] == "remote_code" for item in resolved.model_artifacts))

    def test_resolver_merges_hf_artifacts_into_lockfile(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._prepare_files(root)

            fake_client = FakeHFClient(root)

            manifest = {
                "manifest_version": "v1",
                "run_id": "run-hf-resolve",
                "created_at": "2026-03-05T00:00:00Z",
                "model": {
                    "source": "hf://org/model",
                    "requested_revision": "main",
                    "resolved_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    "tokenizer_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                    "weights_revision": "cccccccccccccccccccccccccccccccccccccccc",
                    "trust_remote_code": False,
                    "required_files": [
                        {
                            "role": "weights_shard",
                            "path": "placeholder.safetensors",
                            "uri": "hf://org/model/placeholder.safetensors",
                            "digest": "sha256:" + ("1" * 64),
                            "size_bytes": 1,
                        },
                        {
                            "role": "config",
                            "path": "placeholder.config",
                            "uri": "hf://org/model/placeholder.config",
                            "digest": "sha256:" + ("2" * 64),
                            "size_bytes": 1,
                        },
                        {
                            "role": "tokenizer",
                            "path": "placeholder.tokenizer",
                            "uri": "hf://org/model/placeholder.tokenizer",
                            "digest": "sha256:" + ("3" * 64),
                            "size_bytes": 1,
                        },
                        {
                            "role": "generation_config",
                            "path": "placeholder.generation",
                            "uri": "hf://org/model/placeholder.generation",
                            "digest": "sha256:" + ("4" * 64),
                            "size_bytes": 1,
                        },
                        {
                            "role": "chat_template",
                            "path": "placeholder.chat",
                            "uri": "hf://org/model/placeholder.chat",
                            "digest": "sha256:" + ("5" * 64),
                            "size_bytes": 1,
                        },
                        {
                            "role": "prompt_formatter",
                            "path": "placeholder.prompt",
                            "uri": "hf://org/model/placeholder.prompt",
                            "digest": "sha256:" + ("6" * 64),
                            "size_bytes": 1,
                        }
                    ]
                },
                "runtime": {
                    "strict_hardware": True,
                    "batch_policy": "fixed",
                    "batch_cardinality": {
                        "min_requests": 1,
                        "target_batch_size": 1,
                        "max_requests": 1
                    },
                    "engine_trace": {
                        "enabled": True,
                        "events": [
                            "batch_composition"
                        ]
                    },
                    "deterministic_knobs": {
                        "seed": 42,
                        "torch_deterministic": True,
                        "cuda_launch_blocking": True
                    },
                    "allow_non_reproducible_egress": False
                },
                "hardware_profile": {
                    "gpu": {
                        "vendor": "nvidia",
                        "model": "H100-SXM-80GB",
                        "count": 1,
                        "driver_version": "550.54.15",
                        "cuda_driver_version": "12.4",
                        "pci_ids": [
                            "0000:65:00.0"
                        ]
                    },
                    "nic": {
                        "model": "ConnectX-7",
                        "pci_id": "0000:17:00.0",
                        "firmware": "28.40.1000",
                        "link_speed_gbps": 200,
                        "offloads": {
                            "tso": False,
                            "gso": False,
                            "checksum": False,
                            "vlan_strip": False
                        }
                    },
                    "topology": {
                        "mode": "single_node",
                        "node_count": 1,
                        "rack_count": 1,
                        "collective_fabric": "none"
                    }
                },
                "network": {
                    "egress_reproducibility": True,
                    "security_mode": "plaintext",
                    "mtu": 1500,
                    "mss": 1460,
                    "tso": False,
                    "gso": False,
                    "checksum_offload": False,
                    "queue_mapping": {
                        "tx_queues": 1,
                        "rx_queues": 1,
                        "mapping_policy": "fixed_core_queue"
                    },
                    "ring_sizes": {
                        "tx": 256,
                        "rx": 256
                    },
                    "thread_affinity": [
                        1
                    ],
                    "internal_batching": {
                        "enabled": False,
                        "max_burst": 1
                    }
                },
                "requests": [
                    {
                        "id": "r1",
                        "prompt": "hello",
                        "max_new_tokens": 8,
                        "temperature": 0
                    }
                ],
                "comparison": {
                    "tokens": {
                        "mode": "exact"
                    },
                    "logits": {
                        "mode": "exact"
                    },
                    "activations": {
                        "mode": "exact"
                    },
                    "network_egress": {
                        "mode": "hash",
                        "algorithm": "sha256"
                    }
                },
                "deterministic_dispatcher": {
                    "enabled": False,
                    "algorithm": "round_robin_hash",
                    "request_order_source": "ingress_sequence",
                    "replay_log_required": True
                },
                "artifact_inputs": [
                    {
                        "artifact_id": "serving-stack",
                        "artifact_type": "serving_stack",
                        "name": "vllm",
                        "source_kind": "oci",
                        "source_uri": "oci://registry.example/vllm",
                        "immutable_ref": "sha256:" + ("a" * 64),
                        "expected_digest": "sha256:" + ("a" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "cuda-lib",
                        "artifact_type": "cuda_lib",
                        "name": "cuda",
                        "source_kind": "oci",
                        "source_uri": "oci://registry.example/cuda",
                        "immutable_ref": "sha256:" + ("b" * 64),
                        "expected_digest": "sha256:" + ("b" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "kernel-lib",
                        "artifact_type": "kernel_library",
                        "name": "kernel",
                        "source_kind": "s3",
                        "source_uri": "s3://mirror/kernel",
                        "immutable_ref": "sha256:" + ("c" * 64),
                        "expected_digest": "sha256:" + ("c" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "network-stack",
                        "artifact_type": "network_stack_binary",
                        "name": "net",
                        "source_kind": "s3",
                        "source_uri": "s3://mirror/net",
                        "immutable_ref": "sha256:" + ("d" * 64),
                        "expected_digest": "sha256:" + ("d" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "pmd-driver",
                        "artifact_type": "pmd_driver",
                        "name": "pmd",
                        "source_kind": "s3",
                        "source_uri": "s3://mirror/pmd",
                        "immutable_ref": "sha256:" + ("e" * 64),
                        "expected_digest": "sha256:" + ("e" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "runtime-knobs",
                        "artifact_type": "runtime_knob_set",
                        "name": "knobs",
                        "source_kind": "inline",
                        "source_uri": "inline://knobs",
                        "immutable_ref": "v1",
                        "expected_digest": "sha256:" + ("f" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "request-set",
                        "artifact_type": "request_set",
                        "name": "requests",
                        "source_kind": "inline",
                        "source_uri": "inline://requests",
                        "immutable_ref": "v1",
                        "expected_digest": "sha256:" + ("1" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "batching-policy",
                        "artifact_type": "batching_policy",
                        "name": "batching",
                        "source_kind": "inline",
                        "source_uri": "inline://batching",
                        "immutable_ref": "v1",
                        "expected_digest": "sha256:" + ("2" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "nic-link",
                        "artifact_type": "nic_link_config",
                        "name": "nic",
                        "source_kind": "inline",
                        "source_uri": "inline://nic",
                        "immutable_ref": "v1",
                        "expected_digest": "sha256:" + ("3" * 64),
                        "size_bytes": 100
                    },
                    {
                        "artifact_id": "compiled-ext",
                        "artifact_type": "compiled_extension",
                        "name": "compiled",
                        "source_kind": "s3",
                        "source_uri": "s3://mirror/compiled",
                        "immutable_ref": "sha256:" + ("4" * 64),
                        "expected_digest": "sha256:" + ("4" * 64),
                        "size_bytes": 100
                    }
                ]
            }

            with mock.patch.object(resolver_main, "HuggingFaceHubClient", return_value=fake_client):
                lockfile = resolve_manifest_to_lockfile(
                    manifest,
                    resolve_hf=True,
                    hf_cache_dir=root,
                    hf_token=None,
                )

            self.assertEqual(manifest["model"]["resolved_revision"], fake_client.commit)
            self.assertGreaterEqual(len(manifest["model"]["required_files"]), 6)
            self.assertTrue(
                any(
                    item["artifact_type"] == "model_weights"
                    for item in lockfile["artifacts"]
                )
            )
            expected_manifest_digest = "sha256:" + hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
            self.assertEqual(lockfile["manifest_digest"], expected_manifest_digest)


if __name__ == "__main__":
    unittest.main()
