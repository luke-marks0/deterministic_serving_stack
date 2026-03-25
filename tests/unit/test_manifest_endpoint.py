"""Unit tests for manifest validation and enforcement (no GPU needed)."""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import unittest
from unittest import mock

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.common.contracts import ValidationError, validate_with_schema

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "server_main", REPO_ROOT / "cmd" / "server" / "main.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_enforce_model_revision = _mod._enforce_model_revision
_validate_requests = _mod._validate_requests
_build_vllm_cmd = _mod._build_vllm_cmd
_set_deterministic_env = _mod._set_deterministic_env
_verify_container_image = _mod._verify_container_image
_enforce_hardware = _mod._enforce_hardware


def _load_manifest() -> dict:
    path = REPO_ROOT / "manifests" / "qwen3-1.7b.manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


class TestSchemaValidation(unittest.TestCase):
    """Schema is the first gate for POST /manifest."""

    def test_valid_manifest_passes(self) -> None:
        validate_with_schema("manifest.v1.schema.json", _load_manifest())

    def test_missing_run_id_rejected(self) -> None:
        m = _load_manifest()
        del m["run_id"]
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_bad_model_source_rejected(self) -> None:
        m = _load_manifest()
        m["model"]["source"] = "not-a-valid-source"
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_empty_dict_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", {})

    def test_missing_model_rejected(self) -> None:
        m = _load_manifest()
        del m["model"]
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_missing_runtime_rejected(self) -> None:
        m = _load_manifest()
        del m["runtime"]
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)

    def test_bad_temperature_rejected(self) -> None:
        m = _load_manifest()
        m["requests"][0]["temperature"] = 5.0
        with self.assertRaises(ValidationError):
            validate_with_schema("manifest.v1.schema.json", m)


class TestModelRevisionEnforcement(unittest.TestCase):
    """Model revision pinning."""

    def test_pinned_revision_returned(self) -> None:
        m = _load_manifest()
        rev = _enforce_model_revision(m)
        self.assertIsNotNone(rev)
        self.assertEqual(len(rev), 40)  # sha1 hex

    def test_missing_revision_returns_none(self) -> None:
        m = _load_manifest()
        del m["model"]["weights_revision"]
        rev = _enforce_model_revision(m)
        self.assertIsNone(rev)


class TestRequestValidation(unittest.TestCase):
    """Requests must be servable with the declared engine config."""

    def test_valid_requests_pass(self) -> None:
        m = _load_manifest()
        errors = _validate_requests(m)
        self.assertEqual(errors, [])

    def test_request_exceeding_max_model_len(self) -> None:
        m = _load_manifest()
        m["requests"] = [
            {"id": "too-long", "prompt": "hi", "max_new_tokens": 999999, "temperature": 0}
        ]
        errors = _validate_requests(m)
        self.assertEqual(len(errors), 1)
        self.assertIn("exceeds max_model_len", errors[0])

    def test_multiple_invalid_requests(self) -> None:
        m = _load_manifest()
        max_len = m["runtime"]["serving_engine"]["max_model_len"]
        m["requests"] = [
            {"id": "a", "prompt": "hi", "max_new_tokens": max_len + 1, "temperature": 0},
            {"id": "b", "prompt": "hi", "max_new_tokens": max_len + 100, "temperature": 0},
        ]
        errors = _validate_requests(m)
        self.assertEqual(len(errors), 2)

    def test_requests_at_exactly_max_len(self) -> None:
        m = _load_manifest()
        max_len = m["runtime"]["serving_engine"]["max_model_len"]
        m["requests"] = [
            {"id": "exact", "prompt": "hi", "max_new_tokens": max_len, "temperature": 0},
        ]
        errors = _validate_requests(m)
        self.assertEqual(errors, [])


class TestBuildVllmCmd(unittest.TestCase):
    """Test that _build_vllm_cmd passes every serving_engine field to vLLM."""

    def test_quantization_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["quantization"] = "awq"
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--quantization", cmd)
        self.assertEqual(cmd[cmd.index("--quantization") + 1], "awq")

    def test_quantization_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--quantization", cmd)

    def test_load_format_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["load_format"] = "safetensors"
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--load-format", cmd)
        self.assertEqual(cmd[cmd.index("--load-format") + 1], "safetensors")

    def test_load_format_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--load-format", cmd)

    def test_kv_cache_dtype_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["kv_cache_dtype"] = "fp8"
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--kv-cache-dtype", cmd)
        self.assertEqual(cmd[cmd.index("--kv-cache-dtype") + 1], "fp8")

    def test_kv_cache_dtype_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--kv-cache-dtype", cmd)

    def test_max_num_batched_tokens_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["max_num_batched_tokens"] = 4096
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--max-num-batched-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--max-num-batched-tokens") + 1], "4096")

    def test_max_num_batched_tokens_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--max-num-batched-tokens", cmd)

    def test_block_size_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["block_size"] = 16
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--block-size", cmd)
        self.assertEqual(cmd[cmd.index("--block-size") + 1], "16")

    def test_block_size_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--block-size", cmd)

    def test_enable_prefix_caching_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["enable_prefix_caching"] = True
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--enable-prefix-caching", cmd)

    def test_enable_prefix_caching_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--enable-prefix-caching", cmd)

    def test_enable_chunked_prefill_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["enable_chunked_prefill"] = True
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--enable-chunked-prefill", cmd)

    def test_enable_chunked_prefill_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--enable-chunked-prefill", cmd)

    def test_scheduling_policy_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["scheduling_policy"] = "fcfs"
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--scheduling-policy", cmd)
        self.assertEqual(cmd[cmd.index("--scheduling-policy") + 1], "fcfs")

    def test_scheduling_policy_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--scheduling-policy", cmd)

    def test_disable_sliding_window_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["disable_sliding_window"] = True
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--disable-sliding-window", cmd)

    def test_disable_sliding_window_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--disable-sliding-window", cmd)

    def test_tensor_parallel_size_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["tensor_parallel_size"] = 4
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--tensor-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--tensor-parallel-size") + 1], "4")

    def test_tensor_parallel_size_one_omitted(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["tensor_parallel_size"] = 1
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--tensor-parallel-size", cmd)

    def test_tensor_parallel_size_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--tensor-parallel-size", cmd)

    def test_pipeline_parallel_size_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["pipeline_parallel_size"] = 2
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--pipeline-parallel-size", cmd)
        self.assertEqual(cmd[cmd.index("--pipeline-parallel-size") + 1], "2")

    def test_pipeline_parallel_size_one_omitted(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["pipeline_parallel_size"] = 1
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--pipeline-parallel-size", cmd)

    def test_pipeline_parallel_size_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--pipeline-parallel-size", cmd)

    def test_disable_custom_all_reduce_flag_present(self) -> None:
        m = _load_manifest()
        m["runtime"]["serving_engine"]["disable_custom_all_reduce"] = True
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertIn("--disable-custom-all-reduce", cmd)

    def test_disable_custom_all_reduce_flag_absent(self) -> None:
        m = _load_manifest()
        cmd = _build_vllm_cmd(m, "127.0.0.1", 8001)
        self.assertNotIn("--disable-custom-all-reduce", cmd)


class TestSetDeterministicEnv(unittest.TestCase):
    """Test that _set_deterministic_env reads knobs from manifest."""

    def test_cublas_workspace_config_from_manifest(self) -> None:
        m = _load_manifest()
        m["runtime"]["deterministic_knobs"]["cublas_workspace_config"] = ":16:8"
        _set_deterministic_env(m)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":16:8")

    def test_cublas_workspace_config_default(self) -> None:
        m = _load_manifest()
        m["runtime"]["deterministic_knobs"].pop("cublas_workspace_config", None)
        _set_deterministic_env(m)
        self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_pythonhashseed_from_manifest(self) -> None:
        m = _load_manifest()
        m["runtime"]["deterministic_knobs"]["pythonhashseed"] = "12345"
        _set_deterministic_env(m)
        self.assertEqual(os.environ["PYTHONHASHSEED"], "12345")

    def test_pythonhashseed_default(self) -> None:
        m = _load_manifest()
        m["runtime"]["deterministic_knobs"].pop("pythonhashseed", None)
        _set_deterministic_env(m)
        self.assertEqual(os.environ["PYTHONHASHSEED"], "0")


class TestContainerImageDigest(unittest.TestCase):
    """Container image digest verification via _verify_container_image."""

    def test_digest_mismatch_raises(self) -> None:
        m = _load_manifest()
        m["runtime"]["container_image_digest"] = "sha256:" + "a" * 64
        report = {"enforced": [], "warnings": []}
        old_val = os.environ.get("CONTAINER_IMAGE_DIGEST")
        try:
            os.environ["CONTAINER_IMAGE_DIGEST"] = "sha256:" + "b" * 64
            with self.assertRaises(ValidationError) as ctx:
                _verify_container_image(m, report)
            self.assertIn("Container image digest mismatch", str(ctx.exception))
        finally:
            if old_val is None:
                os.environ.pop("CONTAINER_IMAGE_DIGEST", None)
            else:
                os.environ["CONTAINER_IMAGE_DIGEST"] = old_val

    def test_digest_match_passes(self) -> None:
        m = _load_manifest()
        digest = "sha256:" + "a" * 64
        m["runtime"]["container_image_digest"] = digest
        report = {"enforced": [], "warnings": []}
        old_val = os.environ.get("CONTAINER_IMAGE_DIGEST")
        try:
            os.environ["CONTAINER_IMAGE_DIGEST"] = digest
            _verify_container_image(m, report)
            self.assertTrue(any("container image digest verified" in e for e in report["enforced"]))
        finally:
            if old_val is None:
                os.environ.pop("CONTAINER_IMAGE_DIGEST", None)
            else:
                os.environ["CONTAINER_IMAGE_DIGEST"] = old_val

    def test_no_digest_in_manifest_skips(self) -> None:
        m = _load_manifest()
        m["runtime"].pop("container_image_digest", None)
        report = {"enforced": [], "warnings": []}
        _verify_container_image(m, report)
        self.assertEqual(report["enforced"], [])
        self.assertEqual(report["warnings"], [])

    def test_env_var_missing_warns(self) -> None:
        m = _load_manifest()
        m["runtime"]["container_image_digest"] = "sha256:" + "a" * 64
        report = {"enforced": [], "warnings": []}
        old_val = os.environ.get("CONTAINER_IMAGE_DIGEST")
        try:
            os.environ.pop("CONTAINER_IMAGE_DIGEST", None)
            _verify_container_image(m, report)
            self.assertTrue(any("not set" in w for w in report["warnings"]))
        finally:
            if old_val is not None:
                os.environ["CONTAINER_IMAGE_DIGEST"] = old_val


class TestDriverVersionVerification(unittest.TestCase):
    """GPU driver and CUDA version verification via _enforce_hardware."""

    def _mock_torch(self, cuda_version="12.4"):
        """Create a mock torch module with CUDA support."""
        mock_torch = mock.MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA H100 80GB HBM3"
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.version.cuda = cuda_version
        return mock_torch

    def test_driver_version_match(self) -> None:
        m = _load_manifest()
        m["hardware_profile"]["gpu"]["driver_version"] = "550.54.15"
        mock_torch = self._mock_torch()
        mock_result = mock.MagicMock()
        mock_result.stdout = "550.54.15\n"
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with mock.patch("subprocess.run", return_value=mock_result):
                warnings = _enforce_hardware(m)
        self.assertFalse(any("GPU driver mismatch" in w for w in warnings))

    def test_driver_version_mismatch(self) -> None:
        m = _load_manifest()
        m["hardware_profile"]["gpu"]["driver_version"] = "550.54.15"
        mock_torch = self._mock_torch()
        mock_result = mock.MagicMock()
        mock_result.stdout = "535.86.01\n"
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with mock.patch("subprocess.run", return_value=mock_result):
                warnings = _enforce_hardware(m)
        self.assertTrue(any("GPU driver mismatch" in w for w in warnings))

    def test_nvidia_smi_failure_warns(self) -> None:
        m = _load_manifest()
        m["hardware_profile"]["gpu"]["driver_version"] = "550.54.15"
        mock_torch = self._mock_torch()
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with mock.patch("subprocess.run", side_effect=OSError("not found")):
                warnings = _enforce_hardware(m)
        self.assertTrue(any("Could not query" in w for w in warnings))

    def test_cuda_version_match(self) -> None:
        m = _load_manifest()
        m["hardware_profile"]["gpu"]["cuda_driver_version"] = "12.4"
        mock_torch = self._mock_torch(cuda_version="12.4")
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with mock.patch("subprocess.run", return_value=mock.MagicMock(stdout="550.54.15\n")):
                warnings = _enforce_hardware(m)
        self.assertFalse(any("CUDA version mismatch" in w for w in warnings))

    def test_cuda_version_mismatch(self) -> None:
        m = _load_manifest()
        m["hardware_profile"]["gpu"]["cuda_driver_version"] = "12.4"
        mock_torch = self._mock_torch(cuda_version="11.8")
        with mock.patch.dict("sys.modules", {"torch": mock_torch}):
            with mock.patch("subprocess.run", return_value=mock.MagicMock(stdout="550.54.15\n")):
                warnings = _enforce_hardware(m)
        self.assertTrue(any("CUDA version mismatch" in w for w in warnings))


if __name__ == "__main__":
    unittest.main()
