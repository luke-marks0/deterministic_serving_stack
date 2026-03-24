"""Unit tests for manifest validation and enforcement (no GPU needed)."""
from __future__ import annotations

import copy
import json
import pathlib
import sys
import unittest

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
        del m["model"]["resolved_revision"]
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


if __name__ == "__main__":
    unittest.main()
