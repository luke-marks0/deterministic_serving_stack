"""End-to-end audit-replay flow over HTTP, no GPU required.

Stands up `cmd/server/main.py`'s `ProxyHandler` in-process against a
stubbed vLLM that returns deterministic, prefix-stable output token IDs.

Flow under test:
1. POST /run returns a bundle with per-request `token_commitments`.
2. POST /replay for a challenged (request_id, token_position) returns a
   commitment that MATCHES the one from step 1.
3. POST /replay at a different position returns a DIFFERENT commitment
   (proves the commitment discriminates on token position, so a match
    above is meaningful).
4. POST /replay with audit disabled returns 409.

The stub's token generator depends only on `(prompt, seed)` — NOT on
`max_tokens` — so the first N tokens of a max_tokens=N request are the
same as the first N tokens of a max_tokens=M request with M≥N. Without
that property replay can never pass, which is also a useful tripwire.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_server_module():
    """Load cmd/server/main.py as a module (cmd/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "server_main", REPO_ROOT / "cmd" / "server" / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deterministic_tokens(prompt: str, seed: int, n: int) -> list[int]:
    """Prefix-stable deterministic token stream for (prompt, seed).

    `n` is only the count — the first `n` tokens here equal the first `n`
    tokens produced when called with any larger count. That's the
    property a real deterministic vLLM provides and that the audit
    replay loop depends on.
    """
    tokens: list[int] = []
    h = hashlib.sha256(f"{prompt}|{seed}".encode()).digest()
    # Walk the hash, extending when we run out of bytes — still only a
    # function of (prompt, seed).
    buf = bytearray(h)
    step = 0
    while len(tokens) < n:
        if len(buf) < 2:
            step += 1
            buf.extend(
                hashlib.sha256(f"{prompt}|{seed}|{step}".encode()).digest()
            )
        tokens.append(int.from_bytes(buf[:2], "big"))
        del buf[:2]
    return tokens


class _FakeVLLMHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-shape stub that returns token_ids."""

    def log_message(self, *_a, **_kw):  # silence test output
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        prompt = body["messages"][0]["content"]
        max_tokens = int(body["max_tokens"])
        seed = int(body.get("seed", 0))
        token_ids = _deterministic_tokens(prompt, seed, max_tokens)

        resp: dict[str, Any] = {
            "id": "chatcmpl-stub",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "stub"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"stub-{len(token_ids)}-toks",
                    },
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": max_tokens,
                "total_tokens": len(prompt.split()) + max_tokens,
            },
        }
        if body.get("return_token_ids"):
            resp["choices"][0]["token_ids"] = token_ids

        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _pick_free_port() -> int:
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _make_manifest(*, audit_enabled: bool) -> dict[str, Any]:
    """Build a minimal manifest that parses, with or without the audit block."""
    real = json.loads(
        (REPO_ROOT / "manifests" / "qwen3-1.7b.manifest.json").read_text("utf-8")
    )
    m = copy.deepcopy(real)
    m["run_id"] = "test-audit-replay-001"
    m["requests"] = [
        {"id": "req-alpha", "prompt": "alpha prompt", "max_new_tokens": 8, "temperature": 0},
        {"id": "req-beta", "prompt": "beta prompt", "max_new_tokens": 12, "temperature": 0},
    ]
    if audit_enabled:
        m["audit"] = {
            "token_commitment": {
                "enabled": True,
                "algorithm": "hmac-sha256",
                "key_source": "inline-shared",
            }
        }
    return m


class _HarnessMixin:
    server_mod = None
    vllm_srv: _ThreadedHTTPServer | None = None
    proxy_srv: _ThreadedHTTPServer | None = None
    tmpdir: Path | None = None
    proxy_port: int = 0
    vllm_port: int = 0

    @classmethod
    def _start_harness(cls, *, audit_enabled: bool):
        cls.server_mod = _load_server_module()
        cls.tmpdir = Path(tempfile.mkdtemp(prefix="audit-replay-"))
        cls.vllm_port = _pick_free_port()
        cls.proxy_port = _pick_free_port()

        cls.vllm_srv = _ThreadedHTTPServer(("127.0.0.1", cls.vllm_port), _FakeVLLMHandler)
        threading.Thread(target=cls.vllm_srv.serve_forever, daemon=True).start()

        manifest_dict = _make_manifest(audit_enabled=audit_enabled)
        manifest = cls.server_mod.Manifest.model_validate(manifest_dict)

        capture_log = cls.server_mod.CaptureLog(cls.tmpdir / "capture.jsonl")
        state = cls.server_mod.ServerState(
            manifest=manifest,
            vllm_proc=None,
            vllm_port=cls.vllm_port,
            capture_log=capture_log,
            out_dir=cls.tmpdir,
        )
        cls.server_mod.ProxyHandler.server_state = state
        cls.server_mod.ProxyHandler.api_key = None

        cls.proxy_srv = _ThreadedHTTPServer(
            ("127.0.0.1", cls.proxy_port), cls.server_mod.ProxyHandler
        )
        threading.Thread(target=cls.proxy_srv.serve_forever, daemon=True).start()

    @classmethod
    def _stop_harness(cls):
        if cls.proxy_srv:
            cls.proxy_srv.shutdown()
            cls.proxy_srv.server_close()
        if cls.vllm_srv:
            cls.vllm_srv.shutdown()
            cls.vllm_srv.server_close()

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        req = Request(
            f"http://127.0.0.1:{self.proxy_port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=15) as resp:
                return resp.status, json.loads(resp.read())
        except Exception as exc:
            # urllib raises HTTPError subclasses for 4xx/5xx; surface body
            from urllib.error import HTTPError
            if isinstance(exc, HTTPError):
                return exc.code, json.loads(exc.read() or b"{}")
            raise


class TestAuditReplayLoop(_HarnessMixin, unittest.TestCase):
    """Full primary-run → challenge → replay round trip, audit enabled."""

    @classmethod
    def setUpClass(cls):
        cls._start_harness(audit_enabled=True)

    @classmethod
    def tearDownClass(cls):
        cls._stop_harness()

    def test_01_run_emits_commitments(self):
        status, bundle = self._post("/run", {})
        self.assertEqual(status, 200, bundle)
        self.assertIn("token_commitments", bundle)
        self.assertEqual(set(bundle["token_commitments"]), {"req-alpha", "req-beta"})
        self.assertEqual(len(bundle["token_commitments"]["req-alpha"]), 8)
        self.assertEqual(len(bundle["token_commitments"]["req-beta"]), 12)
        # Hex SHA-256 commitments
        for stream in bundle["token_commitments"].values():
            for c in stream:
                self.assertRegex(c, r"^[0-9a-f]{64}$")
        self.assertEqual(bundle["audit"]["algorithm"], "hmac-sha256")
        self.assertEqual(bundle["audit"]["key_source"], "inline-shared")
        type(self)._primary_bundle = bundle

    def test_02_replay_matches_primary_commitment(self):
        bundle = type(self)._primary_bundle
        # Challenge req-alpha position 5 (middle of 8 tokens)
        expected = bundle["token_commitments"]["req-alpha"][4]
        status, resp = self._post(
            "/replay", {"request_id": "req-alpha", "token_position": 5}
        )
        self.assertEqual(status, 200, resp)
        self.assertEqual(resp["request_id"], "req-alpha")
        self.assertEqual(resp["token_position"], 5)
        self.assertEqual(resp["commitment"], expected)

    def test_03_replay_at_full_length(self):
        bundle = type(self)._primary_bundle
        expected = bundle["token_commitments"]["req-beta"][-1]
        status, resp = self._post(
            "/replay", {"request_id": "req-beta", "token_position": 12}
        )
        self.assertEqual(status, 200, resp)
        self.assertEqual(resp["commitment"], expected)

    def test_04_replay_at_first_token(self):
        bundle = type(self)._primary_bundle
        expected = bundle["token_commitments"]["req-alpha"][0]
        status, resp = self._post(
            "/replay", {"request_id": "req-alpha", "token_position": 1}
        )
        self.assertEqual(status, 200, resp)
        self.assertEqual(resp["commitment"], expected)

    def test_05_replay_discriminates_on_position(self):
        """Asking for a different position returns a different commitment.

        Protects against a trivial "always return the same digest" bug
        that would make test_02 pass vacuously.
        """
        _, r5 = self._post("/replay", {"request_id": "req-alpha", "token_position": 5})
        _, r6 = self._post("/replay", {"request_id": "req-alpha", "token_position": 6})
        self.assertNotEqual(r5["commitment"], r6["commitment"])

    def test_06_replay_unknown_request_is_404(self):
        status, resp = self._post(
            "/replay", {"request_id": "req-missing", "token_position": 1}
        )
        self.assertEqual(status, 404)
        self.assertIn("known", resp)
        self.assertIn("req-alpha", resp["known"])

    def test_07_replay_out_of_range_is_400(self):
        status, _ = self._post(
            "/replay", {"request_id": "req-alpha", "token_position": 999}
        )
        self.assertEqual(status, 400)
        status, _ = self._post(
            "/replay", {"request_id": "req-alpha", "token_position": 0}
        )
        self.assertEqual(status, 400)

    def test_08_replay_bad_body_is_400(self):
        status, _ = self._post("/replay", {"request_id": "req-alpha"})
        self.assertEqual(status, 400)
        status, _ = self._post("/replay", {"token_position": 1})
        self.assertEqual(status, 400)


class TestReplayRequires409WhenAuditDisabled(_HarnessMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._start_harness(audit_enabled=False)

    @classmethod
    def tearDownClass(cls):
        cls._stop_harness()

    def test_run_has_no_commitments(self):
        status, bundle = self._post("/run", {})
        self.assertEqual(status, 200, bundle)
        self.assertNotIn("token_commitments", bundle)
        self.assertNotIn("audit", bundle)

    def test_replay_is_409(self):
        status, resp = self._post(
            "/replay", {"request_id": "req-alpha", "token_position": 1}
        )
        self.assertEqual(status, 409)
        self.assertIn("error", resp)


if __name__ == "__main__":
    unittest.main()
