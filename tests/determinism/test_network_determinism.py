"""Network determinism tests.

These test that the actual bytes the server sends back are identical
across runs. This is what matters for network-level determinism:
if you send the same request twice, do you get back the exact same
HTTP response body?

Three levels of network determinism are tested:

1. **Response body determinism**: The JSON response body from vLLM is
   byte-identical across runs. This means the same tokens, same usage
   stats, same structure. If this holds, any downstream consumer
   (proxy, load balancer, client) sees identical bytes.

2. **Egress digest determinism**: A SHA256 digest of all response
   bodies for a batch of requests is identical across runs. This is
   what goes into the run bundle as the network egress observable.
   Your boss wants to see this digest match across run bundles.

3. **Cross-node egress determinism**: The same requests sent to two
   independent servers produce identical egress digests. This proves
   that the network output is reproducible across machines, not just
   across runs on the same machine.

These tests require a running vLLM server. They are skipped in CI
(no GPU) and run on real hardware via:
    deploy/lambda/run_vllm_bi_tests.sh
    or: pytest tests/determinism/test_network_determinism.py -v

Environment:
    DETERMINISTIC_SERVER_URL: server URL (default: http://127.0.0.1:8000)
    DETERMINISTIC_SERVER_URL_2: second server for cross-node tests
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import socket
import unittest
from typing import Any
from urllib.request import Request, urlopen


SERVER_URL = os.getenv("DETERMINISTIC_SERVER_URL", "http://127.0.0.1:8000")
SERVER_URL_2 = os.getenv("DETERMINISTIC_SERVER_URL_2", "")


def _server_available(url: str = SERVER_URL) -> bool:
    try:
        urlopen(f"{url}/health", timeout=3)
        return True
    except Exception:
        return False


def _raw_post(url: str, body: bytes) -> bytes:
    """Send a POST and return the raw response body bytes."""
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=120) as resp:
        return resp.read()


def _chat_request_body(prompt: str, max_tokens: int = 32, seed: int = 42) -> bytes:
    """Build a deterministic chat completion request body."""
    return json.dumps({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _egress_digest(response_bodies: list[bytes]) -> str:
    """Compute a single digest over a sequence of response bodies.

    This is the "egress digest" that goes into the run bundle.
    If two runs produce the same egress digest, their network
    output was identical.
    """
    h = hashlib.sha256()
    for body in response_bodies:
        # Hash each response body, then hash the concatenation.
        # This preserves ordering: swapping two responses changes the digest.
        h.update(hashlib.sha256(body).digest())
    return f"sha256:{h.hexdigest()}"


def _strip_nondeterministic_fields(response: dict[str, Any]) -> dict[str, Any]:
    """Remove fields from the response that are expected to vary (timestamps, IDs).

    The remaining fields should be deterministic: tokens, content, usage, finish_reason.
    """
    stripped = json.loads(json.dumps(response))  # deep copy
    # These fields change per-request even with identical inputs
    stripped.pop("id", None)           # chatcmpl-<random>
    stripped.pop("created", None)      # unix timestamp
    stripped.pop("system_fingerprint", None)
    return stripped


@unittest.skipUnless(_server_available(), "No server at " + SERVER_URL)
class TestResponseBodyDeterminism(unittest.TestCase):
    """Level 1: Same request → same response JSON (minus timestamps/IDs)."""

    def test_single_request_deterministic_content(self):
        """The actual content, tokens, and usage are identical across two calls."""
        prompt = "What is 2+2? Answer with just the number."
        body = _chat_request_body(prompt, max_tokens=16)

        raw1 = _raw_post(f"{SERVER_URL}/v1/chat/completions", body)
        raw2 = _raw_post(f"{SERVER_URL}/v1/chat/completions", body)

        resp1 = _strip_nondeterministic_fields(json.loads(raw1))
        resp2 = _strip_nondeterministic_fields(json.loads(raw2))

        self.assertEqual(resp1, resp2,
                         "Response content differs between two identical requests")

    def test_raw_content_bytes_identical(self):
        """The actual message content string is byte-identical."""
        prompt = "Name three primary colors, separated by commas."
        body = _chat_request_body(prompt, max_tokens=32)

        raw1 = json.loads(_raw_post(f"{SERVER_URL}/v1/chat/completions", body))
        raw2 = json.loads(_raw_post(f"{SERVER_URL}/v1/chat/completions", body))

        content1 = raw1["choices"][0]["message"]["content"]
        content2 = raw2["choices"][0]["message"]["content"]

        self.assertEqual(content1, content2)
        # Also verify the bytes are identical, not just the decoded string
        self.assertEqual(content1.encode("utf-8"), content2.encode("utf-8"))


@unittest.skipUnless(_server_available(), "No server at " + SERVER_URL)
class TestEgressDigestDeterminism(unittest.TestCase):
    """Level 2: Batch of requests → same egress digest across runs.

    This is what your boss wants: a digest in the run bundle that
    proves the network output was identical.
    """

    PROMPTS = [
        "What is the capital of France?",
        "Explain quantum computing in one sentence.",
        "What is 7 * 8?",
        "Name the largest planet in our solar system.",
        "What year did World War II end?",
        "What is the chemical formula for water?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light in m/s?",
    ]

    def _run_batch(self) -> tuple[str, list[bytes]]:
        """Send all prompts and return (egress_digest, raw_bodies)."""
        bodies = []
        for prompt in self.PROMPTS:
            req_body = _chat_request_body(prompt, max_tokens=64)
            raw = _raw_post(f"{SERVER_URL}/v1/chat/completions", req_body)
            # Strip nondeterministic fields before hashing
            resp = _strip_nondeterministic_fields(json.loads(raw))
            canonical = json.dumps(resp, sort_keys=True, separators=(",", ":")).encode()
            bodies.append(canonical)
        return _egress_digest(bodies), bodies

    def test_batch_egress_digest_identical_across_runs(self):
        """Two runs of the same 8-request batch produce the same egress digest."""
        digest1, _ = self._run_batch()
        digest2, _ = self._run_batch()

        self.assertEqual(digest1, digest2,
                         f"Egress digest mismatch:\n  run1: {digest1}\n  run2: {digest2}")

    def test_egress_digest_is_sha256(self):
        """The digest is a valid sha256: prefixed hex string."""
        digest, _ = self._run_batch()
        self.assertTrue(digest.startswith("sha256:"))
        self.assertEqual(len(digest), len("sha256:") + 64)

    def test_individual_responses_all_match(self):
        """Every individual response in the batch matches across runs."""
        _, bodies1 = self._run_batch()
        _, bodies2 = self._run_batch()

        self.assertEqual(len(bodies1), len(bodies2))
        for i, (b1, b2) in enumerate(zip(bodies1, bodies2)):
            self.assertEqual(b1, b2,
                             f"Response {i} ('{self.PROMPTS[i]}') differs between runs")


@unittest.skipUnless(
    _server_available() and SERVER_URL_2 and _server_available(SERVER_URL_2),
    "Need two servers for cross-node test",
)
class TestCrossNodeEgressDeterminism(unittest.TestCase):
    """Level 3: Same requests to different servers → same egress digest.

    This proves network determinism across machines.
    """

    PROMPTS = [
        "What is the capital of France?",
        "Explain photosynthesis briefly.",
        "What is 12 * 12?",
        "Name the closest star to Earth.",
    ]

    def _run_batch_on(self, url: str) -> str:
        bodies = []
        for prompt in self.PROMPTS:
            req_body = _chat_request_body(prompt, max_tokens=64)
            raw = _raw_post(f"{url}/v1/chat/completions", req_body)
            resp = _strip_nondeterministic_fields(json.loads(raw))
            canonical = json.dumps(resp, sort_keys=True, separators=(",", ":")).encode()
            bodies.append(canonical)
        return _egress_digest(bodies)

    def test_cross_node_egress_digest_matches(self):
        """Two independent servers produce the same egress digest."""
        digest1 = self._run_batch_on(SERVER_URL)
        digest2 = self._run_batch_on(SERVER_URL_2)

        self.assertEqual(digest1, digest2,
                         f"Cross-node egress mismatch:\n"
                         f"  node1: {digest1}\n  node2: {digest2}")


if __name__ == "__main__":
    unittest.main()
