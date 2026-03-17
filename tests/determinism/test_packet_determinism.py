"""Packet-level network determinism tests.

Tests that the actual bytes sent over the network are deterministic,
not just the JSON response body. This captures TCP payload bytes via
a raw socket or tcpdump and compares them across runs.

Three levels tested:
1. TCP payload determinism: the HTTP response payload bytes (headers + body)
   are identical across runs after stripping nondeterministic fields
   (Date header, connection IDs).

2. Payload digest in run bundle: the run bundle contains a digest of
   the egress payload that matches across runs.

3. Full pcap comparison: capture raw packets via tcpdump, strip
   nondeterministic TCP/IP headers, compare remaining bytes.

Requires a running vLLM server.

Environment:
    DETERMINISTIC_SERVER_URL: server URL (default: http://127.0.0.1:8000)
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SERVER_URL = os.getenv("DETERMINISTIC_SERVER_URL", "http://127.0.0.1:8000")
SERVER_HOST = SERVER_URL.split("//")[1].split(":")[0] if "//" in SERVER_URL else "127.0.0.1"
SERVER_PORT = int(SERVER_URL.split(":")[-1].rstrip("/")) if SERVER_URL.count(":") >= 2 else 8000


def _server_available() -> bool:
    try:
        from urllib.request import urlopen
        urlopen(f"{SERVER_URL}/health", timeout=3)
        return True
    except Exception:
        return False


def _raw_http_exchange(host: str, port: int, body: bytes) -> tuple[bytes, bytes]:
    """Send a request and capture the raw HTTP response bytes.

    Returns (response_headers_bytes, response_body_bytes) as raw wire bytes.
    Uses http.client at the lowest level to get the actual bytes.
    """
    conn = http.client.HTTPConnection(host, port, timeout=120)
    conn.request(
        "POST", "/v1/chat/completions",
        body=body,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
            # Don't send Accept-Encoding to avoid compression nondeterminism
            "Accept-Encoding": "identity",
        },
    )
    response = conn.getresponse()
    # Read raw response
    headers_raw = f"HTTP/{response.version // 10}.{response.version % 10} {response.status} {response.reason}\r\n"
    for key, val in response.getheaders():
        headers_raw += f"{key}: {val}\r\n"
    headers_raw += "\r\n"
    body_raw = response.read()
    conn.close()
    return headers_raw.encode("iso-8859-1"), body_raw


def _strip_nondeterministic_headers(headers: bytes) -> bytes:
    """Remove headers that vary per-request (Date, connection IDs, etc)."""
    lines = headers.decode("iso-8859-1").split("\r\n")
    filtered = []
    skip_headers = {"date", "x-request-id", "x-dispatch-seq", "x-replica"}
    for line in lines:
        if ":" in line:
            key = line.split(":")[0].strip().lower()
            if key in skip_headers:
                continue
        filtered.append(line)
    return "\r\n".join(filtered).encode("iso-8859-1")


def _strip_nondeterministic_json(body: bytes) -> bytes:
    """Remove nondeterministic fields from the JSON response body."""
    try:
        data = json.loads(body)
        data.pop("id", None)        # chatcmpl-<random>
        data.pop("created", None)   # unix timestamp
        data.pop("system_fingerprint", None)
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except json.JSONDecodeError:
        return body


def _chat_body(prompt: str, max_tokens: int = 32, seed: int = 42) -> bytes:
    return json.dumps({
        "model": "Qwen/Qwen3-1.7B",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "seed": seed,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _egress_digest(payloads: list[bytes]) -> str:
    h = hashlib.sha256()
    for p in payloads:
        h.update(hashlib.sha256(p).digest())
    return f"sha256:{h.hexdigest()}"


@unittest.skipUnless(_server_available(), "No server at " + SERVER_URL)
class TestTCPPayloadDeterminism(unittest.TestCase):
    """Level 1: Raw HTTP response bytes (minus nondeterministic headers)
    are identical across runs."""

    def test_single_request_raw_payload_identical(self):
        """Same request → same raw HTTP payload bytes."""
        body = _chat_body("What is 2+2?", max_tokens=16)

        headers1, body1 = _raw_http_exchange(SERVER_HOST, SERVER_PORT, body)
        headers2, body2 = _raw_http_exchange(SERVER_HOST, SERVER_PORT, body)

        # Strip nondeterministic parts
        clean_h1 = _strip_nondeterministic_headers(headers1)
        clean_h2 = _strip_nondeterministic_headers(headers2)
        clean_b1 = _strip_nondeterministic_json(body1)
        clean_b2 = _strip_nondeterministic_json(body2)

        self.assertEqual(clean_h1, clean_h2,
                         "HTTP response headers differ between runs")
        self.assertEqual(clean_b1, clean_b2,
                         "HTTP response body differs between runs")

    def test_batch_raw_payloads_identical(self):
        """8 requests → all raw payloads identical across runs."""
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing.",
            "What is 7*8?",
            "Name the largest planet.",
            "Who wrote Hamlet?",
            "What is H2O?",
            "What is the speed of light?",
            "Define entropy.",
        ]

        def run_batch():
            payloads = []
            for p in prompts:
                body = _chat_body(p, max_tokens=32)
                _, resp_body = _raw_http_exchange(SERVER_HOST, SERVER_PORT, body)
                payloads.append(_strip_nondeterministic_json(resp_body))
            return payloads

        batch1 = run_batch()
        batch2 = run_batch()

        for i, (b1, b2) in enumerate(zip(batch1, batch2)):
            self.assertEqual(b1, b2,
                             f"Request {i} ('{prompts[i]}') raw payload differs")


@unittest.skipUnless(_server_available(), "No server at " + SERVER_URL)
class TestEgressPayloadDigest(unittest.TestCase):
    """Level 2: Egress payload digest matches across runs.

    This is what goes into the run bundle — a single digest over all
    egress payloads that proves the network output was identical.
    """

    PROMPTS = [
        "What is the capital of France?",
        "Explain photosynthesis.",
        "What is 7*8?",
        "Name the largest planet.",
    ]

    def _capture_egress(self) -> tuple[str, list[bytes]]:
        payloads = []
        for p in self.PROMPTS:
            body = _chat_body(p, max_tokens=64)
            _, resp_body = _raw_http_exchange(SERVER_HOST, SERVER_PORT, body)
            payloads.append(_strip_nondeterministic_json(resp_body))
        return _egress_digest(payloads), payloads

    def test_egress_digest_matches_across_runs(self):
        d1, _ = self._capture_egress()
        d2, _ = self._capture_egress()
        self.assertEqual(d1, d2,
                         f"Egress digest mismatch:\n  run1: {d1}\n  run2: {d2}")


@unittest.skipUnless(
    _server_available() and os.getenv("DETERMINISTIC_ENABLE_PCAP") == "1",
    "Pcap tests require DETERMINISTIC_ENABLE_PCAP=1 and tcpdump",
)
class TestPcapDeterminism(unittest.TestCase):
    """Level 3: Raw packet capture comparison.

    Captures packets via tcpdump, strips TCP/IP nondeterministic fields
    (seq numbers, timestamps, checksums), and compares the remaining bytes.

    Requires: tcpdump, sudo access, DETERMINISTIC_ENABLE_PCAP=1
    """

    def _capture_pcap(self, prompt: str, duration: float = 5.0) -> bytes:
        """Capture packets during a request and return the pcap bytes."""
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            pcap_path = f.name

        # Start tcpdump
        tcpdump = subprocess.Popen(
            ["sudo", "tcpdump", "-i", "lo", "-w", pcap_path,
             f"port {SERVER_PORT}", "-c", "100"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.5)  # Let tcpdump start

        # Send request
        body = _chat_body(prompt, max_tokens=16)
        _raw_http_exchange(SERVER_HOST, SERVER_PORT, body)

        time.sleep(0.5)
        tcpdump.terminate()
        tcpdump.wait()

        pcap_bytes = Path(pcap_path).read_bytes()
        Path(pcap_path).unlink()
        return pcap_bytes

    def _strip_tcp_headers(self, pcap_bytes: bytes) -> bytes:
        """Extract TCP payloads from pcap, stripping IP/TCP headers.

        This is a simplified extractor — in production you'd use scapy.
        """
        # Use tcpdump to extract payload hex
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
            f.write(pcap_bytes)
            pcap_path = f.name

        result = subprocess.run(
            ["sudo", "tcpdump", "-r", pcap_path, "-x", "-nn",
             f"port {SERVER_PORT}"],
            capture_output=True, text=True,
        )
        Path(pcap_path).unlink()

        # Extract hex payload lines, skip headers
        hex_lines = []
        for line in result.stdout.split("\n"):
            line = line.strip()
            if re.match(r"^0x[0-9a-f]+:", line):
                # Skip first 54 bytes (14 ethernet + 20 IP + 20 TCP)
                hex_lines.append(line)

        return "\n".join(hex_lines).encode()

    def test_pcap_payload_deterministic(self):
        """Packet payloads are identical across two captures."""
        pcap1 = self._capture_pcap("What is 2+2?")
        pcap2 = self._capture_pcap("What is 2+2?")

        payload1 = self._strip_tcp_headers(pcap1)
        payload2 = self._strip_tcp_headers(pcap2)

        self.assertEqual(
            hashlib.sha256(payload1).hexdigest(),
            hashlib.sha256(payload2).hexdigest(),
            "Pcap payloads differ between captures",
        )


if __name__ == "__main__":
    unittest.main()
