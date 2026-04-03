"""Minimal deterministic HTTP server for warden integration tests.

Returns fixed JSON responses with fixed headers. Every response is
byte-identical for the same path, enabling frame-level determinism
verification.
"""
from __future__ import annotations

import http.server
import threading
from typing import ClassVar


RESPONSE_DETERMINISTIC = b'{"model":"test-model","tokens":[1,2,3,4,5],"request_id":"fixed-request-001"}'
RESPONSE_ALT = b'{"model":"test-model","tokens":[9,8,7],"request_id":"fixed-request-002"}'


class DeterministicHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler that returns fixed, deterministic responses."""

    RESPONSES: ClassVar[dict[str, bytes]] = {
        "/deterministic": RESPONSE_DETERMINISTIC,
        "/deterministic/alt": RESPONSE_ALT,
    }

    def version_string(self) -> str:
        return "TestServer/1.0"

    def date_time_string(self, timestamp=None) -> str:
        return "Thu, 01 Jan 2026 00:00:00 GMT"

    def log_message(self, format, *args) -> None:
        pass  # Suppress request logging.

    def do_GET(self) -> None:
        body = self.RESPONSES.get(self.path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def start_server(port: int = 18080) -> tuple[http.server.HTTPServer, threading.Thread]:
    """Start the deterministic HTTP server in a background thread."""
    server = http.server.HTTPServer(("0.0.0.0", port), DeterministicHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread
