#!/usr/bin/env python3
"""Verifier server (prover-verifier-demo).

Stdlib HTTP server that owns the verifier-side endpoints. Phase 3.1 lands
the skeleton + /traffic ingest; subsequent phases add scheduling,
finalize, and verdict logic.

Usage:
    python3 cmd/verifier_server/main.py \\
        --host 127.0.0.1 --port 0 --port-file /tmp/verifier.port \\
        --out-dir /tmp/verifier-demo \\
        --prover-base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.proverdet.transcript import TranscriptLog  # noqa: E402


class VerifierState:
    """Mutable state shared across verifier handler threads."""

    def __init__(
        self,
        *,
        out_dir: Path,
        prover_base_url: str,
    ) -> None:
        self.out_dir = out_dir
        self.prover_base_url = prover_base_url
        self.transcript = TranscriptLog(out_dir / "transcript.jsonl")
        self._traffic_seq = 0
        self._traffic_lock = threading.Lock()

    def next_traffic_seq(self) -> int:
        with self._traffic_lock:
            self._traffic_seq += 1
            return self._traffic_seq


class VerifierHandler(BaseHTTPRequestHandler):
    state: VerifierState | None = None

    server_version = "VerifierServer/0.1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send_json(200, {"ok": True})
        return self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path == "/traffic":
            return self._handle_post_traffic()
        return self._send_json(404, {"error": "not found"})

    def _handle_post_traffic(self) -> None:
        if self.state is None:
            return self._send_json(500, {"error": "verifier state not initialized"})
        body = self._read_body()
        seq = self.state.next_traffic_seq()
        path = self.state.out_dir / f"traffic-{seq:06d}.bin"
        path.write_bytes(body)
        # Use a relative path in the transcript so the file is reproducible
        # across machines.
        self.state.transcript.record(
            direction="received",
            endpoint="/traffic",
            payload=body,
            payload_path=path.name,
        )
        return self._send_json(200, {"received_bytes": len(body), "seq": seq})


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _write_port_file(port_file: Path, port: int) -> None:
    port_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(port_file), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    try:
        os.write(fd, f"{port}\n".encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verifier server (prover-verifier-demo)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prover-base-url", required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = VerifierState(out_dir=args.out_dir, prover_base_url=args.prover_base_url)
    VerifierHandler.state = state

    server = ThreadedHTTPServer((args.host, args.port), VerifierHandler)
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    if args.port_file:
        _write_port_file(args.port_file, bound_port)

    print(
        f"verifier: serving on {bound_host}:{bound_port} "
        f"out_dir={args.out_dir} prover_base_url={args.prover_base_url}",
        flush=True,
    )

    def shutdown(signum: int, _frame: Any) -> None:
        print(f"verifier: caught signal {signum}, shutting down", flush=True)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
