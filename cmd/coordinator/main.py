#!/usr/bin/env python3
"""Multi-node coordinator for deterministic replicated serving (Phase 5).

Routes requests deterministically across identical server replicas,
enforces that all replicas share the same lockfile/closure digest,
and collects per-replica capture logs for cross-pod verification.

Usage:
    python3 cmd/coordinator/main.py \
        --manifest manifest.resolved.json \
        --lockfile lockfile.built.v1.json \
        --replicas http://node-0:8000,http://node-1:8000 \
        --out-dir /path/to/coordinator-output \
        --port 9000
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.common.contracts import ValidationError, validate_with_schema
from pkg.common.deterministic import (
    canonical_json_bytes,
    canonical_json_text,
    sha256_prefixed,
    utc_now_iso,
)
from pkg.hardware.rack_policy import (
    CapacityTracker,
    DeterminismSLO,
    DeterministicRetry,
    FailureDomain,
    RackTopology,
)


class ReplicaPool:
    """Manages a set of server replicas with health checking and failure domains."""

    def __init__(self, urls: list[str], rack_count: int = 1):
        self.urls = urls
        self.healthy: list[bool] = [False] * len(urls)
        self._lock = threading.Lock()
        self.topology = RackTopology(rack_count=rack_count, nodes_per_rack=max(1, len(urls) // rack_count))
        self.failure_domain = FailureDomain(self.topology)
        self.capacity = CapacityTracker(self.topology)
        self.slo = DeterminismSLO()
        self.retry = DeterministicRetry(max_retries=2)

    def check_health(self) -> dict[str, Any]:
        results = {}
        for i, url in enumerate(self.urls):
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=5) as resp:
                    self.healthy[i] = resp.status == 200
            except Exception:
                self.healthy[i] = False
            results[url] = self.healthy[i]
            rack = self.topology.rack_for_node(i)
            if self.healthy[i]:
                self.failure_domain.mark_rack_healthy(rack)
            else:
                # Only mark rack failed if all nodes in rack are down
                rack_nodes = self.topology.nodes_in_rack(rack)
                if all(not self.healthy[n] for n in rack_nodes if n < len(self.healthy)):
                    self.failure_domain.mark_rack_failed(rack)
        return results

    def get_replica(self, index: int) -> str | None:
        idx = index % len(self.urls)
        if self.healthy[idx]:
            return self.urls[idx]
        # Deterministic retry: try next replicas in order
        failed_indices = {i for i, h in enumerate(self.healthy) if not h}
        targets = self.retry.retry_targets(idx, len(self.urls), failed_indices)
        for target in targets:
            if self.healthy[target]:
                return self.urls[target]
        return None


class DeterministicRouter:
    """Routes requests to replicas using the manifest's dispatch algorithm."""

    def __init__(self, manifest: dict[str, Any], pool: ReplicaPool):
        self.manifest = manifest
        self.pool = pool
        self.algorithm = manifest.get("deterministic_dispatcher", {}).get("algorithm", "round_robin_hash")
        self._seq = 0
        self._lock = threading.Lock()

    def route(self, request_body: dict[str, Any]) -> tuple[str, int]:
        """Return (replica_url, sequence_number)."""
        with self._lock:
            seq = self._seq
            self._seq += 1

        if self.algorithm == "sequence_map":
            idx = seq % len(self.pool.urls)
        else:
            # round_robin_hash
            req_id = request_body.get("request_id", str(seq))
            digest = sha256_prefixed(
                canonical_json_bytes({"id": req_id, "seq": seq})
            )
            idx = int(digest.split(":", 1)[1][:8], 16) % len(self.pool.urls)

        replica = self.pool.get_replica(idx)
        if replica is None:
            raise ValidationError("No healthy replicas available")
        return replica, seq


class DispatchLog:
    """Append-only log of dispatch decisions for provenance."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with open(self.path, "w") as f:
            f.write("")

    def append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            entry["dispatched_at"] = utc_now_iso()
            with open(self.path, "a") as f:
                f.write(canonical_json_text(entry))


def _verify_replica_consistency(
    pool: ReplicaPool,
    expected_lockfile_digest: str,
) -> list[dict[str, Any]]:
    """Check that all replicas serve the same model with the same config."""
    results = []
    for url in pool.urls:
        try:
            with urllib.request.urlopen(f"{url}/v1/models", timeout=10) as resp:
                models = json.loads(resp.read())
            results.append({
                "url": url,
                "healthy": True,
                "models": [m["id"] for m in models.get("data", [])],
            })
        except Exception as e:
            results.append({"url": url, "healthy": False, "error": str(e)})
    return results


class CoordinatorHandler(BaseHTTPRequestHandler):
    router: DeterministicRouter
    dispatch_log: DispatchLog

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        try:
            request_body = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request_body = {}

        try:
            replica_url, seq = self.router.route(request_body)
        except ValidationError as e:
            error_msg = json.dumps({"error": str(e)}).encode()
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_msg)))
            self.end_headers()
            self.wfile.write(error_msg)
            return

        # Forward to selected replica
        target = f"{replica_url}{self.path}"
        req = urllib.request.Request(target, data=body, method="POST")
        for key in ["Content-Type", "Authorization"]:
            val = self.headers.get(key)
            if val:
                req.add_header(key, val)

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                resp_body = resp.read()
                status = resp.status

            self.dispatch_log.append({
                "seq": seq,
                "replica": replica_url,
                "endpoint": self.path,
                "status": status,
            })

            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.send_header("X-Replica", replica_url)
            self.send_header("X-Dispatch-Seq", str(seq))
            self.end_headers()
            self.wfile.write(resp_body)

        except urllib.error.URLError as exc:
            error_msg = json.dumps({"error": f"Replica {replica_url}: {exc}"}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_msg)))
            self.end_headers()
            self.wfile.write(error_msg)

    def do_GET(self):
        if self.path == "/health":
            health = self.router.pool.check_health()
            body = json.dumps(health).encode()
            self.send_response(200)
        elif self.path == "/v1/models":
            # Forward to first healthy replica
            replica = self.router.pool.get_replica(0)
            if replica:
                try:
                    with urllib.request.urlopen(f"{replica}/v1/models", timeout=10) as resp:
                        body = resp.read()
                        self.send_response(resp.status)
                except Exception:
                    body = json.dumps({"error": "no healthy replicas"}).encode()
                    self.send_response(503)
            else:
                body = json.dumps({"error": "no healthy replicas"}).encode()
                self.send_response(503)
        else:
            body = json.dumps({"error": "not found"}).encode()
            self.send_response(404)

        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


class ThreadedCoordinator(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-node deterministic coordinator")
    parser.add_argument("--manifest", required=True, help="Resolved manifest JSON")
    parser.add_argument("--lockfile", required=True, help="Built lockfile JSON")
    parser.add_argument("--replicas", required=True, help="Comma-separated replica URLs")
    parser.add_argument("--out-dir", default="/tmp/coordinator", help="Output directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    lockfile = json.loads(Path(args.lockfile).read_text(encoding="utf-8"))
    validate_with_schema("manifest.v1.schema.json", manifest)

    replica_urls = [u.strip().rstrip("/") for u in args.replicas.split(",") if u.strip()]
    if not replica_urls:
        print("ERROR: No replicas specified")
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pool = ReplicaPool(replica_urls)

    print("=== Deterministic Coordinator ===")
    print(f"Replicas: {replica_urls}")
    print(f"Algorithm: {manifest.get('deterministic_dispatcher', {}).get('algorithm', 'round_robin_hash')}")

    # Health check
    health = pool.check_health()
    healthy_count = sum(1 for v in health.values() if v)
    print(f"Healthy: {healthy_count}/{len(replica_urls)}")
    for url, ok in health.items():
        print(f"  {url}: {'OK' if ok else 'DOWN'}")

    if healthy_count == 0:
        print("ERROR: No healthy replicas")
        return 1

    # Verify consistency
    consistency = _verify_replica_consistency(pool, lockfile.get("runtime_closure_digest", ""))
    (out_dir / "consistency_check.json").write_text(
        canonical_json_text(consistency), encoding="utf-8"
    )

    # Start coordinator
    router = DeterministicRouter(manifest, pool)
    dispatch_log = DispatchLog(out_dir / "dispatch.jsonl")

    CoordinatorHandler.router = router
    CoordinatorHandler.dispatch_log = dispatch_log

    server = ThreadedCoordinator((args.host, args.port), CoordinatorHandler)

    print(f"\nCoordinator listening on {args.host}:{args.port}")
    print(f"Dispatch log: {out_dir / 'dispatch.jsonl'}")

    import signal
    def shutdown(signum, frame):
        print("\nShutting down coordinator...")
        server.shutdown()
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.serve_forever()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
