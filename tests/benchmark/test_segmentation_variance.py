#!/usr/bin/env python3
"""Test: does the kernel TCP stack produce different segmentation across runs?

Sends the same large HTTP request N times through the inline warden on a
single host. For each request, records:
  - Number of frames captured by the warden
  - SHA-256 digest over normalized frames
  - Warden stats (ISN rewrites, options stripped, etc.)

If segmentation is deterministic, all N runs should have identical frame
counts and digests. If it varies, we'll see it directly.

Usage:
    sudo python3 tests/benchmark/test_segmentation_variance.py [--response-size 50000] [--trials 10]
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import socket as sock_mod
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pkg.networkdet.warden import ActiveWarden, ETH_HEADER_LEN
from pkg.networkdet.warden_config import WardenConfig
from pkg.networkdet.warden_service import FAKE_ETH, WardenService

PORT = 18090
QUEUE = 50
SECRET = b"segmentation-variance-test-2026"


# --- Deterministic HTTP server with large responses ---

def _generate_response(size: int) -> bytes:
    """Generate a deterministic response body of exactly `size` bytes."""
    block = hashlib.sha256(b"deterministic-payload-block").digest()  # 32 bytes
    reps = (size // len(block)) + 1
    return (block * reps)[:size]


class LargeHandler(http.server.BaseHTTPRequestHandler):
    response_body: bytes = b""

    def version_string(self):
        return "TestServer/1.0"

    def date_time_string(self, timestamp=None):
        return "Thu, 01 Jan 2026 00:00:00 GMT"

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(self.response_body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(self.response_body)


def setup_iptables():
    # Bidirectional on OUTPUT — on loopback both directions traverse OUTPUT once
    subprocess.run([
        "iptables", "-I", "OUTPUT", "1",
        "-p", "tcp", "--sport", str(PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE),
    ], check=True)
    subprocess.run([
        "iptables", "-I", "OUTPUT", "1",
        "-p", "tcp", "--dport", str(PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE),
    ], check=True)


def teardown_iptables():
    subprocess.run([
        "iptables", "-D", "OUTPUT",
        "-p", "tcp", "--sport", str(PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE),
    ], check=False)
    subprocess.run([
        "iptables", "-D", "OUTPUT",
        "-p", "tcp", "--dport", str(PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE),
    ], check=False)


def run_test(response_size: int, num_trials: int):
    print("=" * 70)
    print("SEGMENTATION VARIANCE TEST")
    print(f"Response size: {response_size:,} bytes")
    print(f"Trials: {num_trials}")
    print(f"Expected frames/trial: ~{response_size // 1460 + 5} "
          f"({response_size // 1460} data + ~5 control)")
    print("=" * 70)
    print()

    # Start HTTP server
    body = _generate_response(response_size)
    LargeHandler.response_body = body
    server = http.server.HTTPServer(("0.0.0.0", PORT), LargeHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    # Start warden with full ISN rewriting in inline mode
    config = WardenConfig(
        secret=SECRET, queue_num=QUEUE,
        skip_isn_rewrite=False, inline=True, stats_interval=9999,
    )
    svc = WardenService(config)

    from netfilterqueue import NetfilterQueue
    nfq = NetfilterQueue()
    nfq.bind(QUEUE, svc._packet_callback)
    threading.Thread(target=nfq.run, daemon=True).start()

    setup_iptables()
    time.sleep(0.5)

    # Run trials
    results = []
    for trial in range(num_trials):
        svc.capture_reset()
        svc.warden.reset()

        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}/data", timeout=30,
            )
            received = resp.read()
            time.sleep(0.5)  # Let trailing FIN/ACK arrive

            digest = svc.capture_digest()
            frame_count = svc.capture.frame_count
            stats = svc.warden.stats.as_dict()
            body_ok = received == body

            results.append({
                "trial": trial,
                "frames": frame_count,
                "digest": digest,
                "body_ok": body_ok,
                "bytes_received": len(received),
                "ip_id_rewrites": stats["ip_id_rewrites"],
                "checksums_recomputed": stats["checksums_recomputed"],
            })

            print(f"  Trial {trial:2d}: {frame_count:4d} frames  "
                  f"digest={digest[7:23]}...  "
                  f"body={'OK' if body_ok else 'MISMATCH'}")

        except Exception as e:
            print(f"  Trial {trial:2d}: FAILED — {e}")
            results.append({
                "trial": trial, "frames": 0, "digest": "ERROR",
                "body_ok": False, "bytes_received": 0,
                "ip_id_rewrites": 0, "checksums_recomputed": 0,
            })

    # Cleanup
    teardown_iptables()
    nfq.unbind()
    server.shutdown()

    # Analysis
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    frame_counts = [r["frames"] for r in results if r["frames"] > 0]
    digests = [r["digest"] for r in results if r["digest"] != "ERROR"]
    all_body_ok = all(r["body_ok"] for r in results)

    print(f"\nDelivery: {'ALL OK' if all_body_ok else 'SOME FAILED'}")
    print(f"  {sum(r['body_ok'] for r in results)}/{num_trials} received correct body")

    if frame_counts:
        unique_counts = sorted(set(frame_counts))
        unique_digests = len(set(digests))
        print(f"\nFrame counts across {len(frame_counts)} trials:")
        print(f"  Min: {min(frame_counts)}")
        print(f"  Max: {max(frame_counts)}")
        print(f"  Unique values: {unique_counts}")
        print(f"  Variance: {max(frame_counts) - min(frame_counts)} frames")

        print(f"\nDigest comparison:")
        print(f"  Unique digests: {unique_digests} out of {len(digests)} trials")
        if unique_digests == 1:
            print(f"  DETERMINISTIC: all trials produced identical frames")
        else:
            print(f"  NON-DETERMINISTIC: {unique_digests} different frame sequences")
            # Show which trials match which
            by_digest = {}
            for r in results:
                if r["digest"] != "ERROR":
                    by_digest.setdefault(r["digest"][:30], []).append(r["trial"])
            for d, trials in sorted(by_digest.items(), key=lambda x: x[1][0]):
                print(f"    {d}... → trials {trials}")

    print()
    if frame_counts and len(set(frame_counts)) == 1 and len(set(digests)) == 1:
        print("VERDICT: Kernel segmentation appears deterministic for this workload")
    elif frame_counts:
        print("VERDICT: Kernel segmentation is NON-DETERMINISTIC")
        print("  Different runs of the same request produce different frame sequences.")
        print("  This is a covert channel via TCP segmentation.")
    print()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-size", type=int, default=50000,
                        help="Response body size in bytes (default: 50000)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of identical requests (default: 10)")
    args = parser.parse_args()
    run_test(args.response_size, args.trials)
