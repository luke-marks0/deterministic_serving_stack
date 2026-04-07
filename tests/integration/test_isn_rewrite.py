#!/usr/bin/env python3
"""Test: full ISN rewriting with bidirectional NFQUEUE on a single host.

Proves that the ActiveWarden can rewrite ISNs on a single host when
both INPUT and OUTPUT chains are intercepted, enabling full MRF
normalization including sequence number rewriting.

iptables rules:
  OUTPUT -p tcp --sport PORT -j NFQUEUE --queue-num Q  (server→client)
  INPUT  -p tcp --dport PORT -j NFQUEUE --queue-num Q  (client→server)

This gives the warden visibility into both directions of the TCP
handshake, so it can track offsets and apply inverse transformations.

Usage:
    sudo python3 tests/integration/test_isn_rewrite.py
"""
from __future__ import annotations

import socket as sock_mod
import struct
import subprocess
import sys
import threading
import time
import urllib.request

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pkg.networkdet.checksums import ip_checksum, tcp_checksum
from pkg.networkdet.warden import ActiveWarden, ETH_HEADER_LEN
from pkg.networkdet.warden_config import WardenConfig
from pkg.networkdet.warden_service import FAKE_ETH, WardenService
from tests.integration.test_http_server import (
    RESPONSE_DETERMINISTIC,
    start_server,
)

QUEUE_NUM = 44
SERVER_PORT = 18081
WARDEN_SECRET = b"isn-rewrite-proof-2026"


def setup_iptables() -> None:
    """Add OUTPUT-only NFQUEUE rules for both directions.

    On loopback, both client→server and server→client packets traverse
    OUTPUT exactly once. Using OUTPUT-only avoids double-processing that
    would occur with INPUT+OUTPUT rules on loopback.
    """
    subprocess.run([
        "iptables", "-I", "OUTPUT", "1",
        "-p", "tcp", "--sport", str(SERVER_PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM),
    ], check=True)
    subprocess.run([
        "iptables", "-I", "OUTPUT", "1",
        "-p", "tcp", "--dport", str(SERVER_PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM),
    ], check=True)


def teardown_iptables() -> None:
    """Remove both NFQUEUE rules."""
    subprocess.run([
        "iptables", "-D", "OUTPUT",
        "-p", "tcp", "--sport", str(SERVER_PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM),
    ], check=False)
    subprocess.run([
        "iptables", "-D", "OUTPUT",
        "-p", "tcp", "--dport", str(SERVER_PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM),
    ], check=False)


def run_test():
    results = []
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        if not condition:
            failed += 1
        else:
            passed += 1
        msg = f"  {name:<55s} {status}"
        if detail and not condition:
            msg += f"  ({detail})"
        results.append(msg)
        print(msg)
        return condition

    print("=" * 60)
    print("ISN REWRITE: BIDIRECTIONAL NFQUEUE ON SINGLE HOST")
    print("=" * 60)
    print()

    # --- Setup ---
    print("Setup: starting HTTP server and warden (ISN rewrite ON)...")
    http_server, _ = start_server(SERVER_PORT)

    config = WardenConfig(
        secret=WARDEN_SECRET,
        queue_num=QUEUE_NUM,
        chain="OUTPUT",
        skip_isn_rewrite=False,
        inline=True,  # Inline mode: subtract offsets on return path
        stats_interval=9999,
    )
    svc = WardenService(config)

    from netfilterqueue import NetfilterQueue
    nfq = NetfilterQueue()
    nfq.bind(config.queue_num, svc._packet_callback)

    warden_thread = threading.Thread(target=nfq.run, daemon=True)

    setup_iptables()
    warden_thread.start()
    time.sleep(0.5)
    print(f"Setup complete. Server on :{SERVER_PORT}, warden on queue {QUEUE_NUM}")
    print(f"ISN rewriting: ENABLED (inline mode, OUTPUT-only bidirectional)")
    print()

    # --- Test 1: HTTP delivery + per-request stats ---
    print("Test 1: HTTP Delivery with ISN Rewriting")
    NUM_TRIALS = 3
    per_request_stats = []
    for trial in range(NUM_TRIALS):
        svc.capture_reset()
        svc.warden.reset()
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{SERVER_PORT}/deterministic", timeout=10,
            )
            body = resp.read()
            time.sleep(0.3)
            stats = svc.warden.stats.as_dict()
            frames = svc.capture.frame_count
            per_request_stats.append(stats)
            check(
                f"Request {trial}: {len(body)}B delivered, "
                f"{stats['isn_rewrites']} ISN rewrites, "
                f"{frames} frames",
                body == RESPONSE_DETERMINISTIC,
                f"got {body[:40]}..." if body != RESPONSE_DETERMINISTIC else "",
            )
        except Exception as e:
            check(f"Request {trial}: HTTP request succeeded", False, str(e))

    print()

    # --- Test 2: Per-request ISN rewrite consistency ---
    print("Test 2: ISN Rewriting Consistency")
    isn_counts = [s["isn_rewrites"] for s in per_request_stats]
    frame_counts = [s["frames_processed"] for s in per_request_stats]
    check(
        f"Every request has 2 ISN rewrites (SYN+SYN-ACK): {isn_counts}",
        all(c == 2 for c in isn_counts),
    )
    # Frame counts may vary by ±2 due to kernel TCP timing (delayed ACK,
    # window probes). Check they're in a reasonable range, not identical.
    check(
        f"Frame counts within range across requests: {frame_counts}",
        max(frame_counts) - min(frame_counts) <= 4,
        f"spread too wide: {frame_counts}",
    )
    check(
        f"Total: {sum(isn_counts)} ISN rewrites across {NUM_TRIALS} requests",
        sum(isn_counts) == NUM_TRIALS * 2,
    )

    print()

    # --- Test 3: Replay determinism ---
    print("Test 3: Replay Determinism (with ISN rewriting)")
    svc.capture_reset()
    svc.warden.reset()
    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{SERVER_PORT}/deterministic", timeout=10,
        )
        resp.read()
        time.sleep(0.3)
    except Exception:
        pass

    raw_packets = svc.raw_packets()
    captured_frames = svc.capture.drain()

    check(f"Captured {len(raw_packets)} raw packets", len(raw_packets) > 0)

    # Replay through fresh warden with same secret and mode.
    fresh = ActiveWarden(secret=WARDEN_SECRET, skip_isn_rewrite=False, inline=True)
    replay_match = 0
    for raw_ip, expected in zip(raw_packets, captured_frames):
        frame = FAKE_ETH + raw_ip
        replayed = fresh.normalize(frame)
        if replayed == expected:
            replay_match += 1

    check(
        f"Replay: {replay_match}/{len(raw_packets)} frames match",
        replay_match == len(raw_packets),
    )

    print()

    # --- Test 4: Independent witness ---
    print("Test 4: Independent Witness (raw socket)")
    svc.capture_reset()
    svc.warden.reset()

    raw_sock = sock_mod.socket(sock_mod.AF_PACKET, sock_mod.SOCK_RAW, sock_mod.htons(3))
    raw_sock.settimeout(0.1)
    raw_sock.bind(("lo", 0))

    # Drain stale packets.
    while True:
        try:
            raw_sock.recv(65535)
        except (sock_mod.timeout, BlockingIOError):
            break

    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{SERVER_PORT}/deterministic", timeout=10,
        )
        resp.read()
    except Exception:
        pass
    time.sleep(0.5)

    witness_packets = []
    while True:
        try:
            witness_packets.append(raw_sock.recv(65535))
        except (sock_mod.timeout, BlockingIOError):
            break
    raw_sock.close()

    # Filter server→client TCP packets.
    wire_ip_packets = []
    for pkt in witness_packets:
        if len(pkt) < 34:
            continue
        ethertype = struct.unpack("!H", pkt[12:14])[0]
        if ethertype != 0x0800:
            continue
        ip_start = 14
        if pkt[ip_start + 9] != 6:
            continue
        ihl = (pkt[ip_start] & 0x0F) * 4
        tcp_start = ip_start + ihl
        if len(pkt) < tcp_start + 4:
            continue
        src_port = struct.unpack("!H", pkt[tcp_start:tcp_start + 2])[0]
        if src_port == SERVER_PORT:
            wire_ip_packets.append(pkt[ip_start:])

    check(f"Raw socket captured {len(wire_ip_packets)} server packets",
          len(wire_ip_packets) > 0)

    warden_frames = list(svc.capture.drain())
    # Only compare server→client frames from warden capture.
    warden_server_ip = []
    for f in warden_frames:
        ip_data = f[ETH_HEADER_LEN:]
        if len(ip_data) < 24:
            continue
        ihl = (ip_data[0] & 0x0F) * 4
        if len(ip_data) < ihl + 4:
            continue
        sp = struct.unpack("!H", ip_data[ihl:ihl + 2])[0]
        if sp == SERVER_PORT:
            warden_server_ip.append(ip_data)

    matched = 0
    wire_set = set(wire_ip_packets)
    for wp in warden_server_ip:
        if wp in wire_set:
            matched += 1

    check(
        f"Wire witness: {matched}/{len(warden_server_ip)} warden frames on wire",
        matched == len(warden_server_ip),
        f"only {matched} of {len(warden_server_ip)} matched",
    )

    print()

    # --- Summary ---
    print("Per-Request Breakdown")
    for i, s in enumerate(per_request_stats):
        print(f"    Request {i}: "
              f"{s['frames_processed']} frames, "
              f"{s['isn_rewrites']} ISN rewrites, "
              f"{s['ip_id_rewrites']} IP ID rewrites, "
              f"{s['checksums_recomputed']} checksums")
    print(f"    Average: "
          f"{sum(s['frames_processed'] for s in per_request_stats)/NUM_TRIALS:.0f} frames/req, "
          f"{sum(s['isn_rewrites'] for s in per_request_stats)/NUM_TRIALS:.0f} ISN rewrites/req")

    print()
    print("=" * 60)
    if failed == 0:
        print(f"ALL {passed} CHECKS PASSED")
        print("ISN rewriting works on a single host with bidirectional NFQUEUE.")
    else:
        print(f"FAILED: {failed}/{passed + failed} checks")
    print("=" * 60)

    # --- Cleanup ---
    teardown_iptables()
    nfq.unbind()
    http_server.shutdown()

    return failed == 0


if __name__ == "__main__":
    if not run_test():
        sys.exit(1)
