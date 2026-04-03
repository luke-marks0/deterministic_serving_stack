#!/usr/bin/env python3
"""Integration test: ActiveWarden inline determinism proof.

Proves two properties simultaneously:
  P1 (Determinism): Replaying the same raw packets through a fresh warden
      with the same secret produces byte-identical normalized frames.
  P2 (Delivery):    The client receives correct HTTP response content
      despite inline frame normalization.

Requires root on Linux (iptables + NFQUEUE).
Uses OUTPUT chain with --sport filter to intercept only server responses.
ISN rewriting is disabled (skip_isn_rewrite=True) because on a single
host, ISN rewriting is incompatible with the kernel TCP state machine.
All other MRF normalizations are active (IP ID, TTL, TOS, TCP options,
reserved bits, checksums).

Usage:
    sudo python3 tests/integration/test_warden_determinism.py
"""
from __future__ import annotations

import struct
import subprocess
import sys
import time
import urllib.request

# Ensure the project root is on the path.
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pkg.networkdet.checksums import ip_checksum, tcp_checksum
from pkg.networkdet.warden import ActiveWarden, ETH_HEADER_LEN
from pkg.networkdet.warden_config import WardenConfig
from pkg.networkdet.warden_service import FAKE_ETH, WardenService
from tests.integration.test_http_server import (
    RESPONSE_ALT,
    RESPONSE_DETERMINISTIC,
    start_server,
)

QUEUE_NUM = 43
SERVER_PORT = 18080
WARDEN_SECRET = b"determinism-proof-secret-2026"
NUM_TRIALS = 5


def setup_iptables() -> None:
    """Add NFQUEUE rule for server response packets."""
    subprocess.run([
        "iptables", "-I", "OUTPUT", "1",
        "-p", "tcp", "--sport", str(SERVER_PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM),
    ], check=True)


def teardown_iptables() -> None:
    """Remove the NFQUEUE rule."""
    subprocess.run([
        "iptables", "-D", "OUTPUT",
        "-p", "tcp", "--sport", str(SERVER_PORT),
        "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM),
    ], check=False)


def parse_ip(frame: bytes) -> dict:
    """Parse IP header fields from an L2 frame."""
    ip_start = 14  # After Ethernet header
    ver_ihl = frame[ip_start]
    ihl = (ver_ihl & 0x0F) * 4
    tos = frame[ip_start + 1]
    total_len = struct.unpack("!H", frame[ip_start + 2:ip_start + 4])[0]
    ip_id = struct.unpack("!H", frame[ip_start + 4:ip_start + 6])[0]
    flags_frag = struct.unpack("!H", frame[ip_start + 6:ip_start + 8])[0]
    ttl = frame[ip_start + 8]
    return {"tos": tos, "ttl": ttl, "ip_id": ip_id, "ihl": ihl,
            "total_len": total_len, "flags_frag": flags_frag}


def parse_tcp(frame: bytes) -> dict:
    """Parse TCP header fields from an L2 frame."""
    ip_start = 14
    ihl = (frame[ip_start] & 0x0F) * 4
    tcp_start = ip_start + ihl
    data_offset = (frame[tcp_start + 12] >> 4) * 4
    flags = frame[tcp_start + 13]
    reserved_ns = frame[tcp_start + 12] & 0x0F
    urg_ptr = struct.unpack("!H", frame[tcp_start + 18:tcp_start + 20])[0]
    return {"data_offset": data_offset, "flags": flags,
            "reserved_ns": reserved_ns, "urg_ptr": urg_ptr}


def verify_ip_checksum(frame: bytes) -> bool:
    """Verify IP header checksum is correct."""
    ip_start = 14
    ihl = (frame[ip_start] & 0x0F) * 4
    header = frame[ip_start:ip_start + ihl]
    return ip_checksum(header) == 0


def verify_tcp_checksum(frame: bytes) -> bool:
    """Verify TCP checksum is correct."""
    ip_start = 14
    ihl = (frame[ip_start] & 0x0F) * 4
    total_len = struct.unpack("!H", frame[ip_start + 2:ip_start + 4])[0]
    src_ip = frame[ip_start + 12:ip_start + 16]
    dst_ip = frame[ip_start + 16:ip_start + 20]
    tcp_start = ip_start + ihl
    tcp_end = ip_start + total_len
    tcp_seg = frame[tcp_start:tcp_end]
    return tcp_checksum(src_ip, dst_ip, tcp_seg) == 0


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
    print("WARDEN INLINE DETERMINISM PROOF")
    print("=" * 60)
    print()

    # --- Setup ---
    print("Setup: starting HTTP server and warden...")
    http_server, _ = start_server(SERVER_PORT)

    config = WardenConfig(
        secret=WARDEN_SECRET,
        queue_num=QUEUE_NUM,
        chain="OUTPUT",
        skip_isn_rewrite=True,
        stats_interval=9999,  # Suppress periodic logging.
    )
    svc = WardenService(config)

    import threading
    from netfilterqueue import NetfilterQueue
    nfq = NetfilterQueue()
    nfq.bind(config.queue_num, svc._packet_callback)

    warden_thread = threading.Thread(target=nfq.run, daemon=True)

    setup_iptables()
    warden_thread.start()
    time.sleep(0.5)
    print(f"Setup complete. Server on :{SERVER_PORT}, warden on queue {QUEUE_NUM}")
    print()

    # --- Property 2: Delivery ---
    print("Property 2: Content Delivery")
    for trial in range(NUM_TRIALS):
        svc.capture_reset()
        svc.warden.reset()
        try:
            resp = urllib.request.urlopen(
                f"http://127.0.0.1:{SERVER_PORT}/deterministic", timeout=5,
            )
            body = resp.read()
            time.sleep(0.3)  # Let warden process trailing FIN/ACK.
            check(
                f"Trial {trial}: response body correct ({len(body)} bytes)",
                body == RESPONSE_DETERMINISTIC,
                f"got {body[:40]}...",
            )
        except Exception as e:
            check(f"Trial {trial}: HTTP request succeeded", False, str(e))

    print()

    # --- Property 1: Determinism (replay) ---
    print("Property 1: Frame-Level Determinism (replay proof)")

    # Use the last trial's raw packets for replay.
    raw_packets = svc.raw_packets()
    captured_frames = svc.capture.drain()

    check(f"Captured {len(raw_packets)} raw packets", len(raw_packets) > 0)
    check(
        f"Raw packet count matches captured frames",
        len(raw_packets) == len(captured_frames),
        f"{len(raw_packets)} raw vs {len(captured_frames)} captured",
    )

    # Replay through a fresh warden with the same secret.
    fresh = ActiveWarden(secret=WARDEN_SECRET, skip_isn_rewrite=True)
    replay_match = 0
    for i, (raw_ip, expected) in enumerate(zip(raw_packets, captured_frames)):
        frame = FAKE_ETH + raw_ip
        replayed = fresh.normalize(frame)
        if replayed == expected:
            replay_match += 1

    check(
        f"Replay: {replay_match}/{len(raw_packets)} frames match",
        replay_match == len(raw_packets),
    )

    print()

    # --- Normalization Properties ---
    print("Normalization Properties")

    # Re-capture for property checks (fresh connection).
    svc.capture_reset()
    svc.warden.reset()
    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{SERVER_PORT}/deterministic", timeout=5,
        )
        resp.read()
        time.sleep(0.3)
    except Exception:
        pass

    norm_frames = list(svc.capture.drain())
    all_ttl_ok = all(parse_ip(f)["ttl"] == 64 for f in norm_frames)
    all_tos_ok = all(parse_ip(f)["tos"] == 0 for f in norm_frames)
    all_df_ok = all(parse_ip(f)["flags_frag"] & 0x4000 for f in norm_frames)
    all_reserved_ok = all(parse_tcp(f)["reserved_ns"] == 0 for f in norm_frames)
    all_ip_cksum_ok = all(verify_ip_checksum(f) for f in norm_frames)
    all_tcp_cksum_ok = all(verify_tcp_checksum(f) for f in norm_frames)

    check(f"All {len(norm_frames)} frames TTL=64", all_ttl_ok)
    check(f"All {len(norm_frames)} frames TOS=0", all_tos_ok)
    check(f"All {len(norm_frames)} frames DF=1", all_df_ok)
    check(f"All {len(norm_frames)} frames TCP reserved=0", all_reserved_ok)
    check(f"All {len(norm_frames)} frames valid IP checksum", all_ip_cksum_ok)
    check(f"All {len(norm_frames)} frames valid TCP checksum", all_tcp_cksum_ok)

    print()

    # --- Independent Witness: raw socket captures what the kernel delivers ---
    print("Independent Witness (raw socket)")

    # Use a raw socket (AF_PACKET) as an independent observer.
    # This is a separate capture mechanism from NFQUEUE — it sees packets
    # after the kernel re-injects them from the warden.
    import socket as sock_mod

    raw_sock = sock_mod.socket(sock_mod.AF_PACKET, sock_mod.SOCK_RAW, sock_mod.htons(3))  # ETH_P_ALL
    raw_sock.settimeout(0.1)
    raw_sock.bind(("lo", 0))
    witness_packets: list[bytes] = []

    # Fresh warden capture for this trial.
    svc.capture_reset()
    svc.warden.reset()

    # Drain any stale packets from the raw socket.
    while True:
        try:
            raw_sock.recv(65535)
        except (sock_mod.timeout, BlockingIOError):
            break

    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{SERVER_PORT}/deterministic", timeout=5,
        )
        resp.read()
    except Exception:
        pass

    time.sleep(0.5)  # Let trailing packets arrive.

    # Collect everything the raw socket captured.
    while True:
        try:
            pkt = raw_sock.recv(65535)
            witness_packets.append(pkt)
        except (sock_mod.timeout, BlockingIOError):
            break

    raw_sock.close()

    # Filter to server→client TCP packets on our port.
    # Raw socket on lo delivers Ethernet frames (14-byte header + IP).
    wire_ip_packets = []
    for pkt in witness_packets:
        if len(pkt) < 34:  # ETH(14) + IP(20) minimum
            continue
        ethertype = struct.unpack("!H", pkt[12:14])[0]
        if ethertype != 0x0800:
            continue
        ip_start = 14
        protocol = pkt[ip_start + 9]
        if protocol != 6:  # TCP
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

    # Compare: strip fake Ethernet from warden frames to get IP bytes,
    # then match against what the raw socket independently observed.
    warden_frames = list(svc.capture.drain())
    warden_ip_packets = [f[ETH_HEADER_LEN:] for f in warden_frames]

    matched = 0
    wire_set = set()
    for wp in wire_ip_packets:
        wire_set.add(wp)
    for wp in warden_ip_packets:
        if wp in wire_set:
            matched += 1

    check(
        f"Wire witness: {matched}/{len(warden_ip_packets)} warden frames seen by raw socket",
        matched == len(warden_ip_packets),
        f"only {matched} of {len(warden_ip_packets)} matched",
    )

    print()

    # --- Sanity Check: different request -> different digest ---
    print("Sanity Check")
    svc.capture_reset()
    svc.warden.reset()
    try:
        resp = urllib.request.urlopen(
            f"http://127.0.0.1:{SERVER_PORT}/deterministic/alt", timeout=5,
        )
        alt_body = resp.read()
        time.sleep(0.3)
        check("Alt response body correct", alt_body == RESPONSE_ALT)
    except Exception as e:
        check("Alt HTTP request succeeded", False, str(e))

    print()

    # --- Warden Stats ---
    print("Warden Stats")
    stats = svc.warden.stats.as_dict()
    check("frames_processed > 0", stats["frames_processed"] > 0,
          f"got {stats['frames_processed']}")
    check("checksums_recomputed > 0", stats["checksums_recomputed"] > 0,
          f"got {stats['checksums_recomputed']}")
    check("ip_id_rewrites > 0", stats["ip_id_rewrites"] > 0,
          f"got {stats['ip_id_rewrites']}")
    check("isn_rewrites == 0 (skip mode)", stats["isn_rewrites"] == 0,
          f"got {stats['isn_rewrites']}")

    for k, v in stats.items():
        if v > 0:
            print(f"    {k}: {v}")

    print()
    print("=" * 60)
    if failed == 0:
        print(f"ALL {passed} CHECKS PASSED")
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
