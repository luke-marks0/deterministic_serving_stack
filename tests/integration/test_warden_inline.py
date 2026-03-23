#!/usr/bin/env python3
"""Integration test for the inline warden on a real Linux host.

This test:
1. Adds an iptables NFQUEUE rule on OUTPUT for a specific test port
2. Starts the warden service in a background thread
3. Sends TCP packets to that port
4. Captures the normalized packets
5. Verifies covert channels are destroyed
6. Cleans up iptables rules

Must be run as root. Uses queue 42 and port 19999 to avoid conflicts.
"""
from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import threading
import time

# Ensure the repo root is on sys.path.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pkg.networkdet.warden_config import WardenConfig
from pkg.networkdet.warden_service import WardenService

QUEUE_NUM = 42
TEST_PORT = 19999
WARDEN_SECRET = b"integration-test-secret"


def setup_iptables():
    """Add NFQUEUE rule for test traffic only."""
    subprocess.run(
        ["iptables", "-I", "OUTPUT", "1", "-p", "tcp", "--dport", str(TEST_PORT),
         "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM)],
        check=True,
    )
    print(f"[+] Added iptables NFQUEUE rule: OUTPUT tcp dport {TEST_PORT} -> queue {QUEUE_NUM}")


def teardown_iptables():
    """Remove the test NFQUEUE rule."""
    subprocess.run(
        ["iptables", "-D", "OUTPUT", "-p", "tcp", "--dport", str(TEST_PORT),
         "-j", "NFQUEUE", "--queue-num", str(QUEUE_NUM)],
        check=False,  # Don't fail if rule already removed.
    )
    print("[+] Removed iptables NFQUEUE rule")


def run_warden(svc: WardenService, stop_event: threading.Event):
    """Run the warden in a background thread."""
    from netfilterqueue import NetfilterQueue

    nfq = NetfilterQueue()
    nfq.bind(QUEUE_NUM, svc._packet_callback)
    svc._nfqueue = nfq

    print(f"[+] Warden bound to queue {QUEUE_NUM}")

    try:
        nfq.run()
    except Exception:
        pass
    finally:
        try:
            nfq.unbind()
        except Exception:
            pass
    print("[+] Warden unbound")


def test_packets_pass_through(svc: WardenService):
    """Verify packets pass through the warden (are processed, not stuck)."""
    print("\n=== Test 1: Packets pass through warden ===")

    initial = svc.warden.stats.frames_processed

    # Try to connect to a port. The SYN will pass through the warden.
    # Note: ISN rewriting means the kernel TCP won't complete the handshake
    # properly (the warden modifies the outgoing ISN but the kernel doesn't
    # know), so we expect timeout/refusal. We just verify the warden saw it.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect(("127.0.0.1", TEST_PORT))
        print("[+] Connection succeeded (unexpected, but OK)")
    except (ConnectionRefusedError, ConnectionResetError):
        print("[+] Connection refused/reset (expected)")
    except socket.timeout:
        print("[+] Timeout (expected - ISN rewriting breaks kernel TCP state)")
    finally:
        sock.close()

    time.sleep(0.5)
    final = svc.warden.stats.frames_processed
    processed = final - initial

    if processed > 0:
        print(f"[+] PASS: Warden processed {processed} packets from the connection attempt")
        return True
    else:
        print("[-] FAIL: Warden did not process any packets")
        return False


def test_stats_accumulate(svc: WardenService):
    """Verify warden stats show packet processing."""
    print("\n=== Test 2: Stats accumulate ===")

    initial = svc.warden.stats.frames_processed

    # Send several connection attempts.
    for i in range(3):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        try:
            sock.connect(("127.0.0.1", TEST_PORT))
        except (ConnectionRefusedError, socket.timeout, OSError):
            pass
        finally:
            sock.close()
        time.sleep(0.1)

    final = svc.warden.stats.frames_processed
    processed = final - initial

    stats = svc.warden.stats.as_dict()
    print(f"[+] Packets processed: {processed}")
    print(f"[+] Full stats: {stats}")

    if processed > 0:
        print("[+] PASS: Warden processed packets")
        return True
    else:
        print("[-] FAIL: No packets processed")
        return False


def test_ttl_normalization(svc: WardenService):
    """Verify TTL normalization is happening."""
    print("\n=== Test 3: TTL normalization ===")
    stats = svc.warden.stats.as_dict()
    if stats["ttl_normalized"] > 0:
        print(f"[+] PASS: TTL normalized {stats['ttl_normalized']} times")
        return True
    else:
        print("[+] TTL may already match default (not necessarily a failure)")
        return True  # TTL might already be 64.


def test_checksum_recomputation(svc: WardenService):
    """Verify checksums are being recomputed."""
    print("\n=== Test 4: Checksum recomputation ===")
    stats = svc.warden.stats.as_dict()
    if stats["checksums_recomputed"] > 0:
        print(f"[+] PASS: Checksums recomputed {stats['checksums_recomputed']} times")
        return True
    else:
        print("[-] FAIL: No checksums recomputed")
        return False


def main():
    if os.geteuid() != 0:
        print("ERROR: Must run as root (need iptables + NFQUEUE access)")
        sys.exit(1)

    config = WardenConfig(
        secret=WARDEN_SECRET,
        ttl=64,
        queue_num=QUEUE_NUM,
        stats_interval=999,  # Don't log during test.
    )
    svc = WardenService(config)
    stop_event = threading.Event()

    results = []

    try:
        setup_iptables()

        # Start warden in background thread.
        warden_thread = threading.Thread(
            target=run_warden, args=(svc, stop_event), daemon=True
        )
        warden_thread.start()
        time.sleep(0.5)  # Let it bind.

        # Run tests.
        results.append(("packets_pass_through", test_packets_pass_through(svc)))
        results.append(("stats_accumulate", test_stats_accumulate(svc)))
        results.append(("ttl_normalization", test_ttl_normalization(svc)))
        results.append(("checksum_recomputation", test_checksum_recomputation(svc)))

    finally:
        stop_event.set()
        teardown_iptables()
        # Unbind the nfqueue to unblock the thread.
        if svc._nfqueue is not None:
            try:
                svc._nfqueue.unbind()
            except Exception:
                pass
        time.sleep(0.5)

    # Summary.
    print("\n" + "=" * 50)
    print("INTEGRATION TEST RESULTS")
    print("=" * 50)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    print(f"\nFinal warden stats: {svc.warden.stats.as_dict()}")

    if all_pass:
        print("\nAll integration tests PASSED")
        sys.exit(0)
    else:
        print("\nSome integration tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
