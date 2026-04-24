#!/usr/bin/env python3
"""Smart attacker: boundary-counting covert channel through bucket pacing.

Instead of PAM (encoding in delay levels, which bucketing destroys), the smart
attacker encodes by shifting tokens across bucket boundaries. The receiver
counts tokens per bucket and decodes from the counts.

Encoding: at each bucket boundary, server stalls s ∈ {0,...,m} tokens past
the boundary. Receiver observes count_k = R·B + s_{k-1} - s_k and recovers
symbols via differential decoding.

Usage:
    python experiments/timing_channel/boundary_attack_experiment.py
"""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import socket
import struct
import threading
import time
from dataclasses import dataclass


WORDS = (
    "the model generates tokens sequentially based on attention patterns and "
    "learned representations from training data which enables coherent text "
    "generation across many different domains including tasks like summarization"
).split()


# ---------------------------------------------------------------------------
# Message framing (reused from covert_channel_demo)
# ---------------------------------------------------------------------------

def bytes_to_symbols(data: bytes, bits_per_sym: int) -> list[int]:
    bits = "".join(f"{b:08b}" for b in data)
    while len(bits) % bits_per_sym:
        bits += "0"
    return [int(bits[i:i + bits_per_sym], 2) for i in range(0, len(bits), bits_per_sym)]


def symbols_to_bytes(symbols: list[int], bits_per_sym: int, nbytes: int) -> bytes:
    bits = "".join(f"{s:0{bits_per_sym}b}" for s in symbols)
    return bytes(int(bits[i:i+8], 2) for i in range(0, nbytes * 8, 8))


def frame_message(secret: bytes, levels: int, bits_per_sym: int) -> list[int]:
    length = struct.pack(">I", len(secret))
    cksum = hashlib.sha256(secret).digest()[:4]
    syms = bytes_to_symbols(length + secret + cksum, bits_per_sym)
    for s in syms:
        assert 0 <= s < levels, f"symbol {s} >= levels {levels}"
    return syms


def decode_message(symbols: list[int], bits_per_sym: int) -> tuple[bytes, bool]:
    if len(symbols) < math.ceil(8 * 8 / bits_per_sym):
        return b"", False
    len_n = math.ceil(4 * 8 / bits_per_sym)
    raw_len = symbols_to_bytes(symbols[:len_n], bits_per_sym, 4)
    msg_len = struct.unpack(">I", raw_len)[0]
    if msg_len > 10000:
        return b"", False
    total = 4 + msg_len + 4
    need = math.ceil(total * 8 / bits_per_sym)
    if len(symbols) < need:
        return b"", False
    raw = symbols_to_bytes(symbols[:need], bits_per_sym, total)
    msg = raw[4:4 + msg_len]
    got_ck = raw[4 + msg_len:4 + msg_len + 4]
    want_ck = hashlib.sha256(msg).digest()[:4]
    return msg, got_ck == want_ck


# ---------------------------------------------------------------------------
# Bucket-pacing proxy (same as bucket_experiment.py)
# ---------------------------------------------------------------------------

def bucket_proxy(listen_port: int, server_port: int, bucket_s: float):
    proxy = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    proxy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy.bind(("0.0.0.0", listen_port))
    proxy.settimeout(60.0)

    data, client_addr = proxy.recvfrom(64)
    proxy.sendto(data, ("127.0.0.1", server_port))

    buffer = []
    lock = threading.Lock()
    done = threading.Event()

    def receiver():
        while not done.is_set():
            try:
                pkt, _ = proxy.recvfrom(4096)
                with lock:
                    buffer.append(pkt)
                if b'"done"' in pkt:
                    done.set()
            except socket.timeout:
                done.set()

    recv_thread = threading.Thread(target=receiver, daemon=True)
    recv_thread.start()

    while not done.is_set():
        time.sleep(bucket_s)
        with lock:
            batch = list(buffer)
            buffer.clear()
        for pkt in batch:
            proxy.sendto(pkt, client_addr)

    time.sleep(0.01)
    with lock:
        for pkt in buffer:
            proxy.sendto(pkt, client_addr)
        buffer.clear()

    recv_thread.join(timeout=2)
    proxy.close()


# ---------------------------------------------------------------------------
# Boundary-shifting server
# ---------------------------------------------------------------------------

def boundary_server_proc(port: int, secret_bytes: bytes,
                         bucket_s: float, tok_rate: float, stall_s: float):
    tpb = int(tok_rate * bucket_s)
    m = min(int(tok_rate * stall_s), tpb)
    levels = m + 1
    if levels < 2:
        return
    bits_per_sym = int(math.log2(levels))
    if bits_per_sym < 1:
        bits_per_sym = 1
    effective_levels = 2 ** bits_per_sym
    symbols = frame_message(secret_bytes, effective_levels, bits_per_sym)

    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.settimeout(30.0)
    data, addr = srv.recvfrom(64)

    gap = 1.0 / tok_rate
    t0 = time.perf_counter()
    tok_idx = 0
    prev_shift = 0

    for bnd_idx, sym in enumerate(symbols):
        # Tokens for this bucket: tpb + carry-in - shift-out
        n_emit = tpb + prev_shift - sym
        bucket_start = t0 + bnd_idx * bucket_s

        for j in range(max(0, n_emit)):
            token = WORDS[tok_idx % len(WORDS)]
            pkt = json.dumps({"t": token, "i": tok_idx}, separators=(",", ":")).encode()
            deadline = bucket_start + j * gap
            while time.perf_counter() < deadline:
                pass
            srv.sendto(pkt, addr)
            tok_idx += 1

        # Wait past the boundary before emitting shifted tokens
        boundary = t0 + (bnd_idx + 1) * bucket_s
        while time.perf_counter() < boundary + 0.0002:
            pass

        prev_shift = sym

    # Final bucket: emit remaining (tpb + prev_shift) tokens
    final_start = t0 + len(symbols) * bucket_s
    for j in range(tpb + prev_shift):
        token = WORDS[tok_idx % len(WORDS)]
        pkt = json.dumps({"t": token, "i": tok_idx}, separators=(",", ":")).encode()
        deadline = final_start + j * gap
        while time.perf_counter() < deadline:
            pass
        srv.sendto(pkt, addr)
        tok_idx += 1

    srv.sendto(b'{"done":true}', addr)
    srv.close()


# ---------------------------------------------------------------------------
# Boundary-counting client
# ---------------------------------------------------------------------------

def boundary_client(host: str, port: int, bucket_s: float, tok_rate: float,
                    stall_s: float, n_symbols: int) -> dict:
    tpb = int(tok_rate * bucket_s)
    m = min(int(tok_rate * stall_s), tpb)
    levels = m + 1
    bits_per_sym = int(math.log2(levels))
    if bits_per_sym < 1:
        bits_per_sym = 1
    effective_levels = 2 ** bits_per_sym

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(60.0)
    sock.sendto(b"start", (host, port))

    timestamps = []
    while True:
        try:
            data, _ = sock.recvfrom(4096)
        except socket.timeout:
            break
        t = time.perf_counter()
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if obj.get("done"):
            break
        timestamps.append(t)

    sock.close()

    if len(timestamps) < 2:
        return {"error": "too few tokens", "tokens": len(timestamps)}

    # Detect bucket boundaries from timing: cluster tokens into bursts
    # Tokens within a bucket arrive nearly simultaneously (sub-ms).
    # Inter-bucket gaps are ~bucket_s.
    gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    threshold = bucket_s * 0.4

    # Group into buckets
    buckets = []
    current_count = 1
    for g in gaps:
        if g > threshold:
            buckets.append(current_count)
            current_count = 1
        else:
            current_count += 1
    buckets.append(current_count)

    # Differential decoding: s_k = s_{k-1} + tpb - count_k
    # First bucket: count_0 = tpb - s_0 (no carry-in), so s_0 = tpb - count_0
    decoded_syms = []
    prev_s = 0
    for count in buckets[:-1]:  # last bucket is the flush, skip it
        s = prev_s + tpb - count
        s = max(0, min(effective_levels - 1, s))
        decoded_syms.append(s)
        prev_s = s

    # Decode message
    msg, ok = decode_message(decoded_syms, bits_per_sym)

    # Symbol accuracy (compare to what was sent)
    wall = timestamps[-1] - timestamps[0]
    total_bits = len(decoded_syms) * bits_per_sym
    measured_bps = total_bits / wall if wall > 0 else 0

    return {
        "tokens": len(timestamps),
        "buckets_detected": len(buckets),
        "symbols_decoded": len(decoded_syms),
        "symbols_expected": n_symbols,
        "wall_s": round(wall, 3),
        "measured_bps": round(measured_bps, 1),
        "decoded_ok": ok,
        "decoded_msg": msg.decode("utf-8", errors="replace") if msg else "",
        "bucket_counts": buckets[:20],  # first 20 for inspection
        "tpb": tpb,
        "levels": effective_levels,
        "bits_per_sym": bits_per_sym,
    }


# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_one(secret: str, bucket_ms: float, tok_rate: float = 50.0,
            stall_ms: float = None, base_port: int = 9200) -> dict:
    bucket_s = bucket_ms / 1000.0
    if stall_ms is None:
        stall_s = bucket_s  # unlimited stall
    else:
        stall_s = stall_ms / 1000.0

    secret_bytes = secret.encode("utf-8")
    tpb = int(tok_rate * bucket_s)
    m = min(int(tok_rate * stall_s), tpb)
    levels = m + 1
    if levels < 2:
        return {"error": f"m={m}, need ≥1", "bucket_ms": bucket_ms}

    bits_per_sym = int(math.log2(levels))
    if bits_per_sym < 1:
        bits_per_sym = 1
    effective_levels = 2 ** bits_per_sym
    symbols = frame_message(secret_bytes, effective_levels, bits_per_sym)

    ctx = multiprocessing.get_context("fork")
    server_port = base_port
    proxy_port = base_port + 1

    proxy_p = ctx.Process(
        target=bucket_proxy,
        args=(proxy_port, server_port, bucket_s),
        daemon=True,
    )
    proxy_p.start()
    time.sleep(0.1)

    srv_p = ctx.Process(
        target=boundary_server_proc,
        args=(server_port, secret_bytes, bucket_s, tok_rate, stall_s),
        daemon=True,
    )
    srv_p.start()
    time.sleep(0.2)

    result = boundary_client("127.0.0.1", proxy_port, bucket_s, tok_rate,
                             stall_s, len(symbols))

    for p in [srv_p, proxy_p]:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    secret = "EXFIL: db_pass=hunter2 api_key=sk-abc123"
    R = 50.0
    bucket_sizes_ms = [20, 50, 100, 200, 500]

    print("=" * 85)
    print("BOUNDARY-COUNTING ATTACK: smart attacker vs bucket pacing")
    print(f"Token rate R = {R:.0f} tok/s, unlimited stall budget")
    print(f"Secret: \"{secret}\" ({len(secret)} bytes)")
    print("=" * 85)

    results = []
    for i, bms in enumerate(bucket_sizes_ms):
        B = bms / 1000
        tpb = int(R * B)
        m = tpb
        levels = m + 1
        bits_per_sym = int(math.log2(levels)) if levels >= 2 else 0
        eff_levels = 2 ** bits_per_sym if bits_per_sym > 0 else 1
        calc_bps = math.log2(eff_levels) / B if eff_levels > 1 else 0

        print(f"\n  Bucket={bms}ms: tpb={tpb}, m={m}, "
              f"levels={eff_levels} ({bits_per_sym} bits/bnd)...", end="", flush=True)

        port = 9200 + i * 10
        r = run_one(secret, bms, tok_rate=R, base_port=port)

        if "error" in r:
            print(f" ERROR: {r['error']}")
            results.append({
                "bucket_ms": bms, "calc_bps": calc_bps, "measured_bps": 0,
                "decoded": False, "error": r["error"],
            })
            continue

        ok = r["decoded_ok"]
        mbps = r["measured_bps"]
        print(f" {mbps:.0f} bps, decoded={'YES' if ok else 'NO'}")
        results.append({
            "bucket_ms": bms,
            "calc_bps": round(calc_bps, 1),
            "measured_bps": mbps,
            "decoded": ok,
            "decoded_msg": r.get("decoded_msg", ""),
            "tokens": r["tokens"],
            "buckets": r["buckets_detected"],
            "symbols": r["symbols_decoded"],
            "wall_s": r["wall_s"],
            "levels": eff_levels,
            "bits_per_sym": bits_per_sym,
        })

    # Also include PAM no-defense baseline from previous experiment
    print(f"\n  (PAM no-defense baseline: 216 bps measured, 200 bps calculated)")

    print("\n" + "=" * 85)
    print("RESULTS: calculated vs measured")
    print("=" * 85)
    print(f"\n{'Defense':>10} {'Attack':>12} {'Calc':>8} {'Meas':>8} "
          f"{'Ratio':>7} {'Decoded':>8} {'Levels':>7} {'Syms':>6}")
    print("-" * 75)

    # PAM baseline
    print(f"{'none':>10} {'16-PAM':>12} {'200.0':>6}bp {'216':>6}bp "
          f"{'1.08':>7} {'YES':>8} {'16':>7} {'—':>6}")

    for r in results:
        bucket = f"{r['bucket_ms']}ms"
        attack = f"bnd-{r.get('levels', '?')}"
        decoded = "YES" if r["decoded"] else "NO"
        ratio = r["measured_bps"] / r["calc_bps"] if r["calc_bps"] > 0 else 0
        syms = r.get("symbols", "—")
        print(f"{bucket:>10} {attack:>12} {r['calc_bps']:>6.1f}bp {r['measured_bps']:>6.0f}bp "
              f"{ratio:>7.2f} {decoded:>8} {r.get('levels', '?'):>7} {syms:>6}")

    # Naive PAM through buckets
    print()
    print(f"{'20ms':>10} {'16-PAM':>12} {'—':>8} {'0':>6}bp "
          f"{'—':>7} {'NO':>8} {'16':>7} {'—':>6}  (PAM destroyed)")
    print(f"{'100ms':>10} {'16-PAM':>12} {'—':>8} {'0':>6}bp "
          f"{'—':>7} {'NO':>8} {'16':>7} {'—':>6}  (PAM destroyed)")

    print(f"\n  Calc = log₂(levels) / B, levels = 2^floor(log₂(R·B+1))")
    print(f"  Ratio = measured / calculated")

    out = "experiments/timing_channel/boundary_attack_results.json"
    with open(out, "w") as f:
        json.dump({"config": {"tok_rate": R, "secret": secret, "stall": "unlimited"},
                   "results": results}, f, indent=2)
    print(f"\n  Results saved to {out}")


if __name__ == "__main__":
    main()
