#!/usr/bin/env python3
"""Measure residual covert channel capacity under bucket pacing defense.

Runs the 16-PAM attacker through a UDP bucket-pacing proxy at various bucket
intervals and measures what the receiver can actually decode.

Usage:
    python experiments/timing_channel/bucket_experiment.py
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


# ---------------------------------------------------------------------------
# Reuse core channel code from covert_channel_demo.py
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    levels: int = 16
    base_delay_s: float = 0.020
    step_s: float = 0.0005
    calibration_symbols: int = 32
    repetitions: int = 3

    @property
    def bits_per_token(self) -> float:
        return math.log2(self.levels)

    @property
    def modulation_range_ms(self) -> float:
        return self.step_s * (self.levels - 1) * 1000

    def symbol_to_delay(self, symbol: int) -> float:
        return self.base_delay_s + self.step_s * symbol

    def delay_to_symbol(self, delay_s: float, base: float, step: float) -> int:
        if step <= 0:
            return 0
        raw = (delay_s - base) / step
        return max(0, min(self.levels - 1, round(raw)))


WORDS = (
    "the model generates tokens sequentially based on attention patterns and "
    "learned representations from training data which enables coherent text "
    "generation across many different domains including tasks like summarization"
).split()


def bytes_to_symbols(data: bytes, bits_per_sym: int) -> list[int]:
    bits = "".join(f"{b:08b}" for b in data)
    while len(bits) % bits_per_sym:
        bits += "0"
    return [int(bits[i:i + bits_per_sym], 2) for i in range(0, len(bits), bits_per_sym)]


def symbols_to_bytes(symbols: list[int], bits_per_sym: int, nbytes: int) -> bytes:
    bits = "".join(f"{s:0{bits_per_sym}b}" for s in symbols)
    return bytes(int(bits[i:i+8], 2) for i in range(0, nbytes * 8, 8))


def frame_message(secret: bytes, cfg: ChannelConfig) -> list[int]:
    length_bytes = struct.pack(">I", len(secret))
    checksum = hashlib.sha256(secret).digest()[:4]
    cal = [0 if i % 2 == 0 else cfg.levels - 1 for i in range(cfg.calibration_symbols)]
    payload_syms = bytes_to_symbols(length_bytes + secret + checksum, int(cfg.bits_per_token))
    return cal + payload_syms


def decode_message(symbols: list[int], cfg: ChannelConfig) -> tuple[bytes, bool]:
    bps = int(cfg.bits_per_token)
    data_syms = symbols[cfg.calibration_symbols:]
    if len(data_syms) < math.ceil(8 * 8 / bps):
        return b"", False
    len_n = math.ceil(4 * 8 / bps)
    msg_len = struct.unpack(">I", symbols_to_bytes(data_syms[:len_n], bps, 4))[0]
    if msg_len > 10000:
        return b"", False
    total = 4 + msg_len + 4
    need = math.ceil(total * 8 / bps)
    if len(data_syms) < need:
        return b"", False
    raw = symbols_to_bytes(data_syms[:need], bps, total)
    msg = raw[4:4 + msg_len]
    got_ck = raw[4 + msg_len:4 + msg_len + 4]
    want_ck = hashlib.sha256(msg).digest()[:4]
    return msg, got_ck == want_ck


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0

def _median(xs):
    if not xs: return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2

def _stddev(xs):
    if len(xs) < 2: return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Bucket-pacing UDP proxy
# ---------------------------------------------------------------------------

def bucket_proxy(listen_port: int, server_port: int, bucket_s: float):
    """UDP proxy that accumulates packets and releases them at fixed intervals."""
    proxy = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    proxy.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    proxy.bind(("0.0.0.0", listen_port))
    proxy.settimeout(60.0)

    # Wait for client "start" message
    data, client_addr = proxy.recvfrom(64)

    # Forward "start" to server
    proxy.sendto(data, ("127.0.0.1", server_port))

    # Accumulate and release loop
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

    # Flush remaining
    time.sleep(0.01)
    with lock:
        for pkt in buffer:
            proxy.sendto(pkt, client_addr)
        buffer.clear()

    recv_thread.join(timeout=2)
    proxy.close()


# ---------------------------------------------------------------------------
# Server process
# ---------------------------------------------------------------------------

def server_proc(port: int, secret_bytes: bytes, cfg_dict: dict):
    cfg = ChannelConfig(**cfg_dict)
    symbols = frame_message(secret_bytes, cfg)
    srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.settimeout(30.0)

    data, addr = srv.recvfrom(64)

    for rep in range(cfg.repetitions):
        for i, symbol in enumerate(symbols):
            idx = rep * len(symbols) + i
            token = WORDS[idx % len(WORDS)]
            obj = {"t": token, "i": idx, "lp": round(-0.5 - symbol * 0.01, 3)}
            pkt = json.dumps(obj, separators=(",", ":")).encode()
            delay = cfg.symbol_to_delay(symbol)
            deadline = time.perf_counter() + delay
            while time.perf_counter() < deadline:
                pass
            srv.sendto(pkt, addr)

    srv.sendto(b'{"done":true}', addr)
    srv.close()


# ---------------------------------------------------------------------------
# Client (decoder) — adapted for bucket-paced input
# ---------------------------------------------------------------------------

def client_decode(host: str, port: int, cfg: ChannelConfig) -> dict:
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
        return {"error": "too few tokens", "tokens_received": len(timestamps)}

    delays = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    cal_n = cfg.calibration_symbols
    bps = int(cfg.bits_per_token)

    if len(delays) < cal_n:
        return {"error": "insufficient data", "tokens_received": len(timestamps)}

    low_delays = [delays[k] for k in range(cal_n) if k % 2 == 1]
    high_delays = [delays[k] for k in range(cal_n) if k % 2 == 0]
    base = _median(low_delays)
    top = _median(high_delays)
    step = (top - base) / max(1, cfg.levels - 1)
    low_jit = _stddev(low_delays) * 1e6
    high_jit = _stddev(high_delays) * 1e6

    cleaned = []
    carry = 0.0
    for d in delays:
        d += carry
        carry = 0.0
        if d < base * 0.3 if base > 0 else d < 0.001:
            carry = d
            cleaned.append(None)
        else:
            cleaned.append(d)

    all_syms = [0]
    for d in cleaned:
        if d is None:
            all_syms.append(0)
        else:
            all_syms.append(cfg.delay_to_symbol(d, base=base, step=step))

    cal_expected = [0 if i % 2 == 0 else cfg.levels - 1 for i in range(cal_n)]
    cal_errors = sum(1 for a, b in zip(cal_expected, all_syms[:cal_n]) if a != b)

    syms_per_rep = len(all_syms) // cfg.repetitions if cfg.repetitions > 0 else len(all_syms)
    first_rep = all_syms[:syms_per_rep]
    single_msg, single_ok = decode_message(first_rep, cfg)

    if cfg.repetitions > 1 and syms_per_rep > 0:
        reps = [all_syms[r * syms_per_rep:(r + 1) * syms_per_rep]
                for r in range(cfg.repetitions)]
        voted = []
        for pos in range(syms_per_rep):
            cands = [reps[r][pos] for r in range(len(reps)) if pos < len(reps[r])]
            voted.append(max(set(cands), key=cands.count))
        voted_msg, voted_ok = decode_message(voted, cfg)
    else:
        voted_msg, voted_ok = single_msg, single_ok

    wall = timestamps[-1] - timestamps[0]
    payload_delays = delays[cal_n:]
    total_bits = len(payload_delays) * bps

    # Compute effective capacity from symbol error rate
    # Use calibration SER as proxy (known symbols → direct measurement)
    ser = cal_errors / cal_n if cal_n > 0 else 0.5
    # Binary entropy
    h_ser = 0.0
    if 0 < ser < 1:
        h_ser = -ser * math.log2(ser) - (1 - ser) * math.log2(1 - ser)
    elif ser >= 1:
        h_ser = 0.0  # all wrong = still informative (invert)
    effective_bps_per_tok = max(0, bps * (1 - h_ser))
    tok_rate = len(timestamps) / wall if wall > 0 else 0
    effective_bps = effective_bps_per_tok * tok_rate

    return {
        "tokens_received": len(timestamps),
        "wall_time_s": round(wall, 3),
        "calibration": {
            "base_ms": round(base * 1000, 4),
            "top_ms": round(top * 1000, 4),
            "step_us": round(step * 1e6, 2),
            "low_jitter_us": round(low_jit, 2),
            "high_jitter_us": round(high_jit, 2),
            "errors": cal_errors,
            "error_rate": round(cal_errors / cal_n, 4) if cal_n > 0 else 0,
        },
        "channel": {
            "bits_per_token": bps,
            "total_covert_bits": total_bits,
            "bitrate_bps": round(total_bits / wall, 1) if wall > 0 else 0,
            "tok_per_sec": round(len(timestamps) / wall, 1) if wall > 0 else 0,
        },
        "decoding": {
            "single_ok": single_ok,
            "single_msg": single_msg.decode("utf-8", errors="replace") if single_msg else "",
            "voted_ok": voted_ok,
            "voted_msg": voted_msg.decode("utf-8", errors="replace") if voted_msg else "",
        },
        "effective": {
            "ser": round(ser, 4),
            "h_ser": round(h_ser, 4),
            "effective_bps": round(effective_bps, 1),
        },
    }


# ---------------------------------------------------------------------------
# Run one experiment: server → [optional proxy] → client
# ---------------------------------------------------------------------------

def run_one(cfg: ChannelConfig, secret: str, bucket_ms: float | None,
            base_port: int = 9100) -> dict:
    secret_bytes = secret.encode("utf-8")
    cfg_dict = {
        "levels": cfg.levels, "base_delay_s": cfg.base_delay_s,
        "step_s": cfg.step_s, "calibration_symbols": cfg.calibration_symbols,
        "repetitions": cfg.repetitions,
    }

    ctx = multiprocessing.get_context("fork")
    server_port = base_port
    client_port = base_port  # client connects to server directly

    procs = []

    if bucket_ms is not None:
        proxy_port = base_port + 1
        client_port = proxy_port
        bucket_s = bucket_ms / 1000.0
        proxy_p = ctx.Process(
            target=bucket_proxy,
            args=(proxy_port, server_port, bucket_s),
            daemon=True,
        )
        proxy_p.start()
        procs.append(proxy_p)
        time.sleep(0.1)

    srv_p = ctx.Process(
        target=server_proc,
        args=(server_port, secret_bytes, cfg_dict),
        daemon=True,
    )
    srv_p.start()
    procs.append(srv_p)
    time.sleep(0.2)

    result = client_decode("127.0.0.1", client_port, cfg)

    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()

    return result


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

PRESETS = {
    "conservative": ChannelConfig(levels=2, base_delay_s=0.015, step_s=0.010,
                                  calibration_symbols=32, repetitions=3),
    "moderate":     ChannelConfig(levels=4, base_delay_s=0.010, step_s=0.005,
                                  calibration_symbols=48, repetitions=5),
    "aggressive":   ChannelConfig(levels=8, base_delay_s=0.008, step_s=0.003,
                                  calibration_symbols=64, repetitions=7),
    "maximum":      ChannelConfig(levels=16, base_delay_s=0.005, step_s=0.002,
                                  calibration_symbols=64, repetitions=9),
}


def main():
    secret = "EXFIL: db_pass=hunter2 api_key=sk-abc123"
    cfg = PRESETS["maximum"]  # 16-PAM, the strongest attacker
    bucket_sizes = [None, 10, 20, 50, 100, 200, 500]

    print("=" * 85)
    print("BUCKET PACING EXPERIMENT: measured vs calculated")
    print(f"Attacker: 16-PAM, base={cfg.base_delay_s*1e3:.0f}ms, "
          f"step={cfg.step_s*1e6:.0f}µs, {cfg.repetitions} reps")
    print(f"Secret: \"{secret}\" ({len(secret)} bytes)")
    print("=" * 85)

    R = 50  # approximate tok/s

    results = []
    for i, bucket_ms in enumerate(bucket_sizes):
        label = "none" if bucket_ms is None else f"{bucket_ms}ms"
        print(f"\n  Running bucket={label}...", end="", flush=True)

        port = 9100 + i * 10
        r = run_one(cfg, secret, bucket_ms, base_port=port)

        ch = r.get("channel", {})
        dec = r.get("decoding", {})
        cal = r.get("calibration", {})
        measured_bps = ch.get("bitrate_bps", 0)
        decoded_ok = dec.get("voted_ok") or dec.get("single_ok")

        # Calculated capacity
        if bucket_ms is None:
            calc_bps = math.log2(cfg.levels) / (cfg.base_delay_s + cfg.step_s * (cfg.levels - 1) / 2)
        else:
            B = bucket_ms / 1000
            # unlimited stall (attacker has full control)
            m = int(R * B)
            calc_bps = math.log2(m + 1) / B if m > 0 else 0

        eff = r.get("effective", {})
        eff_bps = eff.get("effective_bps", 0)
        ser = eff.get("ser", 0)

        results.append({
            "bucket_ms": bucket_ms,
            "measured_bps": measured_bps,
            "effective_bps": eff_bps,
            "calc_bps": calc_bps,
            "decoded": decoded_ok,
            "ser": ser,
            "cal_error_rate": cal.get("error_rate", 0),
            "tokens": r.get("tokens_received", 0),
            "wall_s": r.get("wall_time_s", 0),
        })
        print(f" eff={eff_bps:.1f} bps, SER={ser:.0%}, decoded={'YES' if decoded_ok else 'NO'}")

    # Summary table
    print("\n" + "=" * 90)
    print("RESULTS: 16-PAM attacker vs bucket pacing")
    print("=" * 90)
    print(f"\n{'Bucket':>8} {'Attmpt':>8} {'Eff bps':>8} {'Calc':>8} "
          f"{'SER':>6} {'Decoded':>8} {'Tokens':>7} {'Wall':>6}")
    print("-" * 70)
    for r in results:
        bucket = "none" if r["bucket_ms"] is None else f"{r['bucket_ms']}ms"
        decoded = "YES" if r["decoded"] else "NO"
        print(f"{bucket:>8} {r['measured_bps']:>6.0f}bp {r['effective_bps']:>6.1f}bp "
              f"{r['calc_bps']:>6.1f}bp {r['ser']:>5.0%} {decoded:>8} "
              f"{r['tokens']:>7} {r['wall_s']:>5.1f}s")

    print(f"\n  Attmpt  = raw bits attacker tried to send (meaningless under bucketing)")
    print(f"  Eff bps = effective capacity: bps × (1 − H(SER)), from calibration symbol errors")
    print(f"  Calc    = analytic: log₂(M)/T_s (none) or log₂(R·B+1)/B (bucketed), R={R}")
    print(f"  SER     = symbol error rate on calibration preamble (50% = random)")
    print(f"  Decoded = secret recovered via majority vote")

    # Save results
    out_path = "experiments/timing_channel/bucket_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump({"config": {"levels": cfg.levels, "base_delay_s": cfg.base_delay_s,
                              "step_s": cfg.step_s, "repetitions": cfg.repetitions,
                              "secret": secret, "token_rate_approx": R},
                   "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
