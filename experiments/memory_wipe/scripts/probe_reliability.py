#!/usr/bin/env python3
"""Test allocation reliability at various fractions of the probed ceiling.

For each region, tries allocating at -5% to +10% of the known max,
10 repetitions each. Reports success rate per level.

Usage:
    uv run python scripts/probe_reliability.py
"""

import ctypes
import ctypes.util
import mmap
import os
import sys
import time

GiB = 1024 ** 3
BLOCK = 4096

# Probed ceilings from probe_limits.py (GiB)
HBM_MAX = 92.96 * GiB
DRAM_MAX = 423.13 * GiB
DISK_MAX = 3943.77 * GiB

OFFSETS_PCT = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
REPS = 10


def _align(size: int) -> int:
    return (size // BLOCK) * BLOCK


# ---------------------------------------------------------------------------
# HBM
# ---------------------------------------------------------------------------

def try_hbm(size: int, device: int = 0) -> bool:
    try:
        from pose.detect import get_cuda_runtime
        rt = get_cuda_runtime()
        ptr = rt.malloc(device, size)
        rt.free(device, ptr)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# DRAM
# ---------------------------------------------------------------------------

def try_dram(size: int, numa_node: int = 0) -> bool:
    try:
        buf = mmap.mmap(-1, size)
        # NUMA bind
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            MPOL_BIND, MPOL_MF_MOVE = 2, 2
            nodemask = ctypes.c_ulong(1 << numa_node)
            buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            libc.mbind(
                ctypes.c_void_p(buf_addr), ctypes.c_ulong(size),
                ctypes.c_int(MPOL_BIND), ctypes.byref(nodemask),
                ctypes.c_ulong(64), ctypes.c_uint(MPOL_MF_MOVE),
            )
        except Exception:
            pass

        # Populate: touch one byte per 2 MiB to force physical allocation
        step = 2 * 1024 * 1024
        for offset in range(0, size, step):
            buf[offset] = 0

        buf.close()
        return True
    except (OSError, MemoryError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Disk
# ---------------------------------------------------------------------------

def try_disk(size: int, path: str = "/tmp") -> bool:
    fpath = os.path.join(path, f".pose_rel_probe_{os.getpid()}.tmp")
    try:
        fd = os.open(fpath, os.O_RDWR | os.O_CREAT, 0o600)
        os.posix_fallocate(fd, 0, size)
        os.close(fd)
        os.unlink(fpath)
        return True
    except OSError:
        try:
            os.close(fd)
        except Exception:
            pass
        if os.path.exists(fpath):
            os.unlink(fpath)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_region(name: str, base_max: int, try_fn, reps: int = REPS):
    print(f"\n{'='*60}")
    print(f" {name}: base max = {base_max / GiB:.2f} GiB, {reps} reps per level")
    print(f"{'='*60}")
    print(f" {'Offset':>7s}  {'Size (GiB)':>11s}  {'Pass':>4s}/{reps:<4d}  {'Rate':>6s}  {'Avg(s)':>7s}")
    print(f" {'-'*7}  {'-'*11}  {'-'*9}  {'-'*6}  {'-'*7}")

    for pct in OFFSETS_PCT:
        size = _align(int(base_max * (1 + pct / 100)))
        if size <= 0:
            continue

        passes = 0
        total_time = 0
        for rep in range(reps):
            t0 = time.monotonic()
            ok = try_fn(size)
            elapsed = time.monotonic() - t0
            total_time += elapsed
            if ok:
                passes += 1

        avg_time = total_time / reps
        rate = passes / reps * 100
        marker = "<<<" if 0 < passes < reps else ""
        print(f" {pct:>+5d}%   {size / GiB:>10.2f}   {passes:>4d}/{reps:<4d}  {rate:>5.0f}%  {avg_time:>6.1f}s  {marker}")

    print()


def main():
    print("Reliability probe: testing allocation at various fractions of ceiling")
    print(f"Offsets: {OFFSETS_PCT}")
    print(f"Reps per level: {REPS}")

    # Reset GPU first
    try:
        from pose.detect import get_cuda_runtime
        get_cuda_runtime()
        print("\nGPU available — testing HBM")
        run_region("HBM", HBM_MAX, try_hbm)
    except Exception as e:
        print(f"\nGPU not available ({e}) — skipping HBM")

    print("\nTesting DRAM (this will take a while)...")
    run_region("DRAM", DRAM_MAX, try_dram)

    print("\nTesting Disk...")
    run_region("Disk", DISK_MAX, try_disk)


if __name__ == "__main__":
    main()
