#!/usr/bin/env python3
"""Probe maximum allocatable memory per region via binary search.

Much faster than a full wipe — tests allocation only, not data fill.
Finds the ceiling within 1 GiB precision in ~30 seconds total.

Usage:
    uv run python scripts/probe_limits.py
    uv run python scripts/probe_limits.py --disk-path /tmp --precision-gib 0.5
"""

import argparse
import ctypes
import ctypes.util
import mmap
import os
import sys
import time

GiB = 1024 ** 3
BLOCK = 4096


# ---------------------------------------------------------------------------
# HBM probe: binary search on cudaMalloc
# ---------------------------------------------------------------------------

def probe_hbm(device: int = 0) -> dict:
    """Find max allocatable GPU HBM via binary search on cudaMalloc."""
    try:
        from pose.detect import get_cuda_runtime
        rt = get_cuda_runtime()
        free, total = rt.mem_get_info(device)
    except Exception as e:
        return {"total": 0, "max_alloc": 0, "error": str(e)}

    lo, hi = 0, free
    best = 0

    while hi - lo > BLOCK:
        mid = ((lo + hi) // 2 // BLOCK) * BLOCK  # align
        try:
            ptr = rt.malloc(device, mid)
            rt.free(device, ptr)
            best = mid
            lo = mid + BLOCK
        except RuntimeError:
            hi = mid - BLOCK

    return {"total": total, "free": free, "max_alloc": best}


# ---------------------------------------------------------------------------
# DRAM probe: binary search on mmap + MAP_POPULATE
# ---------------------------------------------------------------------------

def _try_mmap_populate(size: int, numa_node: int = 0) -> bool:
    """Try to mmap and populate (fault in) `size` bytes. Returns success."""
    try:
        buf = mmap.mmap(-1, size)
        # Bind to NUMA node 0 (LPDDR5X) to avoid stealing HBM pages
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

        # MAP_POPULATE equivalent: madvise MADV_POPULATE_WRITE (Linux 5.14+)
        MADV_POPULATE_WRITE = 23
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            buf_addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
            ret = libc.madvise(ctypes.c_void_p(buf_addr), ctypes.c_ulong(size), MADV_POPULATE_WRITE)
            if ret != 0:
                # Fallback: touch one byte per page to force commit
                _touch_pages(buf, size)
        except Exception:
            _touch_pages(buf, size)

        buf.close()
        return True
    except (OSError, MemoryError, ValueError):
        return False


def _touch_pages(buf, size: int):
    """Touch one byte per 2MB hugepage to force physical allocation."""
    step = 2 * 1024 * 1024  # 2 MiB
    for offset in range(0, size, step):
        buf[offset] = 0


def probe_dram(numa_node: int = 0, precision: int = GiB) -> dict:
    """Find max allocatable DRAM on the given NUMA node via binary search."""
    from pathlib import Path
    node_path = Path(f"/sys/devices/system/node/node{numa_node}/meminfo")
    total = 0
    if node_path.exists():
        for line in node_path.read_text().splitlines():
            if "MemTotal" in line:
                total = int(line.split()[3]) * 1024
                break

    # Read current free
    free = 0
    if node_path.exists():
        for line in node_path.read_text().splitlines():
            if "MemFree" in line:
                free = int(line.split()[3]) * 1024
                break

    # Binary search: start from free, search down
    lo, hi = 0, free
    best = 0

    while hi - lo > precision:
        mid = ((lo + hi) // 2 // BLOCK) * BLOCK
        print(f"  DRAM probe: trying {mid / GiB:.1f} GiB...", end=" ", flush=True)
        t0 = time.monotonic()
        ok = _try_mmap_populate(mid, numa_node)
        elapsed = time.monotonic() - t0
        print(f"{'OK' if ok else 'FAIL'} ({elapsed:.1f}s)")
        if ok:
            best = mid
            lo = mid + precision
        else:
            hi = mid - precision

    return {"total": total, "free": free, "max_alloc": best}


# ---------------------------------------------------------------------------
# Disk probe: fallocate (instant)
# ---------------------------------------------------------------------------

def probe_disk(path: str) -> dict:
    """Find allocatable disk space via statvfs + fallocate."""
    st = os.statvfs(path)
    total = st.f_blocks * st.f_frsize
    avail = st.f_bavail * st.f_frsize

    # Verify with actual fallocate
    test_file = os.path.join(path, ".pose_probe.tmp")
    try:
        fd = os.open(test_file, os.O_RDWR | os.O_CREAT, 0o600)
        # Try to preallocate the available space
        try:
            os.posix_fallocate(fd, 0, avail)
            max_alloc = avail
        except OSError:
            # Binary search
            lo, hi = 0, avail
            max_alloc = 0
            while hi - lo > GiB:
                mid = (lo + hi) // 2
                try:
                    os.ftruncate(fd, 0)
                    os.posix_fallocate(fd, 0, mid)
                    max_alloc = mid
                    lo = mid + GiB
                except OSError:
                    hi = mid - GiB
        os.close(fd)
    finally:
        if os.path.exists(test_file):
            os.unlink(test_file)

    return {"total": total, "available": avail, "max_alloc": max_alloc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Probe memory allocation limits")
    parser.add_argument("--disk-path", default="/tmp")
    parser.add_argument("--precision-gib", type=float, default=1.0,
                        help="Binary search precision in GiB")
    args = parser.parse_args()

    precision = int(args.precision_gib * GiB)

    print("=" * 60)
    print(" Memory Allocation Probe")
    print("=" * 60)

    # HBM
    print("\n--- HBM (cudaMalloc) ---")
    t0 = time.monotonic()
    hbm = probe_hbm()
    print(f"  Total:     {hbm['total'] / GiB:.2f} GiB")
    print(f"  Free:      {hbm.get('free', 0) / GiB:.2f} GiB")
    print(f"  Max alloc: {hbm['max_alloc'] / GiB:.2f} GiB")
    print(f"  Time:      {time.monotonic() - t0:.1f}s")

    # DRAM
    print("\n--- DRAM (mmap + populate, NUMA node 0) ---")
    t0 = time.monotonic()
    dram = probe_dram(precision=precision)
    print(f"  Total:     {dram['total'] / GiB:.2f} GiB")
    print(f"  Free:      {dram['free'] / GiB:.2f} GiB")
    print(f"  Max alloc: {dram['max_alloc'] / GiB:.2f} GiB")
    print(f"  Time:      {time.monotonic() - t0:.1f}s")

    # Disk
    print(f"\n--- Disk ({args.disk_path}) ---")
    t0 = time.monotonic()
    disk = probe_disk(args.disk_path)
    print(f"  Total:     {disk['total'] / GiB:.2f} GiB")
    print(f"  Available: {disk['available'] / GiB:.2f} GiB")
    print(f"  Max alloc: {disk['max_alloc'] / GiB:.2f} GiB")
    print(f"  Time:      {time.monotonic() - t0:.1f}s")

    # Summary
    total_alloc = hbm["max_alloc"] + dram["max_alloc"] + disk["max_alloc"]
    total_phys = hbm["total"] + dram["total"] + disk["total"]
    print(f"\n{'=' * 60}")
    print(f" Max wipeable: {total_alloc / GiB:.1f} GiB / {total_phys / GiB:.1f} GiB "
          f"({total_alloc / total_phys * 100:.1f}%)" if total_phys > 0 else "")
    print(f"{'=' * 60}")

    # Output env vars
    print(f"\n# For system test:")
    print(f"export POSE_DRAM_FRAC={dram['max_alloc'] / dram['total']:.4f}" if dram['total'] > 0 else "")
    print(f"export POSE_HBM_FRAC={hbm['max_alloc'] / hbm.get('free', 1):.4f}" if hbm.get('free', 0) > 0 else "")
    print(f"export POSE_DISK_FRAC={disk['max_alloc'] / disk['available']:.4f}" if disk['available'] > 0 else "")


if __name__ == "__main__":
    main()
