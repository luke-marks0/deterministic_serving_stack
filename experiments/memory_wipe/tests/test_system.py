"""System test -- run on GH200 only.

Usage:
    uv run pytest tests/test_system.py -v -s

Auto-detects memory ceilings from hardware queries (NUMA nodes,
cudaMemGetInfo, statvfs). No env vars needed for memory sizes.

Override fractions with env vars if desired:
    POSE_DRAM_FRAC=0.85  POSE_HBM_FRAC=0.90  POSE_DISK_FRAC=0.95
"""

import os
import json
import pytest
from pose.detect import compute_ceilings
from pose.protocol import run_protocol
from pose.report import generate_report
from pose.memory.dram import DramRegion
from pose.memory.hbm import HbmRegion
from pose.memory.nvme import NvmeRegion

BLOCK_SIZE = 4096
DISK_PATH = os.environ.get("POSE_DISK_PATH", "/tmp")
NUM_ROUNDS = int(os.environ.get("POSE_ROUNDS", "1000"))
REPORT_DIR = os.environ.get("POSE_REPORT_DIR", "reports")

# Target fractions: how much of each detected region to wipe.
# Generous defaults — increase these to tighten coverage later.
DRAM_FRAC = float(os.environ.get("POSE_DRAM_FRAC", "0.85"))
HBM_FRAC = float(os.environ.get("POSE_HBM_FRAC", "0.90"))
DISK_FRAC = float(os.environ.get("POSE_DISK_FRAC", "0.50"))  # conservative for first run


@pytest.mark.slow
def test_full_wipe():
    # --- Auto-detect ceilings ---
    ceil = compute_ceilings(
        disk_path=DISK_PATH,
        dram_fraction=DRAM_FRAC,
        hbm_fraction=HBM_FRAC,
        disk_fraction=DISK_FRAC,
    )

    print(f"\n--- Detected ceilings ---")
    print(f"DRAM:  {ceil.dram_wipeable / (1024**3):.1f} GiB of {ceil.dram_total / (1024**3):.1f} GiB (frac={DRAM_FRAC})")
    print(f"HBM:   {ceil.hbm_wipeable / (1024**3):.1f} GiB of {ceil.hbm_total / (1024**3):.1f} GiB (frac={HBM_FRAC})")
    print(f"Disk:  {ceil.disk_wipeable / (1024**3):.1f} GiB of {ceil.disk_total / (1024**3):.1f} GiB (frac={DISK_FRAC})")
    print(f"Total: {ceil.total_wipeable / (1024**3):.1f} GiB")

    # --- Allocate regions ---
    nvme_file = os.path.join(DISK_PATH, "pose_wipe.bin")

    dram = DramRegion(size_bytes=ceil.dram_wipeable, block_size=BLOCK_SIZE)
    hbm = HbmRegion(size_bytes=ceil.hbm_wipeable, block_size=BLOCK_SIZE)
    nvme = NvmeRegion(nvme_file, num_blocks=ceil.disk_wipeable // BLOCK_SIZE,
                      block_size=BLOCK_SIZE)

    regions = {"dram": dram, "hbm": hbm, "nvme": nvme}
    region_info = {
        "dram": {
            "total_bytes": ceil.dram_total,
            "reserved_bytes": ceil.dram_reserved,
            "reserved_reason": ceil.dram_reserved_reason,
        },
        "hbm": {
            "total_bytes": ceil.hbm_total,
            "reserved_bytes": ceil.hbm_reserved,
            "reserved_reason": ceil.hbm_reserved_reason,
        },
        "nvme": {
            "total_bytes": ceil.disk_total,
            "reserved_bytes": ceil.disk_reserved,
            "reserved_reason": ceil.disk_reserved_reason,
        },
    }

    # --- Run protocol ---
    result = run_protocol(
        regions=regions,
        region_info=region_info,
        block_size=BLOCK_SIZE,
        num_rounds=NUM_ROUNDS,
    )

    # --- Generate report ---
    os.makedirs(REPORT_DIR, exist_ok=True)
    report = generate_report(result)

    report_path = os.path.join(REPORT_DIR, "wipe_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print human-readable summary
    print(f"\n{'='*60}")
    print(f" PoSE-DB Phase 1 Wipe Report")
    print(f"{'='*60}")
    print(f" Protocol:       Unconditional PoSE-DB (AES-256-CTR)")
    print(f" Result:         {'PASS' if result.passed else 'FAIL'}")
    print(f" Rounds:         {result.rounds_passed}/{result.rounds_total}")
    print(f"")
    print(f" --- Memory Inventory ---")
    for rm in result.region_metrics:
        print(f" {rm.name.upper():6s}  total={rm.total_bytes/(1024**3):.1f} GiB"
              f"  wiped={rm.wiped_bytes/(1024**3):.1f} GiB"
              f"  reserved={rm.reserved_bytes/(1024**3):.1f} GiB")
        print(f"         reason: {rm.reserved_reason[:80]}...")
    print(f"")
    print(f" --- Key Metrics ---")
    print(f" Total memory:   {result.bytes_total/(1024**3):.1f} GiB")
    print(f" Bytes wiped:    {result.bytes_wiped/(1024**3):.1f} GiB")
    print(f" Coverage:       {result.coverage*100:.2f}%")
    print(f" Wipe time:      {result.fill_time_s:.1f}s")
    for rm in result.region_metrics:
        print(f"   {rm.name:6s}       {rm.fill_time_s:.1f}s"
              f"  ({rm.fill_throughput_gbps:.1f} GiB/s)")
    print(f" Verify time:    {result.verify_time_s:.3f}s")
    print(f" Resume time:    {result.resume_time_s:.3f}s")
    print(f"{'='*60}")
    print(f" Report written to: {report_path}")
    print(f"{'='*60}")

    assert result.passed, f"Failed {result.rounds_total - result.rounds_passed} rounds"

    if os.path.exists(nvme_file):
        os.unlink(nvme_file)
