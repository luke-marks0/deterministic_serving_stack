#!/usr/bin/env python3
"""Full addressable memory inventory for GH200."""
import os, ctypes

GiB = 1024**3
MiB = 1024**2

print("=" * 65)
print(" GH200 Addressable Memory Inventory")
print("=" * 65)

# DRAM: NUMA node 0 (LPDDR5X)
node0_total = node0_free = 0
with open("/sys/devices/system/node/node0/meminfo") as f:
    for line in f:
        if "MemTotal" in line:
            node0_total = int(line.split()[3]) * 1024
        if "MemFree" in line:
            node0_free = int(line.split()[3]) * 1024

# DRAM: NUMA node 1 (HBM mapped as system memory)
node1_total = 0
with open("/sys/devices/system/node/node1/meminfo") as f:
    for line in f:
        if "MemTotal" in line:
            node1_total = int(line.split()[3]) * 1024

# GPU HBM via cudaMemGetInfo
try:
    cuda = ctypes.CDLL("libcudart.so")
    cuda.cudaSetDevice(0)
    free_c = ctypes.c_size_t()
    total_c = ctypes.c_size_t()
    cuda.cudaMemGetInfo(ctypes.byref(free_c), ctypes.byref(total_c))
    hbm_total = total_c.value
    hbm_free = free_c.value
except Exception:
    hbm_total = hbm_free = 0

# Disk
st = os.statvfs("/")
disk_total = st.f_blocks * st.f_frsize
disk_avail = st.f_bavail * st.f_frsize
disk_used = disk_total - st.f_bfree * st.f_frsize

# System-wide
system_total = 0
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemTotal:"):
            system_total = int(line.split()[1]) * 1024

print()
print("  SYSTEM VIEW (/proc/meminfo)")
print(f"    MemTotal:       {system_total / GiB:.1f} GiB  (both NUMA nodes)")
print()
print("  NUMA NODE 0 -- LPDDR5X (CPU)")
print(f"    Total:          {node0_total / GiB:.1f} GiB")
print(f"    Free:           {node0_free / GiB:.1f} GiB")
print()
print("  NUMA NODE 1 -- HBM3 (mapped as system memory)")
print(f"    Total:          {node1_total / GiB:.1f} GiB")
print()
print("  GPU HBM (cudaMemGetInfo)")
print(f"    Total:          {hbm_total / GiB:.1f} GiB  ({hbm_total // MiB} MiB)")
print(f"    Free:           {hbm_free / GiB:.1f} GiB  ({hbm_free // MiB} MiB)")
print(f"    Driver reserve: {(hbm_total - hbm_free) // MiB} MiB")
print()
print("  DISK (/dev/vda1, ext4, virtio)")
print(f"    Total:          {disk_total / GiB:.1f} GiB")
print(f"    Available:      {disk_avail / GiB:.1f} GiB")
print(f"    Used:           {disk_used / GiB:.1f} GiB")

# Probed allocation ceilings
hbm_max_alloc = 92.96 * GiB  # from reliability probe: 100% at -1%, 0% at +0%
dram_max_alloc = 423.13 * GiB  # from probe_limits.py
disk_max_alloc = disk_avail  # statvfs is exact

total_addressable = node0_total + hbm_total + disk_total
total_wipeable = dram_max_alloc + hbm_max_alloc + disk_max_alloc

print()
print("=" * 65)
print("  SUMMARY")
print("=" * 65)
row = "  {:<22s} {:>10s}  {:>10s}  {:>10s}  {:>6s}"
print(row.format("Region", "Total", "Wipeable", "Reserved", ""))
print(row.format("-" * 22, "-" * 10, "-" * 10, "-" * 10, "-" * 6))

dram_res = node0_total - dram_max_alloc
print(row.format("LPDDR5X (NUMA 0)",
    f"{node0_total/GiB:.1f} GiB",
    f"{dram_max_alloc/GiB:.1f} GiB",
    f"{dram_res/GiB:.1f} GiB",
    f"{dram_max_alloc/node0_total*100:.1f}%"))

hbm_res = hbm_total - hbm_max_alloc
print(row.format("HBM3 (GPU)",
    f"{hbm_total/GiB:.1f} GiB",
    f"{hbm_max_alloc/GiB:.1f} GiB",
    f"{hbm_res/GiB:.1f} GiB",
    f"{hbm_max_alloc/hbm_total*100:.1f}%"))

disk_res = disk_total - disk_max_alloc
print(row.format("Disk (virtio)",
    f"{disk_total/GiB:.1f} GiB",
    f"{disk_max_alloc/GiB:.1f} GiB",
    f"{disk_res/GiB:.1f} GiB",
    f"{disk_max_alloc/disk_total*100:.1f}%"))

print(row.format("-" * 22, "-" * 10, "-" * 10, "-" * 10, "-" * 6))
print(row.format("TOTAL",
    f"{total_addressable/GiB:.1f} GiB",
    f"{total_wipeable/GiB:.1f} GiB",
    f"{(total_addressable-total_wipeable)/GiB:.1f} GiB",
    f"{total_wipeable/total_addressable*100:.1f}%"))
print("=" * 65)
