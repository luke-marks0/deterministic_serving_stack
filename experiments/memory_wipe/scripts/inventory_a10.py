#!/usr/bin/env python3
"""Memory inventory for non-GH200 instances."""
import os, ctypes

GiB = 1024**3
MiB = 1024**2

# DRAM
with open("/proc/meminfo") as f:
    for line in f:
        if line.startswith("MemTotal:"): dram_total = int(line.split()[1]) * 1024
        if line.startswith("MemAvailable:"): dram_avail = int(line.split()[1]) * 1024

# HBM
cuda = ctypes.CDLL("libcudart.so")
cuda.cudaSetDevice(0)
free_c = ctypes.c_size_t()
total_c = ctypes.c_size_t()
cuda.cudaMemGetInfo(ctypes.byref(free_c), ctypes.byref(total_c))
hbm_total = total_c.value
hbm_free = free_c.value

# Disk
st = os.statvfs("/")
disk_total = st.f_blocks * st.f_frsize
disk_avail = st.f_bavail * st.f_frsize
disk_used = (st.f_blocks - st.f_bfree) * st.f_frsize

print("=" * 60)
print(" A10 Addressable Memory Inventory")
print("=" * 60)
print()
print("  DRAM (x86 Xeon)")
print(f"    Total:       {dram_total/GiB:.1f} GiB")
print(f"    Available:   {dram_avail/GiB:.1f} GiB")
print()
print("  GPU HBM (A10)")
print(f"    Total:       {hbm_total/GiB:.1f} GiB  ({hbm_total//MiB} MiB)")
print(f"    Free:        {hbm_free/GiB:.1f} GiB  ({hbm_free//MiB} MiB)")
print(f"    Driver:      {(hbm_total-hbm_free)//MiB} MiB")
print()
print("  Disk (virtio)")
print(f"    Total:       {disk_total/GiB:.1f} GiB")
print(f"    Available:   {disk_avail/GiB:.1f} GiB")
print(f"    Used:        {disk_used/GiB:.1f} GiB")
print()

hdr = f"  {'Region':<20s} {'Total':>10s}  {'Wipeable':>10s}  {'Reserved':>10s}"
sep = f"  {'-'*20} {'-'*10}  {'-'*10}  {'-'*10}"
print("=" * 60)
print(hdr)
print(sep)

dram_res = dram_total - dram_avail
print(f"  {'DRAM':<20s} {dram_total/GiB:>9.1f}G  {dram_avail/GiB:>9.1f}G  {dram_res/GiB:>9.1f}G")
hbm_res = hbm_total - hbm_free
print(f"  {'HBM (A10)':<20s} {hbm_total/GiB:>9.1f}G  {hbm_free/GiB:>9.1f}G  {hbm_res/GiB:>9.1f}G")
print(f"  {'Disk':<20s} {disk_total/GiB:>9.1f}G  {disk_avail/GiB:>9.1f}G  {disk_used/GiB:>9.1f}G")

total = dram_total + hbm_total + disk_total
wipeable = dram_avail + hbm_free + disk_avail
print(sep)
print(f"  {'TOTAL':<20s} {total/GiB:>9.1f}G  {wipeable/GiB:>9.1f}G  {(total-wipeable)/GiB:>9.1f}G")
print(f"  Coverage potential: {wipeable/total*100:.1f}%")
print("=" * 60)
