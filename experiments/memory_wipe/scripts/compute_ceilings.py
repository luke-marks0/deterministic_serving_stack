"""Compute exact wipeable memory ceilings from hardware queries.

HBM: cudaMemGetInfo() — exact
Disk: statvfs() — exact
DRAM: NUMA node0 MemFree minus estimated overhead — approximate
"""
import os

GiB = 1024**3

# --- HBM: exact via cudaMemGetInfo ---
try:
    import cupy
    free_hbm, total_hbm = cupy.cuda.runtime.memGetInfo()
    hbm_safety = 512 * 1024**2  # 512 MiB for CUDA driver internals
    hbm_wipeable = free_hbm - hbm_safety
except Exception as e:
    print(f"HBM: {e}")
    free_hbm = total_hbm = hbm_wipeable = 0

# --- DRAM: from NUMA node 0 (LPDDR5X on GH200) ---
node0_total_kb = node0_free_kb = 0
try:
    with open("/sys/devices/system/node/node0/meminfo") as f:
        for line in f:
            if "MemTotal" in line:
                node0_total_kb = int(line.split()[3])
            if "MemFree" in line:
                node0_free_kb = int(line.split()[3])
except FileNotFoundError:
    pass
node0_free = node0_free_kb * 1024
node0_total = node0_total_kb * 1024
pte_overhead = int(node0_free * 0.002)  # ~8 bytes per 4KB page
kernel_reserve = 3 * GiB
process_reserve = 2 * GiB  # python + crypto + batch buffers
dram_wipeable = node0_free - pte_overhead - kernel_reserve - process_reserve

# --- Disk: exact via statvfs ---
st = os.statvfs("/")
disk_avail = st.f_bavail * st.f_frsize
disk_reserve = 25 * GiB
disk_wipeable = disk_avail - disk_reserve

total = dram_wipeable + hbm_wipeable + disk_wipeable

print(f"Region    {'Total':>10} {'Free':>10} {'Wipeable':>10}  Method")
print("-" * 75)
print(f"DRAM      {node0_total/GiB:>9.1f}G {node0_free/GiB:>9.1f}G {dram_wipeable/GiB:>9.1f}G  node0 MemFree - 5G overhead")
print(f"HBM       {total_hbm/GiB:>9.1f}G {free_hbm/GiB:>9.1f}G {hbm_wipeable/GiB:>9.1f}G  cudaMemGetInfo() - 512 MiB")
print(f"Disk      {disk_avail/GiB:>9.1f}G {disk_avail/GiB:>9.1f}G {disk_wipeable/GiB:>9.1f}G  statvfs(/) - 25 GiB")
print("-" * 75)
print(f"TOTAL     {'':>10} {'':>10} {total/GiB:>9.1f}G")
print()
print("# System test env vars:")
print(f"export POSE_DRAM_TOTAL_GB={node0_total/GiB:.1f}")
print(f"export POSE_DRAM_GB={dram_wipeable/GiB:.1f}")
print(f"export POSE_HBM_TOTAL_GB={total_hbm/GiB:.1f}")
print(f"export POSE_HBM_GB={hbm_wipeable/GiB:.1f}")
print(f"export POSE_NVME_TOTAL_GB={disk_avail/GiB:.1f}")
print(f"export POSE_NVME_GB={disk_wipeable/GiB:.1f}")
