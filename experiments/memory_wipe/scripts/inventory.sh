#!/usr/bin/env bash
set -euo pipefail

# Run on the GH200 instance to inventory all addressable memory.
# Usage: ssh ubuntu@<ip> 'bash -s' < scripts/inventory.sh

echo "========================================="
echo " GH200 Memory Inventory"
echo "========================================="

echo ""
echo "--- CPU Info ---"
lscpu | grep -E 'Model name|Architecture|CPU\(s\)|Thread|Core'

echo ""
echo "--- Host DRAM ---"
free -h
echo ""
grep -E 'MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal' /proc/meminfo

echo ""
echo "--- NUMA Topology ---"
numactl --hardware 2>/dev/null || echo "(numactl not installed)"

echo ""
echo "--- GPU Info ---"
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version \
    --format=csv,noheader 2>/dev/null || echo "(nvidia-smi not found)"

echo ""
echo "--- GPU Memory Detail ---"
nvidia-smi 2>/dev/null || true

echo ""
echo "--- Block Devices ---"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE

echo ""
echo "--- NVMe Drives ---"
if command -v nvme &>/dev/null; then
    nvme list
else
    echo "(nvme-cli not installed — install with: sudo apt install nvme-cli)"
fi

echo ""
echo "--- Disk Usage ---"
df -h

echo ""
echo "--- Hugepages ---"
grep -i huge /proc/meminfo

echo ""
echo "--- CUDA Version ---"
nvcc --version 2>/dev/null || cat /usr/local/cuda/version.txt 2>/dev/null || echo "(CUDA not found)"

echo ""
echo "========================================="
echo " Summary for PoSE-DB Protocol"
echo "========================================="

# Calculate wipeable memory
mem_total_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
mem_total_gb=$(echo "scale=1; $mem_total_kb / 1048576" | bc)
mem_reserve_gb=2
mem_wipeable_gb=$(echo "scale=1; $mem_total_gb - $mem_reserve_gb" | bc)
echo "Host DRAM total:    ${mem_total_gb} GB"
echo "Host DRAM reserve:  ${mem_reserve_gb} GB (OS + wipe process)"
echo "Host DRAM wipeable: ${mem_wipeable_gb} GB"

gpu_mem_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
if [ -n "$gpu_mem_mb" ]; then
    gpu_total_gb=$(echo "scale=1; $gpu_mem_mb / 1024" | bc)
    gpu_reserve_mb=256
    gpu_wipeable_gb=$(echo "scale=1; ($gpu_mem_mb - $gpu_reserve_mb) / 1024" | bc)
    echo "GPU HBM total:      ${gpu_total_gb} GB"
    echo "GPU HBM reserve:    ${gpu_reserve_mb} MB (drivers + runtime)"
    echo "GPU HBM wipeable:   ${gpu_wipeable_gb} GB"
fi

# NVMe: find the largest mounted partition that isn't the root device
nvme_total_gb="0"
nvme_mount=""
while read -r size mount; do
    # size is in 1K blocks
    gb=$(echo "scale=1; $size / 1048576" | bc 2>/dev/null || echo "0")
    if [ "$(echo "$gb > $nvme_total_gb" | bc 2>/dev/null)" = "1" ] && [ "$mount" != "/" ]; then
        nvme_total_gb="$gb"
        nvme_mount="$mount"
    fi
done < <(df --output=size,target --block-size=1K 2>/dev/null | tail -n+2)

if [ "$nvme_total_gb" = "0" ]; then
    echo "NVMe:               (no non-root NVMe mount found — check lsblk)"
else
    nvme_reserve_gb=1
    nvme_wipeable_gb=$(echo "scale=1; $nvme_total_gb - $nvme_reserve_gb" | bc)
    echo "NVMe total:         ${nvme_total_gb} GB (mounted at ${nvme_mount})"
    echo "NVMe reserve:       ${nvme_reserve_gb} GB (filesystem metadata)"
    echo "NVMe wipeable:      ${nvme_wipeable_gb} GB"
fi

echo ""
echo "========================================="
echo " Env Vars for System Test"
echo "========================================="
echo "# Copy-paste these to run the system test:"
echo "export POSE_DRAM_TOTAL_GB=${mem_total_gb}"
echo "export POSE_DRAM_GB=${mem_wipeable_gb}"
if [ -n "$gpu_mem_mb" ]; then
    echo "export POSE_HBM_TOTAL_GB=${gpu_total_gb}"
    echo "export POSE_HBM_GB=${gpu_wipeable_gb}"
fi
if [ "$nvme_total_gb" != "0" ]; then
    echo "export POSE_NVME_TOTAL_GB=${nvme_total_gb}"
    echo "export POSE_NVME_GB=${nvme_wipeable_gb}"
    echo "export POSE_NVME_PATH=${nvme_mount}"
fi
echo "========================================="
