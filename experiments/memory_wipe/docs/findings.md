# Phase 1 Findings — 2026-03-31 / 2026-04-01

Comprehensive record of all experiments, results, and design decisions from
the Phase 1 implementation session.

---

## 1. Hardware: Lambda GH200

Instance: `gpu_1x_gh200` in us-east-3, $1.99/hr.
IP: `192.222.57.125` (instance `1a11d84025d544e480dbc370098ce750`).

### Specs

| Component | Detail |
|-----------|--------|
| CPU | ARM Neoverse-V2, 64 cores, 1 thread/core |
| DRAM | 431.2 GiB LPDDR5X (NUMA node 0) |
| GPU | NVIDIA GH200 480GB, 94.5 GiB HBM3 (NUMA node 1) |
| Interconnect | NVLink-C2C, 900 GB/s bidirectional |
| Disk | 4 TB virtual disk (virtio `/dev/vda1`, ext4) |
| CUDA | 12.8, driver 570.148.08 |
| Kernel | 6.8.0-1013-nvidia-64k (aarch64) |

### Key Discovery: Unified Memory Architecture

Linux sees both LPDDR5X and HBM3 as system memory under separate NUMA nodes:

```
MemTotal (from /proc/meminfo): 525.7 GiB = 431.2 GiB (node 0) + 94.5 GiB (node 1)
```

This means mmap allocations can spill from LPDDR5X into HBM unless NUMA-pinned
with `mbind(MPOL_BIND)`. This caused OOM when we tried to allocate 450 GiB of
DRAM + 92 GiB of HBM simultaneously — the mmap ate into HBM pages, leaving
nothing for cudaMalloc.

**NUMA topology** (9 nodes, only 2 populated):
- Node 0: 431.2 GiB LPDDR5X, CPU cores 0-63
- Node 1: 94.5 GiB HBM3, no CPUs
- Nodes 2-8: empty (reserved for MIG)

---

## 2. Memory Inventory

### Probed Allocation Ceilings

Used `scripts/probe_limits.py` (binary search on actual allocation):

| Region | Method | Probed Max | Total | Utilization |
|--------|--------|-----------|-------|-------------|
| HBM | `cudaMalloc` (16 MiB steps) | 93.94 GiB | 94.50 GiB | 99.4% |
| DRAM | `mmap` + touch (no NUMA bind) | 430.5 GiB | 431.18 GiB | 99.8% |
| DRAM | `mmap` + `mbind` node 0 | ~418 GiB | 431.18 GiB | 96.9% |
| Disk | `posix_fallocate` | 3,943.8 GiB | 3,968.9 GiB | 99.4% |

### HBM Fine-Grained Probe

Used `scripts/probe_reliability.py` — fine-grained cudaMalloc search:

```
93.83 GiB  (reserve=685 MiB): OK
93.85 GiB  (reserve=669 MiB): OK
93.86 GiB  (reserve=653 MiB): OK
93.88 GiB  (reserve=637 MiB): OK
93.89 GiB  (reserve=621 MiB): OK
93.91 GiB  (reserve=605 MiB): OK
93.93 GiB  (reserve=589 MiB): OK
93.94 GiB  (reserve=573 MiB): OK    ← max allocatable
93.96 GiB  (reserve=557 MiB): FAIL  ← hard wall
```

**Minimum HBM reserve: 573 MiB.** Deterministic hard edge — 100% → 0% in one
16 MiB step. No stochastic zone.

### HBM Reliability Probe (10 reps per level)

```
-5% (88.31 GiB):  10/10  100%
-4% (89.24 GiB):  10/10  100%
-3% (90.17 GiB):  10/10  100%
-2% (91.10 GiB):  10/10  100%
-1% (92.03 GiB):  10/10  100%   ← safe ceiling at coarse granularity
+0% (92.96 GiB):   0/10    0%   ← appeared to be wall (was wrong — step too coarse)
```

The initial probe at 1% steps (0.93 GiB gaps) misidentified the wall at
92.96 GiB. Fine-grained probing revealed the actual ceiling is 93.94 GiB —
almost 1 GiB higher.

### DRAM: NUMA Binding vs Free Allocation

With `mbind(MPOL_BIND, node 0)`: max ~418 GiB. mbind checks available memory
on the target node and rejects if insufficient — even 414 GiB with 4 GiB
headroom failed.

Without NUMA binding: 430.5 GiB works (kernel allocates from both nodes).
This means the DRAM fill may partially land on HBM pages, but for a wipe
this is acceptable — we're wiping everything anyway.

### Disk: Lambda Bloat

Lambda preinstalls ~20 GiB of ML frameworks:

| Package | Size |
|---------|------|
| nvidia-cuda-dev | 3.2 GiB |
| python3-torch-cuda | 3.2 GiB |
| python3-tensorflow-cuda | 2.4 GiB |
| python3-jax-cuda12-plugin | 1.0 GiB |
| python3-flash-attn-cuda | 1.0 GiB |
| google-cloud-sdk (snap) | 2.8 GiB |
| azure-cli | 0.6 GiB |
| Various others | ~5 GiB |

After removal: 25 GiB → 7.7 GiB used. The remaining 7.7 GiB is kernel,
CUDA runtime libs, systemd, network stack, and our code.

### Unwipeable Memory Summary

| Region | Total | Wipeable | Reserved | Why |
|--------|-------|----------|----------|-----|
| LPDDR5X | 431.2 GiB | ~430 GiB | ~1 GiB | Kernel + page tables + process |
| HBM3 | 94.5 GiB | 93.9 GiB | 573 MiB | CUDA driver context (hard wall) |
| Disk | 3,968.9 GiB | ~3,961 GiB | ~8 GiB | Kernel + CUDA runtime + ext4 metadata |
| **Total** | **4,494.6 GiB** | **~4,485 GiB** | **~10 GiB** | **99.8% wipeable** |

### Unaccounted Specialized Memory

| Memory | Size | Status |
|--------|------|--------|
| CPU L1/L2 cache | ~72 MiB | Evicted by DRAM fill (transparent) |
| CPU SLC (shared L3) | ~64 MiB | Evicted by DRAM fill |
| GPU L2 cache | ~50 MiB | Flushed by cudaDeviceSynchronize |
| GPU shared memory | ~29 MiB | **Not wiped** — needs CUDA kernel |
| GPU register file | ~33 MiB | Cleared on context reset |
| HBM ECC parity bits | ~12 GiB | **Inaccessible** — hidden by memory controller |
| NIC memory (ConnectX mlx5) | ~32 MiB | Not accessible from host |
| Kernel slab caches | ~few MiB | In the DRAM reserve |

---

## 3. Throughput Benchmarks

### Raw Throughput (single core, 256 MiB chunks)

| Operation | Throughput |
|-----------|-----------|
| Zero-fill mmap (memset) | 10.1 GiB/s |
| memcpy to mmap | 10.1 GiB/s |
| AES-256-CTR raw (OpenSSL) | 4.3 GiB/s |
| AES + mmap write | 3.0 GiB/s |
| `/dev/urandom` (single core) | 0.6 GiB/s |
| `/dev/urandom` (16 cores parallel) | 9.5 GiB/s |

### Pre-generation Pipeline (verifier NVMe → prover memory)

| Phase | Throughput |
|-------|-----------|
| Parallel urandom (16 cores) → verifier file | 1.2 GiB/s (disk write bound) |
| Verifier file → DRAM (mmap) | 4.8 GiB/s |
| Verifier file → GPU HBM (cudaMemcpy) | 8.9 GiB/s |
| Verifier file → prover disk (same device cp) | 1.3 GiB/s (read-write contention) |

### System Test Results (best run: 93% DRAM, 96% HBM, 90% disk)

| Region | Wiped | Throughput | Time |
|--------|-------|-----------|------|
| DRAM | 401.0 GiB | 2.4 GiB/s | 169s |
| HBM | 90.2 GiB | 2.8 GiB/s | 32s |
| Disk | 3,550.2 GiB | 1.7 GiB/s | 2,046s |
| **Total** | **4,041.4 GiB** | | **2,247s (37.4 min)** |

| Metric | Value |
|--------|-------|
| Coverage | 89.9% |
| Challenge rounds | 1000/1000 |
| Verify time | 0.256s |
| Resume time | 2.2s |

### Bottleneck Analysis

The wipe is CPU-bound on noise generation, not I/O-bound. AES-256-CTR on a
single ARM core tops out at 4.3 GiB/s, while LPDDR5X can accept writes at
10.1 GiB/s and NVLink-C2C can transfer at 900 GB/s.

Pre-generating noise to a verifier NVMe drive would move AES off the critical
path. With two physical NVMe drives (Luke's design), the wipe becomes pure
I/O at 5-7 GiB/s per drive. Lambda only provides one virtual disk, so
disk-to-disk copy (1.3 GiB/s) is degraded by read-write contention on the
same device.

---

## 4. Architecture Decisions

### CuPy → Raw ctypes

Replaced CuPy (numpy + CUDA pool allocator) with direct ctypes bindings to
`libcudart.so`. This:
- Eliminated 140 MiB dependency
- Removed CuPy's pool allocator overhead
- Made GPU memory accounting exact (cudaMemGetInfo matches what we can allocate)
- Simplified the aarch64 build (no numpy wheel needed)

Key ctypes functions: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemGetInfo`,
`cudaSetDevice`, `cudaDeviceSynchronize`. All in `src/pose/detect.py` class
`CudaRuntime`.

### Auto-Detection via Hardware Queries

Replaced manual env-var configuration with `compute_ceilings()` in
`src/pose/detect.py`:
- DRAM: reads `/sys/devices/system/node/node0/meminfo` for NUMA-aware total
- HBM: calls `cudaMemGetInfo()` for exact free/total
- Disk: calls `os.statvfs()` for exact available
- Cgroup limits checked for container environments
- Target fractions applied to each (configurable, default 85%/90%/50%)

### NUMA Pinning

Added `mbind(MPOL_BIND, node 0)` to `DramRegion.__init__()` to prevent mmap
from allocating HBM pages. This is in `src/pose/memory/dram.py`. Falls back
silently if mbind is unavailable.

However, mbind is strict — it fails if the node doesn't have enough free
memory for the entire mapping. For maximum DRAM coverage, dropping NUMA binding
and letting the kernel place pages freely gives better results (430 GiB vs
418 GiB), at the cost of some pages landing on HBM.

### O_DIRECT Alignment Fix

NVMe reads with O_DIRECT require page-aligned buffers. Fixed by allocating a
reusable `mmap(-1, block_size)` buffer for read operations (kernel guarantees
mmap is page-aligned). Both reads and writes use this buffer when O_DIRECT is
active. Falls back to normal I/O on systems without O_DIRECT (macOS).

### Bulk Noise Generation

Added `generate_noise_bulk()` to `src/pose/noise.py` — generates 256 MiB
chunks per AES call instead of 4 KiB blocks. The prover's `fill_region_bulk()`
writes each chunk in a single `write_range()` call. This reduced Python loop
overhead but did not significantly improve throughput because the bottleneck
is AES itself (4.3 GiB/s), not Python iteration.

---

## 5. Protocol Verification

### Challenge-Response Correctness

Verified with 10,000 challenges across a 3-region layout (100 DRAM + 50 HBM +
200 disk blocks):

```
10000 challenges ALL PASSED
  dram :  2878 (28.8%)  expected ~28.6%
  hbm  :  1406 (14.1%)  expected ~14.3%
  nvme :  5716 (57.2%)  expected ~57.1%
```

Challenge distribution matches block count proportions (uniform random over
the global address space). The verifier regenerates each challenged block from
its secret seed and does byte-for-byte comparison. Tampered responses (zeros)
correctly fail verification.

### Prover Does Not Have the Seed

The `Prover` class has no `seed` attribute. It receives opaque block iterators
from the verifier's `noise_stream()`. The only way to respond to a challenge
is to read from stored memory. This matches the paper's security model
(Definition 2).

### Security Scope

Phase 1 is a functional demo, not a security proof:
- No distance-bounding (timing checks)
- Co-located verifier (same machine)
- PRF replaces true randomness (CPA-secure, not unconditional)

See `docs/plans/phase1.md` "Security Scope" section for full discussion.

---

## 6. Coverage Improvement Path

| Level | Technique | Coverage | Effort |
|-------|-----------|----------|--------|
| Current | Python + CUDA + full OS | 89.9% | Done |
| Remove bloat | Strip Lambda packages | ~98% | 1 hour (done) |
| Aggressive fractions | 98/99/95 | ~99% | Config change |
| Minimal host | C binary, no Python | ~99.99% | 2-3 days |
| kexec | Minimal wipe kernel | ~99.997% | 3-4 days |
| PXE netboot | Remote orchestrator | ~100% | 1-2 weeks |

---

## 7. Scripts Created

| Script | Purpose |
|--------|---------|
| `scripts/provision_gh200.sh` | Poll Lambda Cloud for GH200 capacity, auto-launch |
| `scripts/inventory.sh` | Run on GH200 to inventory DRAM/HBM/NVMe |
| `scripts/inventory_full.py` | Detailed inventory with NUMA + cudaMemGetInfo + statvfs |
| `scripts/probe_limits.py` | Binary search for max allocatable memory per region |
| `scripts/probe_reliability.py` | Test allocation reliability at -5% to +10% of ceiling, 10 reps |
| `scripts/compute_ceilings.py` | Compute wipeable ceilings from hardware queries |

---

## 8. Divergences from Paper

See `plan/modifications.md` for the full 9-point analysis. Key divergences
implemented in Phase 1:

1. **PRF replaces random noise** — AES-256-CTR, seed-based, CPA-secure
2. **Co-located verifier** — same machine, single process
3. **Multiple memory regions** — DRAM + HBM + disk vs paper's single memory
4. **No distance-bounding** — no timing checks on challenge-response
5. **Pre-generation design** — Luke's verifier-NVMe architecture not fully
   implemented due to single-disk Lambda constraint

---

## 9. Open Questions for Phase 2+

1. **Two NVMe drives**: Luke's design requires separate verifier/prover drives.
   Lambda provides one virtual disk. AWS p4d (8x 1TB NVMe) or GCP a2-ultragpu
   (2x 375 GiB NVMe) would work. ~$2-22/hr.

2. **On-GPU noise generation**: The `memory-sanitization` repo generates BLAKE3
   labels directly on GPU via a CUDA kernel, bypassing CPU→GPU transfer. Would
   dramatically improve HBM fill speed.

3. **Parallel AES**: 64 ARM cores available but we use 1 for AES-CTR. Multi-
   threaded generation would approach memory bandwidth (10 GiB/s).

4. **Container-based recovery**: Pull a pre-built docker image after wipe
   instead of reinstalling packages. Estimated ~5-20s recovery for a small
   model vs minutes for apt install.

5. **GPU shared memory wipe**: ~29 MiB of explicitly addressable GPU shared
   memory is not currently wiped. Needs a simple CUDA kernel.
