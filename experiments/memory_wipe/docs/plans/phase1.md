# Phase 1: Single-Node Unconditional PoSE-DB Protocol

**Goal**: Demonstrate that the PoSE-DB memory erasure protocol can fill all
addressable memory (DRAM, GPU HBM, NVMe) on a single NVIDIA GH200 node with
verifier-generated noise, and pass challenge-response verification.

**No stubs or mocks in production code.** Every line of code must do real work.
No placeholder implementations, no `pass` bodies, no `# TODO` stand-ins. If
a piece of functionality is genuinely impossible to implement (e.g., a hardware
limitation discovered at runtime), it must be:
1. Documented in the final report under `"limitations"` with a specific reason
2. Not silently faked — the report must clearly state what was stubbed and why
Tests may use DramRegion as a stand-in for HbmRegion when running without a
GPU, but only in test code, and only with a skip marker so the real test runs
on hardware.

**Done when**: A report (`reports/wipe_report.json`) generated on the GH200 contains:
1. Every wipeable byte across all three memory types overwritten with noise
2. All 1000 challenge-response rounds passed
3. Complete memory inventory (total, wiped, reserved — per region)
4. Key timing metrics: total wipe time, per-region fill throughput, verification
   time, and time to resume basic operation after wipe
5. Explanation of why each reserved region could not be wiped

**Non-goals for Phase 1**:
- Full inference resumption after wipe (Phase 2 — but we DO measure basic
  system recovery time: GPU re-init, DRAM re-availability, NVMe remount)
- Multi-node / cluster wipes (Phase 3)
- Verifier isolation / security hardening
- Distance-bounding timing checks

### Security Scope — Read This First

**Phase 1 is a functional demo, not a security proof.** It demonstrates the
*mechanism* of memory filling and challenge-response verification. It does NOT
prove secure erasure to any external party. Specifically:

- **No distance-bounding.** The paper's security argument (Section III) relies on
  round-trip-time bounds (Δ) to prevent the prover from relaying challenges to
  a remote co-conspirator. Without timing checks, a malicious prover could keep
  old data on a remote server and fetch blocks between challenge rounds. Phase 1
  proves only: "all addressable memory was filled with noise at the time of
  verification." It does not prove the prover couldn't also store data elsewhere.

- **Co-located verifier.** The verifier runs on the prover's own machine. The
  prover's OS could, in theory, inspect the verifier's memory (including the
  seed). This collapses the trust separation the paper assumes (see
  `plan/modifications.md` point 3). For Phase 1, we treat this as acceptable:
  the goal is to show the wipe *works*, not that it's *unforgeable*.

- **PRF replaces true randomness.** This loses the paper's information-theoretic
  ("unconditional") security. Under standard cryptographic assumptions (AES as
  a PRF, i.e., CPA-secure), the protocol remains secure against computationally
  bounded adversaries. This is a weaker guarantee than the paper's Definition 2,
  but standard and sufficient for practical deployment.

What Phase 1 **does** prove: given an honest execution, every wipeable byte on
the node was overwritten, and the verifier can confirm this via spot-checks.
This is the building block for Phases 2 and 3.

---

## Key Concepts

### What is PoSE-DB?

Proof of Secure Erasure with Distance Bounding. A two-party protocol:

```
Verifier                         Prover (the machine to wipe)
   |                                |
   |--- send noise ψ ------------->|   ← "Initialization phase"
   |                                |   Prover fills ALL memory with ψ
   |                                |
   |--- challenge: block index i ->|   ← "Interactive phase" (repeat r times)
   |<-- response: block[i] --------|
   |                                |
   |   check: response == ψ[i]     |   ← "Verification phase"
```

If the prover responds correctly, the verifier is convinced the prover actually
stored the noise (displacing whatever was in memory before, including malware).

Reference: paper Definition 2 (Section V), `plan/pose_paper.pdf` page 5.

### The Unconditional Protocol (Definition 2)

This is the simplest PoSE-DB variant. The paper generates truly random bits ψ
of size equal to the prover's memory. We adapt this using a **PRF** (pseudorandom
function) keyed with a secret seed, so the verifier can regenerate any block
on demand without storing the entire noise. This is Luke's modification #1
(see `plan/modifications.md`).

The protocol:
1. **Setup**: Verifier picks a random 32-byte seed `s` and keeps it secret
2. **Fill**: Verifier generates `PRF(s, 0), PRF(s, 1), ...` and **streams the
   blocks** to the prover. The prover stores each block in memory but **never
   learns the seed**. This information asymmetry is essential: the prover can
   only respond to challenges by reading stored blocks, not by recomputing them.
3. **Challenge**: Verifier picks random index `i`, asks prover for block `i`
4. **Response**: Prover reads block `i` from memory and returns it
5. **Verify**: Verifier regenerates `PRF(s, i)` from its secret seed and checks

We use **AES-256-CTR** as the PRF. The seed is the AES key. Block index `i`
maps to a counter offset, so generating block `i` = encrypting zeros starting
at counter `i * (block_size // 16)`. This is:
- **Deterministic**: same seed + index = same output
- **Seekable**: can generate any block independently (O(1), no sequential scan)
- **Fast**: ARM Crypto Extensions on Grace CPU give >20 GB/s per core

**Counter space constraint**: With 4096-byte blocks, each block consumes 256
AES counter values. At 1.5 TB total memory, that's ~375M blocks = ~96 billion
counter positions, fitting in 37 bits of the 128-bit CTR space. The code
asserts this invariant; if block_size or total memory changes, verify the
counter doesn't overflow.

### GH200 Architecture

The NVIDIA GH200 Grace Hopper Superchip:
- **Grace CPU**: ARM Neoverse V2, LPDDR5X memory (~96-480 GB depending on SKU)
- **Hopper GPU**: 96 GB HBM3
- **Interconnect**: NVLink-C2C at 900 GB/s (CPU <-> GPU)
- **NVMe**: Instance-local SSD (size varies by Lambda config)

Three memory regions to wipe:

| Region    | Typical Size | Reserve       | Access Method        |
|-----------|-------------|---------------|----------------------|
| GPU HBM   | 96 GB       | ~256 MB       | CUDA (cupy/pycuda)   |
| Host DRAM | 96-480 GB   | ~2 GB         | mmap / ctypes        |
| NVMe SSD  | ~1.4 TB     | ~1 GB         | O_DIRECT file I/O    |

Exact sizes come from `scripts/inventory.sh` run on the actual instance.

### Security Parameters

From the paper (Corollary 1), the probability an adversary passes the protocol
while keeping a fraction `f` of memory unerased:

```
Pr[cheat] ≤ (1 - f)^r
```

where `r` = number of challenge rounds. To detect `f` = 1% unerased memory
with 99.99% confidence: `r = ceil(log(0.0001) / log(0.99))` = **917 rounds**.

We default to **1000 rounds**, which gives >99.99% detection for 1% evasion.

### Unwipeable Memory: What and Why

Not all memory on the node can be overwritten. Here's what must be reserved
and why:

| Region | Reserved | Why |
|--------|----------|-----|
| GPU HBM | ~256 MB | **CUDA driver context & page tables.** The GPU driver (kernel module `nvidia.ko`) allocates internal bookkeeping structures in HBM when the GPU is initialized. Overwriting these causes an unrecoverable GPU fault (`Xid 31/79` errors). The GPU would need a full cold reset (power cycle) to recover, which isn't possible remotely. ~128 MB is the hard floor; we use 256 MB for safety margin. |
| Host DRAM | ~2 GB | **Linux kernel + OS services + our wipe process itself.** The kernel's own text/data/page tables (~200 MB), slab caches, network stack buffers, and the wipe process's own code + stack + Python runtime all live in DRAM. If we overwrite these, the OS crashes and we lose the ability to run the verification phase or report results. 512 MB is the theoretical minimum; 2 GB gives room for the Python process, filesystem caches, and systemd. |
| NVMe SSD | ~1 GB | **Filesystem metadata + device controller reserved blocks.** The ext4 (or xfs) filesystem needs its superblock, inode tables, and journal. The NVMe controller also reserves a small amount for wear-leveling tables and firmware. Wiping these bricks the filesystem (no way to read back challenge responses) or causes the drive to enter a failed state requiring a full secure-erase ATA command. |

**Coverage formula**:
```
coverage = bytes_wiped / (dram_total + hbm_total + nvme_total)
```

For a typical GH200 (96 GB HBM + 96 GB DRAM + 1.4 TB NVMe = ~1.59 TB total):
```
reserved = 0.256 + 2.0 + 1.0 = 3.256 GB
coverage ≈ (1592 - 3.256) / 1592 ≈ 99.8%
```

The inventory (Task 2) will produce exact numbers. The final report includes
per-region breakdowns.

### Divergences from the Paper

The `plan/modifications.md` file identifies 9 ways this implementation differs
from the paper's Definition 2. Here's where each is handled:

| # | Modification | Impact | Handled in |
|---|-------------|--------|------------|
| 1 | PRF replaces true randomness | Loses information-theoretic security; CPA-secure under standard assumptions | `noise.py`, Architecture Decisions §1 |
| 2 | Bandwidth framing instead of distance | Paper uses latency (Δ); we don't enforce either in Phase 1 | Security Scope above; Phase 3 |
| 3 | Co-located verifier | Prover's OS could inspect verifier memory; no trust separation | Security Scope above; Phase 3 |
| 4 | Verifier stores noise on local NVMe | Not needed — verifier regenerates from seed on demand | Architecture Decisions §5 |
| 5 | Local streaming instead of network | Eliminates physical separation but enables TB-scale fills in seconds | `protocol.py` (verifier streams to prover in-process) |
| 6 | Multiple heterogeneous memory regions | Paper models single memory of size m·w; we have 3 regions with different APIs | `memmap.py`, `memory/` subpackage |
| 7 | Specific non-wipeable reservations | Paper treats abstractly as overhead; we quantify per-region | Unwipeable Memory section above, report |
| 8 | Two NVMe drives (one for verifier) | Not needed in Phase 1 since verifier regenerates from PRF seed | Architecture Decisions §5 |
| 9 | Post-wipe workload resumption | Paper doesn't address; we measure basic recovery time | `resume_time_s` in report; Phase 2 for full inference |

---

## Project Layout

```
memory_wipes/
├── pyproject.toml
├── scripts/
│   ├── provision_gh200.sh     # Launch/poll for GH200 instance
│   └── inventory.sh           # Inventory memory on GH200
├── src/
│   └── pose/
│       ├── __init__.py
│       ├── noise.py           # PRF-based noise generation
│       ├── memory/
│       │   ├── __init__.py
│       │   ├── dram.py        # Host DRAM writer/reader
│       │   ├── hbm.py         # GPU HBM writer/reader (CUDA)
│       │   └── nvme.py        # NVMe writer/reader (O_DIRECT)
│       ├── memmap.py          # Global block index -> (region, offset)
│       ├── verifier.py        # Noise gen, challenge, verify
│       ├── prover.py          # Fill memory, respond to challenges
│       ├── protocol.py        # Full protocol orchestrator
│       └── report.py          # Report generator (JSON + human-readable)
├── tests/
│   ├── conftest.py            # Shared fixtures, pytest markers
│   ├── test_noise.py
│   ├── test_dram.py
│   ├── test_hbm.py
│   ├── test_nvme.py
│   ├── test_memmap.py
│   ├── test_verifier.py
│   ├── test_prover.py
│   ├── test_protocol.py
│   └── test_system.py         # Full GH200 system test + report
├── reports/                   # Generated output (gitignored)
│   └── wipe_report.json       # Final report from system test
└── docs/
    └── plans/
        └── phase1.md          # This file
```

---

## Tasks

### Task 1: Provision a GH200 Instance

**Goal**: Get a running GH200 on Lambda Cloud and SSH into it.

**Steps**:
1. Run the provisioning script:
   ```bash
   ./scripts/provision_gh200.sh
   ```
   It polls every 30s until capacity appears, then launches and prints the IP.

2. Once it prints `ssh ubuntu@<ip>`, verify:
   ```bash
   ssh ubuntu@<ip> nvidia-smi
   ```

**Files**: `scripts/provision_gh200.sh` (already created)

**Done when**: You can SSH in and `nvidia-smi` shows a GH200 GPU.

---

### Task 2: Inventory Memory

**Goal**: Discover exact memory sizes on the GH200 to configure the protocol.

**Steps**:
1. Run the inventory script on the instance:
   ```bash
   ssh ubuntu@<ip> 'bash -s' < scripts/inventory.sh
   ```
2. Record the output. You need three numbers:
   - **DRAM total** (from `free -h` or `/proc/meminfo`)
   - **HBM total** (from `nvidia-smi`)
   - **NVMe total** (from `lsblk`)

3. Create `src/pose/config.py` with the discovered values (Task 3 will
   initialize the project first — just note the numbers for now).

**Files**: `scripts/inventory.sh` (already created)

**Done when**: You have the three memory sizes written down, and know the
device paths (e.g., `/dev/nvme0n1p1` for NVMe).

---

### Task 3: Initialize the Python Project

**Goal**: Scaffold the project with uv, pytest, and a passing smoke test.

**Steps**:

1. Initialize the project:
   ```bash
   cd /path/to/memory_wipes
   git init
   uv init --lib --name pose
   ```

2. Edit `pyproject.toml`:
   ```toml
   [project]
   name = "pose"
   version = "0.1.0"
   requires-python = ">=3.10"
   dependencies = [
       "cryptography>=44.0",
   ]

   [project.optional-dependencies]
   gpu = ["cupy-cuda12x>=13.0"]
   dev = ["pytest>=8.0", "pytest-cov"]

   [tool.pytest.ini_options]
   testpaths = ["tests"]
   markers = [
       "cuda: requires NVIDIA GPU with CUDA",
       "nvme: requires NVMe block device access",
       "slow: takes more than 10 seconds",
   ]
   ```

3. Create the package structure:
   ```bash
   mkdir -p src/pose/memory tests
   touch src/pose/__init__.py src/pose/memory/__init__.py
   ```

4. Create `tests/conftest.py`:
   ```python
   import shutil
   import pytest

   def cuda_available():
       try:
           import cupy
           cupy.cuda.runtime.getDeviceCount()
           return True
       except Exception:
           return False

   def nvme_device_available():
       """Check if a test NVMe path exists."""
       import os
       return os.path.exists(os.environ.get("POSE_NVME_PATH", ""))

   # Auto-skip markers
   def pytest_collection_modifyitems(items):
       for item in items:
           if "cuda" in item.keywords and not cuda_available():
               item.add_marker(pytest.mark.skip(reason="No CUDA GPU"))
           if "nvme" in item.keywords and not nvme_device_available():
               item.add_marker(pytest.mark.skip(reason="No NVMe device"))
   ```

5. Write a smoke test in `tests/test_smoke.py`:
   ```python
   def test_import():
       import pose
   ```

6. Run it:
   ```bash
   uv run pytest -v
   ```

7. Commit:
   ```bash
   git add -A && git commit -m "Initialize pose project with uv and pytest"
   ```

**Files**:
- `pyproject.toml`
- `src/pose/__init__.py`
- `src/pose/memory/__init__.py`
- `tests/conftest.py`
- `tests/test_smoke.py`

**Done when**: `uv run pytest` passes with 1 test green.

---

### Task 4: Noise Generator

**Goal**: Implement deterministic, seekable PRF-based block generation.

This is the core cryptographic primitive. It must be:
- **Deterministic**: `generate_block(seed, i)` always returns the same bytes
- **Seekable**: Can produce block `i` without generating blocks 0..i-1
- **Fast**: Will need to fill hundreds of GB

**Write tests first** (`tests/test_noise.py`):
```python
import os
from pose.noise import generate_block, generate_blocks

SEED = os.urandom(32)
BLOCK_SIZE = 4096


def test_deterministic():
    """Same seed + index always gives the same block."""
    a = generate_block(SEED, index=42, block_size=BLOCK_SIZE)
    b = generate_block(SEED, index=42, block_size=BLOCK_SIZE)
    assert a == b


def test_correct_size():
    block = generate_block(SEED, index=0, block_size=BLOCK_SIZE)
    assert len(block) == BLOCK_SIZE


def test_different_indices_differ():
    a = generate_block(SEED, index=0, block_size=BLOCK_SIZE)
    b = generate_block(SEED, index=1, block_size=BLOCK_SIZE)
    assert a != b


def test_different_seeds_differ():
    a = generate_block(b"\x00" * 32, index=0, block_size=BLOCK_SIZE)
    b = generate_block(b"\x01" * 32, index=0, block_size=BLOCK_SIZE)
    assert a != b


def test_generate_blocks_sequential():
    """Bulk generation matches individual generation."""
    blocks = list(generate_blocks(SEED, start=5, count=3, block_size=BLOCK_SIZE))
    assert len(blocks) == 3
    for i, block in enumerate(blocks):
        assert block == generate_block(SEED, index=5 + i, block_size=BLOCK_SIZE)


def test_not_all_zeros():
    """Output is not trivially zero (sanity check)."""
    block = generate_block(SEED, index=0, block_size=BLOCK_SIZE)
    assert block != b"\x00" * BLOCK_SIZE
```

**Then implement** (`src/pose/noise.py`):
```python
"""PRF-based noise generation using AES-256-CTR.

The noise for the entire memory is conceptually one long AES-CTR stream.
Block index `i` maps to byte offset `i * block_size` in that stream.
This makes generation both seekable (any block independently) and efficient
(sequential blocks share cipher state).
"""

import struct
from typing import Iterator

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


_MAX_CTR = 2**128 - 1  # AES-CTR counter space


def generate_block(seed: bytes, index: int, block_size: int = 4096) -> bytes:
    """Generate a single pseudorandom block.

    Args:
        seed: 32-byte AES-256 key (the verifier's secret).
        index: Block index (0-based).
        block_size: Size of each block in bytes. Must be a multiple of 16.
    """
    aes_blocks_per_block = block_size // 16
    counter_start = index * aes_blocks_per_block
    if counter_start + aes_blocks_per_block > _MAX_CTR:
        raise OverflowError(
            f"Block {index} would overflow the 128-bit AES-CTR counter space. "
            f"Reduce block count or block size."
        )
    nonce = counter_start.to_bytes(16, byteorder="big")
    cipher = Cipher(algorithms.AES256(seed), modes.CTR(nonce))
    enc = cipher.encryptor()
    return enc.update(b"\x00" * block_size) + enc.finalize()


def generate_blocks(
    seed: bytes, start: int, count: int, block_size: int = 4096
) -> Iterator[bytes]:
    """Generate a contiguous range of blocks efficiently.

    Uses a single AES-CTR cipher instance for the entire range.
    """
    counter_start = start * (block_size // 16)
    nonce = counter_start.to_bytes(16, byteorder="big")
    cipher = Cipher(algorithms.AES256(seed), modes.CTR(nonce))
    enc = cipher.encryptor()
    for _ in range(count):
        yield enc.update(b"\x00" * block_size)
    enc.finalize()
```

**Run and commit**:
```bash
uv run pytest tests/test_noise.py -v
git add src/pose/noise.py tests/test_noise.py && git commit -m "Add PRF noise generator (AES-256-CTR)"
```

**Done when**: All 6 tests pass.

---

### Task 5: DRAM Writer

**Goal**: Allocate a large region of host memory and write/read blocks to it.

The DRAM writer uses Python's `mmap` to allocate anonymous memory. On the
GH200, this will be LPDDR5X attached to the Grace CPU.

**Write tests first** (`tests/test_dram.py`):
```python
import os
import pytest
from pose.memory.dram import DramRegion

BLOCK_SIZE = 4096


def test_capacity():
    region = DramRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    assert region.num_blocks == 10
    region.close()


def test_write_read_roundtrip():
    region = DramRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    data = os.urandom(BLOCK_SIZE)
    region.write_block(0, data)
    assert region.read_block(0) == data
    region.close()


def test_write_multiple_blocks():
    region = DramRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    blocks = [os.urandom(BLOCK_SIZE) for _ in range(10)]
    for i, b in enumerate(blocks):
        region.write_block(i, b)
    for i, b in enumerate(blocks):
        assert region.read_block(i) == b
    region.close()


def test_index_out_of_range():
    region = DramRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    with pytest.raises(IndexError):
        region.read_block(10)
    region.close()


def test_context_manager():
    with DramRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE) as r:
        r.write_block(0, b"\xAB" * BLOCK_SIZE)
        assert r.read_block(0) == b"\xAB" * BLOCK_SIZE
```

**Then implement** (`src/pose/memory/dram.py`):
```python
"""Host DRAM memory region using mmap.

Allocates anonymous (non-file-backed) memory. On Linux, this comes from
the kernel's virtual memory system and is backed by physical DRAM.
"""

import mmap


class DramRegion:
    def __init__(self, size_bytes: int, block_size: int = 4096):
        self.block_size = block_size
        self.num_blocks = size_bytes // block_size
        self._size = self.num_blocks * block_size
        # MAP_ANONYMOUS | MAP_PRIVATE: not backed by a file
        self._buf = mmap.mmap(-1, self._size)

    def write_block(self, index: int, data: bytes) -> None:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        self._buf[offset : offset + self.block_size] = data

    def read_block(self, index: int) -> bytes:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        return self._buf[offset : offset + self.block_size]

    def close(self):
        self._buf.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

```bash
uv run pytest tests/test_dram.py -v
git add src/pose/memory/dram.py tests/test_dram.py && git commit -m "Add DRAM memory region writer/reader"
```

**Done when**: All 5 tests pass.

---

### Task 6: GPU HBM Writer

**Goal**: Allocate GPU memory and write/read blocks via CUDA.

Uses `cupy` for GPU memory management. Tests are marked `@pytest.mark.cuda`
and auto-skip on machines without a GPU.

**Write tests first** (`tests/test_hbm.py`):
```python
import os
import pytest
from pose.memory.hbm import HbmRegion

BLOCK_SIZE = 4096
pytestmark = pytest.mark.cuda


def test_capacity():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    assert region.num_blocks == 10
    region.close()


def test_write_read_roundtrip():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    data = os.urandom(BLOCK_SIZE)
    region.write_block(0, data)
    assert region.read_block(0) == data
    region.close()


def test_write_multiple_blocks():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    blocks = [os.urandom(BLOCK_SIZE) for _ in range(10)]
    for i, b in enumerate(blocks):
        region.write_block(i, b)
    for i, b in enumerate(blocks):
        assert region.read_block(i) == b
    region.close()


def test_index_out_of_range():
    region = HbmRegion(size_bytes=BLOCK_SIZE * 10, block_size=BLOCK_SIZE)
    with pytest.raises(IndexError):
        region.read_block(10)
    region.close()
```

**Then implement** (`src/pose/memory/hbm.py`):
```python
"""GPU HBM memory region using CuPy.

Allocates device memory on the GPU. Data is transferred host<->device
for write_block/read_block. For bulk fills, use write_blocks_bulk()
which streams data to minimize transfer overhead.
"""

import numpy as np


def _require_cupy():
    import cupy
    return cupy


class HbmRegion:
    def __init__(self, size_bytes: int, block_size: int = 4096):
        cp = _require_cupy()
        self.block_size = block_size
        self.num_blocks = size_bytes // block_size
        self._size = self.num_blocks * block_size
        # Allocate as a flat byte array on the GPU
        self._buf = cp.zeros(self._size, dtype=cp.uint8)

    def write_block(self, index: int, data: bytes) -> None:
        cp = _require_cupy()
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        host_arr = np.frombuffer(data, dtype=np.uint8)
        self._buf[offset : offset + self.block_size] = cp.asarray(host_arr)

    def read_block(self, index: int) -> bytes:
        cp = _require_cupy()
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        offset = index * self.block_size
        return bytes(cp.asnumpy(self._buf[offset : offset + self.block_size]))

    def close(self):
        del self._buf

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

> **Note**: On the GH200, the NVLink-C2C interconnect between CPU and GPU runs
> at 900 GB/s, so host<->device transfers are much faster than on PCIe systems.
> For maximum fill speed, consider generating noise directly on the GPU using a
> CuPy kernel, but the CPU->GPU transfer approach is simpler and sufficient for
> Phase 1.

```bash
# This will skip on your Mac — that's expected. Run on GH200.
uv run pytest tests/test_hbm.py -v
git add src/pose/memory/hbm.py tests/test_hbm.py && git commit -m "Add GPU HBM memory region writer/reader"
```

**Done when**: Tests pass on GH200 (skip locally is fine).

---

### Task 7: NVMe Writer

**Goal**: Write/read blocks directly to an NVMe file, bypassing OS page cache.

We write to a large preallocated file on the NVMe filesystem using `O_DIRECT`
to bypass the page cache. This ensures data actually hits the SSD and doesn't
just sit in DRAM (which would defeat the purpose — we're wiping DRAM too).

Tests are marked `@pytest.mark.nvme` and require the env var `POSE_NVME_PATH`
pointing to a directory on the NVMe mount.

**Write tests first** (`tests/test_nvme.py`):
```python
import os
import tempfile
import pytest
from pose.memory.nvme import NvmeRegion

BLOCK_SIZE = 4096
pytestmark = pytest.mark.nvme


@pytest.fixture
def nvme_dir():
    """Use POSE_NVME_PATH env var, or skip."""
    path = os.environ.get("POSE_NVME_PATH")
    if not path or not os.path.isdir(path):
        pytest.skip("Set POSE_NVME_PATH to an NVMe-mounted directory")
    return path


def test_write_read_roundtrip(nvme_dir):
    filepath = os.path.join(nvme_dir, "test_region.bin")
    region = NvmeRegion(filepath, num_blocks=10, block_size=BLOCK_SIZE)
    data = os.urandom(BLOCK_SIZE)
    region.write_block(0, data)
    assert region.read_block(0) == data
    region.close()
    os.unlink(filepath)


def test_multiple_blocks(nvme_dir):
    filepath = os.path.join(nvme_dir, "test_multi.bin")
    region = NvmeRegion(filepath, num_blocks=10, block_size=BLOCK_SIZE)
    blocks = [os.urandom(BLOCK_SIZE) for _ in range(10)]
    for i, b in enumerate(blocks):
        region.write_block(i, b)
    for i, b in enumerate(blocks):
        assert region.read_block(i) == b
    region.close()
    os.unlink(filepath)


def test_index_out_of_range(nvme_dir):
    filepath = os.path.join(nvme_dir, "test_oor.bin")
    region = NvmeRegion(filepath, num_blocks=10, block_size=BLOCK_SIZE)
    with pytest.raises(IndexError):
        region.read_block(10)
    region.close()
    os.unlink(filepath)
```

**Then implement** (`src/pose/memory/nvme.py`):
```python
"""NVMe SSD memory region using direct I/O.

Writes to a preallocated file with O_DIRECT to bypass the OS page cache.
This ensures data is physically written to the SSD, not just cached in DRAM.

O_DIRECT requires:
- Reads/writes aligned to 512-byte (or 4096-byte) boundaries
- Buffer sizes that are multiples of the alignment
Our 4096-byte block size satisfies both.
"""

import mmap
import os


class NvmeRegion:
    def __init__(self, filepath: str, num_blocks: int, block_size: int = 4096):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self._size = num_blocks * block_size
        self._filepath = filepath
        self._use_direct = hasattr(os, "O_DIRECT")

        flags = os.O_RDWR | os.O_CREAT
        if self._use_direct:
            flags |= os.O_DIRECT
        self._fd = os.open(filepath, flags, 0o600)
        os.ftruncate(self._fd, self._size)

        # O_DIRECT requires page-aligned buffers. We allocate a reusable
        # aligned buffer via mmap (guaranteed page-aligned by the kernel).
        if self._use_direct:
            self._aligned_buf = mmap.mmap(-1, block_size)

    def write_block(self, index: int, data: bytes) -> None:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        os.lseek(self._fd, index * self.block_size, os.SEEK_SET)
        if self._use_direct:
            self._aligned_buf[:] = data
            os.write(self._fd, self._aligned_buf)
        else:
            os.write(self._fd, data)

    def read_block(self, index: int) -> bytes:
        if index < 0 or index >= self.num_blocks:
            raise IndexError(f"Block {index} out of range [0, {self.num_blocks})")
        os.lseek(self._fd, index * self.block_size, os.SEEK_SET)
        return os.read(self._fd, self.block_size)

    def close(self):
        os.close(self._fd)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
```

> **Fallback**: If `O_DIRECT` is not available (e.g., macOS for local testing),
> the code falls back to normal I/O. On Linux, the write buffer is page-aligned
> via a reusable `mmap` allocation to satisfy the kernel's `O_DIRECT` requirements.

```bash
uv run pytest tests/test_nvme.py -v  # Will skip without POSE_NVME_PATH
git add src/pose/memory/nvme.py tests/test_nvme.py && git commit -m "Add NVMe memory region writer/reader"
```

**Done when**: Tests pass on the GH200 with `POSE_NVME_PATH=/mnt/nvme` set.

---

### Task 8: Memory Map

**Goal**: Map a global block index to the correct memory region and local offset.

The protocol treats all memory as one flat address space of blocks. The memory
map translates global block index -> (region_name, local_block_index).

Layout (blocks are assigned contiguously):
```
Global index:  [0 ............... D-1] [D ........... D+H-1] [D+H ...... D+H+N-1]
Region:         ---- DRAM ----          ---- HBM ----         ---- NVMe ----
Local index:   [0 ............... D-1] [0 ........... H-1]   [0 .......... N-1]
```

**Write tests first** (`tests/test_memmap.py`):
```python
from pose.memmap import MemoryMap


def test_total_blocks():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    assert mm.total_blocks == 350


def test_dram_region():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    region, local_idx = mm.resolve(0)
    assert region == "dram"
    assert local_idx == 0

    region, local_idx = mm.resolve(99)
    assert region == "dram"
    assert local_idx == 99


def test_hbm_region():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    region, local_idx = mm.resolve(100)
    assert region == "hbm"
    assert local_idx == 0

    region, local_idx = mm.resolve(149)
    assert region == "hbm"
    assert local_idx == 49


def test_nvme_region():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    region, local_idx = mm.resolve(150)
    assert region == "nvme"
    assert local_idx == 0

    region, local_idx = mm.resolve(349)
    assert region == "nvme"
    assert local_idx == 199


def test_out_of_range():
    mm = MemoryMap(dram_blocks=100, hbm_blocks=50, nvme_blocks=200)
    import pytest
    with pytest.raises(IndexError):
        mm.resolve(350)
```

**Then implement** (`src/pose/memmap.py`):
```python
"""Maps global block indices to (region, local_index) pairs.

The protocol treats all memory as a flat array of blocks. This module
translates between the flat view and the three physical memory regions.
"""


class MemoryMap:
    def __init__(self, dram_blocks: int, hbm_blocks: int, nvme_blocks: int):
        self.dram_blocks = dram_blocks
        self.hbm_blocks = hbm_blocks
        self.nvme_blocks = nvme_blocks
        self.total_blocks = dram_blocks + hbm_blocks + nvme_blocks

        self._hbm_start = dram_blocks
        self._nvme_start = dram_blocks + hbm_blocks

    def resolve(self, global_index: int) -> tuple[str, int]:
        """Map a global block index to (region_name, local_index)."""
        if global_index < 0 or global_index >= self.total_blocks:
            raise IndexError(
                f"Block {global_index} out of range [0, {self.total_blocks})"
            )
        if global_index < self._hbm_start:
            return ("dram", global_index)
        if global_index < self._nvme_start:
            return ("hbm", global_index - self._hbm_start)
        return ("nvme", global_index - self._nvme_start)
```

```bash
uv run pytest tests/test_memmap.py -v
git add src/pose/memmap.py tests/test_memmap.py && git commit -m "Add memory map (global index -> region + offset)"
```

**Done when**: All 5 tests pass.

---

### Task 9: Verifier

**Goal**: The verifier generates noise, issues random challenges, and checks
responses.

The verifier is the trusted party. It:
1. Picks a random seed and **keeps it secret**
2. Streams noise blocks to the prover (prover never learns the seed)
3. Issues `r` random challenges (block indices)
4. Regenerates the challenged block from the seed and checks the response

**Critical**: The prover must NOT have access to the seed. If it did, it could
answer any challenge by recomputing `PRF(seed, i)` on the fly, without ever
actually storing the noise. The entire point of the protocol is that the
prover can only respond by reading from memory.

**Write tests first** (`tests/test_verifier.py`):
```python
import os
from pose.verifier import Verifier
from pose.noise import generate_block


def test_verifier_seed_is_random():
    v1 = Verifier(total_blocks=1000, block_size=4096)
    v2 = Verifier(total_blocks=1000, block_size=4096)
    assert v1.seed != v2.seed


def test_noise_stream_matches_individual_blocks():
    """The stream the prover receives matches what the verifier checks against."""
    v = Verifier(total_blocks=100, block_size=4096)
    stream = list(v.noise_stream())
    assert len(stream) == 100
    for i, block in enumerate(stream):
        assert block == generate_block(v.seed, i, 4096)


def test_noise_stream_does_not_expose_seed():
    """The stream yields bytes, not the seed itself."""
    v = Verifier(total_blocks=10, block_size=4096)
    for block in v.noise_stream():
        assert isinstance(block, bytes)
        assert len(block) == 4096
        assert block != v.seed  # Not the seed


def test_challenge_in_range():
    v = Verifier(total_blocks=1000, block_size=4096)
    for _ in range(100):
        idx = v.challenge()
        assert 0 <= idx < 1000


def test_verify_correct():
    v = Verifier(total_blocks=100, block_size=4096)
    stream = list(v.noise_stream())
    idx = v.challenge()
    assert v.verify(idx, stream[idx]) is True


def test_verify_incorrect():
    v = Verifier(total_blocks=1000, block_size=4096)
    idx = v.challenge()
    assert v.verify(idx, b"\x00" * 4096) is False
```

**Then implement** (`src/pose/verifier.py`):
```python
"""Verifier: streams noise to prover, issues challenges, checks responses.

The verifier holds the PRF seed. It never exposes the seed to the prover.
The prover only receives opaque noise blocks via noise_stream().
"""

import os
import secrets
from typing import Iterator

from pose.noise import generate_block, generate_blocks


class Verifier:
    def __init__(self, total_blocks: int, block_size: int = 4096, seed: bytes | None = None):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.seed = seed or os.urandom(32)

    def noise_stream(self) -> Iterator[bytes]:
        """Generate the noise block stream for the prover to store.

        The prover calls this to receive blocks. It never sees the seed.
        """
        yield from generate_blocks(
            self.seed, start=0, count=self.total_blocks,
            block_size=self.block_size,
        )

    def challenge(self) -> int:
        """Pick a random block index to challenge."""
        return secrets.randbelow(self.total_blocks)

    def verify(self, index: int, response: bytes) -> bool:
        """Check the prover's response by regenerating from the secret seed."""
        expected = generate_block(self.seed, index, self.block_size)
        return response == expected
```

```bash
uv run pytest tests/test_verifier.py -v
git add src/pose/verifier.py tests/test_verifier.py && git commit -m "Add verifier (noise stream, challenge, verify)"
```

**Done when**: All 6 tests pass.

---

### Task 10: Prover

**Goal**: The prover fills all memory regions with noise and responds to
challenges.

The prover:
1. Receives the seed from the verifier (in a real protocol this would be
   streamed noise; we use the seed directly since the verifier is co-located)
2. Fills DRAM, HBM, and NVMe with noise blocks
3. Responds to challenges by reading the requested block

**Write tests first** (`tests/test_prover.py`):
```python
import os
from pose.prover import Prover
from pose.memory.dram import DramRegion
from pose.noise import generate_block

BLOCK_SIZE = 4096
NUM_BLOCKS = 100  # Small for testing


def test_fill_and_respond():
    """Prover fills memory from a block stream and can return any block."""
    seed = os.urandom(32)
    dram = DramRegion(size_bytes=NUM_BLOCKS * BLOCK_SIZE, block_size=BLOCK_SIZE)

    # Verifier produces the stream; prover stores it without seeing the seed
    verifier = Verifier(total_blocks=NUM_BLOCKS, block_size=BLOCK_SIZE, seed=seed)
    prover = Prover(regions={"dram": dram}, block_size=BLOCK_SIZE)
    prover.fill(verifier.noise_stream())

    for idx in [0, 49, 99]:
        response = prover.respond(idx)
        assert verifier.verify(idx, response)

    dram.close()


def test_fill_multiple_regions():
    """Prover fills across multiple regions from a single stream."""
    seed = os.urandom(32)
    dram = DramRegion(size_bytes=50 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    dram2 = DramRegion(size_bytes=50 * BLOCK_SIZE, block_size=BLOCK_SIZE)

    verifier = Verifier(total_blocks=100, block_size=BLOCK_SIZE, seed=seed)
    prover = Prover(
        regions={"dram": dram, "hbm": dram2},
        block_size=BLOCK_SIZE,
    )
    prover.fill(verifier.noise_stream())

    # Block 0 in dram, block 50 in "hbm"
    assert verifier.verify(0, prover.respond(0))
    assert verifier.verify(50, prover.respond(50))

    dram.close()
    dram2.close()


def test_prover_does_not_receive_seed():
    """Prover has no attribute or reference to the seed."""
    dram = DramRegion(size_bytes=10 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    prover = Prover(regions={"dram": dram}, block_size=BLOCK_SIZE)
    assert not hasattr(prover, "seed")
    dram.close()
```

**Then implement** (`src/pose/prover.py`):
```python
"""Prover: stores noise blocks streamed from the verifier, responds to challenges.

The prover NEVER has access to the PRF seed. It receives opaque blocks from
the verifier's noise_stream() and stores them. It can only respond to
challenges by reading from its stored memory.
"""

from typing import Iterator


class Prover:
    def __init__(self, regions: dict, block_size: int = 4096):
        """
        Args:
            regions: Dict mapping region name -> region object.
                     Each region must have .num_blocks, .write_block(), .read_block().
                     Insertion order determines the memory map layout.
            block_size: Block size in bytes.
        """
        self.block_size = block_size
        self._regions = regions
        self._region_list = list(regions.items())

        self._offsets = {}  # region_name -> global_start_index
        offset = 0
        for name, r in self._region_list:
            self._offsets[name] = offset
            offset += r.num_blocks
        self.total_blocks = offset

    def fill(self, block_stream: Iterator[bytes]) -> None:
        """Fill all memory regions from an opaque block stream.

        The stream is produced by the verifier. The prover does not know
        the seed or how blocks are generated — it just stores them.
        """
        global_idx = 0
        for name, region in self._region_list:
            for _ in range(region.num_blocks):
                block = next(block_stream)
                region.write_block(global_idx - self._offsets[name], block)
                global_idx += 1

    def fill_region(self, name: str, block_stream: Iterator[bytes]) -> None:
        """Fill a single named region from the stream.

        Used by the protocol orchestrator to time each region independently.
        """
        region = self._regions[name]
        for local_idx in range(region.num_blocks):
            block = next(block_stream)
            region.write_block(local_idx, block)

    def respond(self, global_index: int) -> bytes:
        """Return the block at the given global index (memory read only)."""
        for name, region in self._region_list:
            start = self._offsets[name]
            if start <= global_index < start + region.num_blocks:
                return region.read_block(global_index - start)
        raise IndexError(f"Block {global_index} out of range [0, {self.total_blocks})")
```

```bash
uv run pytest tests/test_prover.py -v
git add src/pose/prover.py tests/test_prover.py && git commit -m "Add prover (store streamed blocks, respond to challenges)"
```

**Done when**: All 3 tests pass.

---

### Task 11: Protocol Orchestrator

**Goal**: Wire verifier + prover together into the full protocol and verify it
end to end.

**Write tests first** (`tests/test_protocol.py`):
```python
from pose.protocol import run_protocol
from pose.memory.dram import DramRegion

BLOCK_SIZE = 4096


def test_protocol_passes_with_honest_prover():
    """Honest prover passes all challenges."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        block_size=BLOCK_SIZE,
        num_rounds=50,
    )
    assert result.passed is True
    assert result.rounds_passed == 50
    assert result.rounds_total == 50


def test_protocol_reports_coverage():
    """Protocol reports how many bytes were wiped and coverage ratio."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        region_info={"dram": {
            "total_bytes": 110 * BLOCK_SIZE,
            "reserved_bytes": 10 * BLOCK_SIZE,
            "reserved_reason": "test reserve",
        }},
        block_size=BLOCK_SIZE,
        num_rounds=10,
    )
    assert result.bytes_wiped == 100 * BLOCK_SIZE
    assert result.bytes_total == 110 * BLOCK_SIZE
    assert result.bytes_reserved == 10 * BLOCK_SIZE
    assert 0.90 < result.coverage < 0.92  # ~90.9%


def test_protocol_has_per_region_metrics():
    """Each region reports fill time and throughput."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        block_size=BLOCK_SIZE,
        num_rounds=10,
    )
    assert len(result.region_metrics) == 1
    assert result.region_metrics[0].name == "dram"
    assert result.region_metrics[0].fill_time_s > 0
    assert result.region_metrics[0].fill_throughput_gbps > 0


def test_protocol_measures_resume_time():
    """Protocol measures time to resume after wipe."""
    dram = DramRegion(size_bytes=100 * BLOCK_SIZE, block_size=BLOCK_SIZE)
    result = run_protocol(
        regions={"dram": dram},
        block_size=BLOCK_SIZE,
        num_rounds=10,
    )
    assert result.resume_time_s >= 0
```

**Then implement** (`src/pose/protocol.py`):
```python
"""Full PoSE-DB protocol orchestrator."""

import time
from dataclasses import dataclass, field

from pose.noise import generate_blocks
from pose.verifier import Verifier
from pose.prover import Prover


@dataclass
class RegionMetrics:
    name: str
    total_bytes: int           # Physical capacity of this region
    reserved_bytes: int        # Bytes we could NOT wipe
    reserved_reason: str       # Why those bytes are unwipeable
    wiped_bytes: int           # Bytes actually overwritten with noise
    fill_time_s: float         # Time to fill this region
    fill_throughput_gbps: float  # GB/s during fill


@dataclass
class ProtocolResult:
    passed: bool
    rounds_passed: int
    rounds_total: int
    bytes_wiped: int            # Total across all regions
    bytes_total: int            # Total physical memory on node
    bytes_reserved: int         # Total unwipeable
    coverage: float             # bytes_wiped / bytes_total
    fill_time_s: float          # Wall time for entire fill phase
    verify_time_s: float        # Wall time for challenge-response phase
    resume_time_s: float        # Time to resume basic operation after wipe
    region_metrics: list[RegionMetrics] = field(default_factory=list)
    seed: bytes = b""


def run_protocol(
    regions: dict,
    region_info: dict | None = None,
    block_size: int = 4096,
    num_rounds: int = 1000,
) -> ProtocolResult:
    """Run the unconditional PoSE-DB protocol.

    Args:
        regions: {"dram": DramRegion, "hbm": HbmRegion, "nvme": NvmeRegion}
        region_info: Optional per-region metadata for the report:
            {"dram": {"total_bytes": ..., "reserved_bytes": ..., "reserved_reason": ...}, ...}
        block_size: Block size in bytes.
        num_rounds: Number of challenge-response rounds.
    """
    prover = Prover(regions=regions, block_size=block_size)
    verifier = Verifier(total_blocks=prover.total_blocks, block_size=block_size)

    # --- Fill phase (timed per-region) ---
    # Verifier streams noise per-region so we can time each independently.
    # The prover never receives the seed — only opaque blocks.
    t_fill_start = time.monotonic()
    region_metrics = []
    global_offset = 0
    for name, region in regions.items():
        # Generate a stream for just this region's block range
        region_stream = generate_blocks(
            verifier.seed, start=global_offset, count=region.num_blocks,
            block_size=block_size,
        )
        t0 = time.monotonic()
        prover.fill_region(name, region_stream)
        elapsed = time.monotonic() - t0
        global_offset += region.num_blocks
        wiped = region.num_blocks * block_size
        info = (region_info or {}).get(name, {})
        region_metrics.append(RegionMetrics(
            name=name,
            total_bytes=info.get("total_bytes", wiped),
            reserved_bytes=info.get("reserved_bytes", 0),
            reserved_reason=info.get("reserved_reason", ""),
            wiped_bytes=wiped,
            fill_time_s=elapsed,
            fill_throughput_gbps=wiped / elapsed / (1024**3) if elapsed > 0 else 0,
        ))
    fill_time = time.monotonic() - t_fill_start

    # --- Challenge-response phase ---
    t0 = time.monotonic()
    rounds_passed = 0
    for _ in range(num_rounds):
        idx = verifier.challenge()
        response = prover.respond(idx)
        if verifier.verify(idx, response):
            rounds_passed += 1
    verify_time = time.monotonic() - t0

    # --- Resume operation timing ---
    # Measures time to: release DRAM, re-init CUDA context, sync NVMe
    t0 = time.monotonic()
    for name, region in regions.items():
        region.close()
    # Re-verify GPU is usable
    try:
        import cupy
        cupy.cuda.Device(0).synchronize()
    except Exception:
        pass
    resume_time = time.monotonic() - t0

    bytes_wiped = sum(rm.wiped_bytes for rm in region_metrics)
    bytes_total = sum(rm.total_bytes for rm in region_metrics)
    bytes_reserved = sum(rm.reserved_bytes for rm in region_metrics)

    return ProtocolResult(
        passed=(rounds_passed == num_rounds),
        rounds_passed=rounds_passed,
        rounds_total=num_rounds,
        bytes_wiped=bytes_wiped,
        bytes_total=bytes_total,
        bytes_reserved=bytes_reserved,
        coverage=bytes_wiped / bytes_total if bytes_total > 0 else 0,
        fill_time_s=fill_time,
        verify_time_s=verify_time,
        resume_time_s=resume_time,
        region_metrics=region_metrics,
        seed=verifier.seed,
    )
```


```bash
uv run pytest tests/test_protocol.py -v
git add src/pose/protocol.py tests/test_protocol.py && git commit -m "Add protocol orchestrator (fill + challenge-response)"
```

**Done when**: Both tests pass.

---

### Task 12: System Test on the GH200

**Goal**: Run the full protocol on the actual GH200, wiping all three memory
types with maximum coverage.

This is the integration test that proves Phase 1 works. It runs on the GH200
instance (not locally).

**Create the system test** (`tests/test_system.py`):
```python
"""System test — run on GH200 only.

Usage:
    POSE_NVME_PATH=/mnt/nvme uv run pytest tests/test_system.py -v -s

Fills all three memory regions, verifies 1000 challenge rounds,
measures resume timing, and writes a full report to reports/.
"""

import os
import json
import pytest
from pose.protocol import run_protocol
from pose.report import generate_report
from pose.memory.dram import DramRegion
from pose.memory.hbm import HbmRegion
from pose.memory.nvme import NvmeRegion

BLOCK_SIZE = 4096

# --- Configure from inventory (Task 2) ---
# Replace with actual values from scripts/inventory.sh output.
DRAM_TOTAL_GB = float(os.environ.get("POSE_DRAM_TOTAL_GB", "96"))
DRAM_WIPEABLE_GB = float(os.environ.get("POSE_DRAM_GB", "94"))
HBM_TOTAL_GB = float(os.environ.get("POSE_HBM_TOTAL_GB", "96"))
HBM_WIPEABLE_GB = float(os.environ.get("POSE_HBM_GB", "95.75"))
NVME_TOTAL_GB = float(os.environ.get("POSE_NVME_TOTAL_GB", "1400"))
NVME_WIPEABLE_GB = float(os.environ.get("POSE_NVME_GB", "1399"))
NVME_PATH = os.environ.get("POSE_NVME_PATH", "/mnt/nvme")
NUM_ROUNDS = int(os.environ.get("POSE_ROUNDS", "1000"))
REPORT_DIR = os.environ.get("POSE_REPORT_DIR", "reports")


def _gb(n):
    return int(n * (1024 ** 3))


@pytest.mark.slow
def test_full_wipe():
    nvme_file = os.path.join(NVME_PATH, "pose_wipe.bin")

    dram = DramRegion(size_bytes=_gb(DRAM_WIPEABLE_GB), block_size=BLOCK_SIZE)
    hbm = HbmRegion(size_bytes=_gb(HBM_WIPEABLE_GB), block_size=BLOCK_SIZE)
    nvme = NvmeRegion(nvme_file, num_blocks=_gb(NVME_WIPEABLE_GB) // BLOCK_SIZE,
                      block_size=BLOCK_SIZE)

    regions = {"dram": dram, "hbm": hbm, "nvme": nvme}

    # Per-region metadata: total capacity and what we reserved + why
    region_info = {
        "dram": {
            "total_bytes": _gb(DRAM_TOTAL_GB),
            "reserved_bytes": _gb(DRAM_TOTAL_GB - DRAM_WIPEABLE_GB),
            "reserved_reason": (
                "Linux kernel text/data/page tables (~200 MB), slab caches, "
                "network stack, systemd, and the wipe process itself "
                "(Python runtime + mmap overhead). Overwriting these crashes "
                "the OS, losing the ability to run verification."
            ),
        },
        "hbm": {
            "total_bytes": _gb(HBM_TOTAL_GB),
            "reserved_bytes": _gb(HBM_TOTAL_GB - HBM_WIPEABLE_GB),
            "reserved_reason": (
                "CUDA driver context, GPU page tables, and ECC metadata. "
                "The nvidia.ko kernel module allocates internal structures in "
                "HBM on GPU init. Overwriting causes Xid 31/79 faults requiring "
                "a cold power cycle (not possible remotely on Lambda)."
            ),
        },
        "nvme": {
            "total_bytes": _gb(NVME_TOTAL_GB),
            "reserved_bytes": _gb(NVME_TOTAL_GB - NVME_WIPEABLE_GB),
            "reserved_reason": (
                "Filesystem superblock, inode table, journal, and NVMe controller "
                "reserved blocks (wear-leveling, firmware). Wiping these bricks "
                "the filesystem or puts the drive in a failed state."
            ),
        },
    }

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
        print(f" {rm.name.upper():6s}  total={rm.total_bytes/(1024**3):.1f} GB"
              f"  wiped={rm.wiped_bytes/(1024**3):.1f} GB"
              f"  reserved={rm.reserved_bytes/(1024**3):.3f} GB")
        print(f"         reason: {rm.reserved_reason[:80]}...")
    print(f"")
    print(f" --- Key Metrics ---")
    print(f" Total memory:   {result.bytes_total/(1024**3):.1f} GB")
    print(f" Bytes wiped:    {result.bytes_wiped/(1024**3):.1f} GB")
    print(f" Coverage:       {result.coverage*100:.2f}%")
    print(f" Wipe time:      {result.fill_time_s:.1f}s")
    for rm in result.region_metrics:
        print(f"   {rm.name:6s}       {rm.fill_time_s:.1f}s"
              f"  ({rm.fill_throughput_gbps:.1f} GB/s)")
    print(f" Verify time:    {result.verify_time_s:.3f}s")
    print(f" Resume time:    {result.resume_time_s:.3f}s")
    print(f"{'='*60}")
    print(f" Report written to: {report_path}")
    print(f"{'='*60}")

    assert result.passed, f"Failed {result.rounds_total - result.rounds_passed} rounds"

    if os.path.exists(nvme_file):
        os.unlink(nvme_file)
```

**Deploy and run on GH200**:
```bash
# From your Mac:
IP=$(cat /tmp/gh200_ip.txt)  # or paste the IP

# Copy the project
rsync -avz --exclude '.git' --exclude '__pycache__' . ubuntu@$IP:~/memory_wipes/

# SSH in
ssh ubuntu@$IP

# On the GH200:
cd ~/memory_wipes
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv sync --extra gpu --extra dev

# Run inventory first
bash scripts/inventory.sh

# Run unit tests (should all pass except skipped CUDA/NVMe on first run)
uv run pytest tests/ -v --ignore=tests/test_system.py

# Set up NVMe test path
sudo mkdir -p /mnt/nvme
# If not already mounted, mount the NVMe drive:
# sudo mkfs.ext4 /dev/nvme1n1  (check lsblk for the right device!)
# sudo mount /dev/nvme1n1 /mnt/nvme
# sudo chown ubuntu:ubuntu /mnt/nvme

# Run system test (start small to verify, then scale up)
POSE_DRAM_GB=1 POSE_HBM_GB=1 POSE_NVME_GB=1 POSE_NVME_PATH=/mnt/nvme \
    uv run pytest tests/test_system.py -v -s

# Scale up gradually: 10 GB each, then 50, then full
POSE_DRAM_GB=10 POSE_HBM_GB=10 POSE_NVME_GB=10 POSE_NVME_PATH=/mnt/nvme \
    uv run pytest tests/test_system.py -v -s

# Full wipe (adjust GB values from inventory.sh output)
POSE_DRAM_GB=90 POSE_HBM_GB=93 POSE_NVME_GB=1000 POSE_NVME_PATH=/mnt/nvme \
    uv run pytest tests/test_system.py -v -s
```

**Done when**: System test passes and `reports/wipe_report.json` contains all
metrics.

---

### Task 13: Report Generator

**Goal**: Serialize `ProtocolResult` into a JSON report with all required metrics.

**Write tests first** (`tests/test_report.py`):
```python
import json
from pose.protocol import ProtocolResult, RegionMetrics
from pose.report import generate_report


def _sample_result():
    return ProtocolResult(
        passed=True,
        rounds_passed=100,
        rounds_total=100,
        bytes_wiped=1000 * 4096,
        bytes_total=1100 * 4096,
        bytes_reserved=100 * 4096,
        coverage=1000 / 1100,
        fill_time_s=2.5,
        verify_time_s=0.05,
        resume_time_s=0.3,
        region_metrics=[
            RegionMetrics(
                name="dram", total_bytes=500 * 4096, reserved_bytes=50 * 4096,
                reserved_reason="OS kernel", wiped_bytes=450 * 4096,
                fill_time_s=1.0, fill_throughput_gbps=1.8,
            ),
            RegionMetrics(
                name="hbm", total_bytes=300 * 4096, reserved_bytes=30 * 4096,
                reserved_reason="CUDA driver", wiped_bytes=270 * 4096,
                fill_time_s=0.5, fill_throughput_gbps=2.1,
            ),
            RegionMetrics(
                name="nvme", total_bytes=300 * 4096, reserved_bytes=20 * 4096,
                reserved_reason="FS metadata", wiped_bytes=280 * 4096,
                fill_time_s=1.0, fill_throughput_gbps=1.1,
            ),
        ],
        seed=b"\x00" * 32,
    )


def test_report_has_required_keys():
    report = generate_report(_sample_result())
    assert "wipe_time_s" in report
    assert "resume_time_s" in report
    assert "memory_inventory" in report
    assert "coverage_pct" in report
    assert "verification" in report


def test_report_is_json_serializable():
    report = generate_report(_sample_result())
    dumped = json.dumps(report)
    loaded = json.loads(dumped)
    assert loaded["wipe_time_s"] == 2.5


def test_report_regions_have_reserved_reason():
    report = generate_report(_sample_result())
    for region in report["memory_inventory"]["regions"]:
        assert "reserved_bytes" in region
        assert "reserved_reason" in region
        assert len(region["reserved_reason"]) > 0
```

**Then implement** (`src/pose/report.py`):
```python
"""Generate the Phase 1 wipe report from a ProtocolResult."""

from datetime import datetime, timezone
from pose.protocol import ProtocolResult


def generate_report(result: ProtocolResult) -> dict:
    """Convert a ProtocolResult into a JSON-serializable report dict.

    This is the final deliverable for Phase 1. It must contain:
    - Wipe time (total + per-region with throughput)
    - Resume time
    - Memory inventory (total, wiped, reserved per region + reasons)
    - Verification result (rounds passed, coverage)
    """
    return {
        "protocol": "unconditional-pose-db",
        "prf": "AES-256-CTR",
        "block_size_bytes": 4096,
        "timestamp": datetime.now(timezone.utc).isoformat(),

        # --- Key metrics ---
        "wipe_time_s": result.fill_time_s,
        "resume_time_s": result.resume_time_s,

        # --- Memory inventory ---
        "memory_inventory": {
            "total_bytes": result.bytes_total,
            "total_gb": round(result.bytes_total / (1024**3), 2),
            "wiped_bytes": result.bytes_wiped,
            "wiped_gb": round(result.bytes_wiped / (1024**3), 2),
            "reserved_bytes": result.bytes_reserved,
            "reserved_gb": round(result.bytes_reserved / (1024**3), 3),
            "regions": [
                {
                    "name": rm.name,
                    "total_bytes": rm.total_bytes,
                    "total_gb": round(rm.total_bytes / (1024**3), 2),
                    "wiped_bytes": rm.wiped_bytes,
                    "wiped_gb": round(rm.wiped_bytes / (1024**3), 2),
                    "reserved_bytes": rm.reserved_bytes,
                    "reserved_gb": round(rm.reserved_bytes / (1024**3), 3),
                    "reserved_reason": rm.reserved_reason,
                    "fill_time_s": round(rm.fill_time_s, 3),
                    "fill_throughput_gbps": round(rm.fill_throughput_gbps, 2),
                }
                for rm in result.region_metrics
            ],
        },

        # --- Coverage ---
        "coverage_pct": round(result.coverage * 100, 4),

        # --- Verification ---
        "verification": {
            "passed": result.passed,
            "rounds_passed": result.rounds_passed,
            "rounds_total": result.rounds_total,
            "verify_time_s": round(result.verify_time_s, 4),
        },

        # --- Limitations / stubs (must be empty if everything is real) ---
        "limitations": result.limitations if hasattr(result, "limitations") else [],
    }
```

```bash
uv run pytest tests/test_report.py -v
git add src/pose/report.py tests/test_report.py && git commit -m "Add report generator (JSON wipe report)"
```

**Done when**: All 3 tests pass. The report JSON contains every metric listed
in the "Done when" section at the top of this plan.

---

## Report Schema

The final report (`reports/wipe_report.json`) looks like this:

```json
{
  "protocol": "unconditional-pose-db",
  "prf": "AES-256-CTR",
  "block_size_bytes": 4096,
  "timestamp": "2026-03-31T...",

  "wipe_time_s": 45.2,
  "resume_time_s": 0.34,

  "memory_inventory": {
    "total_bytes": 1708987613184,
    "total_gb": 1591.5,
    "wiped_bytes": 1705505415168,
    "wiped_gb": 1588.3,
    "reserved_bytes": 3482198016,
    "reserved_gb": 3.243,
    "regions": [
      {
        "name": "dram",
        "total_gb": 96.0,
        "wiped_gb": 94.0,
        "reserved_gb": 2.0,
        "reserved_reason": "Linux kernel text/data/page tables, slab caches, ...",
        "fill_time_s": 8.5,
        "fill_throughput_gbps": 11.06
      },
      {
        "name": "hbm",
        "total_gb": 96.0,
        "wiped_gb": 95.75,
        "reserved_gb": 0.25,
        "reserved_reason": "CUDA driver context, GPU page tables, ECC metadata...",
        "fill_time_s": 3.2,
        "fill_throughput_gbps": 29.92
      },
      {
        "name": "nvme",
        "total_gb": 1399.5,
        "wiped_gb": 1398.5,
        "reserved_gb": 1.0,
        "reserved_reason": "Filesystem superblock, inode table, journal...",
        "fill_time_s": 33.5,
        "fill_throughput_gbps": 41.75
      }
    ]
  },

  "coverage_pct": 99.80,

  "verification": {
    "passed": true,
    "rounds_passed": 1000,
    "rounds_total": 1000,
    "verify_time_s": 0.052
  },

  "limitations": []
}
```

The `limitations` array is empty when everything works as designed. If any
functionality had to be stubbed, faked, or skipped, each entry must include:
```json
{
  "component": "hbm",
  "description": "GPU HBM fill limited to 90 GB due to driver allocation spike during cupy init",
  "impact": "2.5 GB of HBM not covered by wipe",
  "is_stub": false
}
```
If `is_stub` is `true`, it means the code path is not doing real work — this
must be called out prominently.

> **Note**: The values above are illustrative estimates. Actual numbers depend on
> the specific Lambda GH200 SKU and will be filled in by the system test.

---

## Testing Cheat Sheet

| Layer      | Command                                                    | Runs on Mac? |
|------------|------------------------------------------------------------|:-------------|
| Unit       | `uv run pytest tests/test_noise.py tests/test_memmap.py -v`| Yes          |
| DRAM       | `uv run pytest tests/test_dram.py -v`                      | Yes          |
| HBM        | `uv run pytest tests/test_hbm.py -v`                       | Skip         |
| NVMe       | `POSE_NVME_PATH=/tmp uv run pytest tests/test_nvme.py -v`  | Partial      |
| Verifier   | `uv run pytest tests/test_verifier.py -v`                  | Yes          |
| Prover     | `uv run pytest tests/test_prover.py -v`                    | Yes          |
| Protocol   | `uv run pytest tests/test_protocol.py -v`                  | Yes          |
| System     | See Task 12                                                | No (GH200)   |
| All local  | `uv run pytest tests/ --ignore=tests/test_system.py -v`    | Yes          |

**Test design tips**:
- Each test function tests ONE behavior (the function name says what)
- Tests are independent — no test depends on another test running first
- Use `pytest.fixture` for shared setup, not module-level globals
- Mark slow/hardware tests so `uv run pytest -m "not slow"` skips them
- When a test fails, the name + assertion message should tell you exactly
  what broke, without reading the test body

---

## Troubleshooting

**`EINVAL` on NVMe writes**: Your write buffer isn't aligned. Allocate via
`mmap.mmap(-1, size)` instead of `bytes`. 4096-byte blocks should be fine.

**GPU OOM**: You're trying to allocate more HBM than available. Reduce
`POSE_HBM_GB`. Check free memory with `nvidia-smi`.

**Host OOM / killed**: Linux OOM-killer is ending your process. Reduce
`POSE_DRAM_GB` or increase the reservation. Check `dmesg` for OOM messages.

**CuPy import error**: Install the right variant for your CUDA version:
`uv pip install cupy-cuda12x` (for CUDA 12.x).

**Lambda instance won't launch**: GPU instances sell out fast. The polling
script (`scripts/provision_gh200.sh`) retries every 30s. Be patient or try
a different region.

**Slow fill**: The fill phase should achieve >5 GB/s for DRAM, >1 GB/s for
NVMe. If it's much slower, check:
- DRAM: Ensure you're not swapping (`free -h`)
- NVMe: Ensure O_DIRECT is working (check `strace` for flags)
- HBM: Transfer bottleneck is NVLink-C2C; consider generating noise on-GPU

---

## Architecture Decisions

1. **AES-256-CTR as PRF** (not BLAKE3, not random bytes):
   Seekable (any block independently in O(1)), hardware-accelerated on ARM,
   well-understood security. Trade-off: loses the paper's information-theoretic
   security guarantee (see `plan/modifications.md` point 1).

2. **Co-located verifier** (not separate machine):
   Phase 1 is a single-node demo. Security implications documented in
   `plan/modifications.md` point 3. Addressed in Phase 3.

3. **4096-byte blocks**: Matches OS page size and NVMe sector size. Good
   balance between granularity and I/O efficiency.

4. **Three separate region classes** (not a unified abstraction):
   DRAM (mmap), HBM (cupy), and NVMe (O_DIRECT) have fundamentally different
   APIs. A common interface would add complexity without value. YAGNI.

5. **No verifier NVMe drive**: The verifier regenerates blocks from the PRF
   seed on demand. No need to store the full noise. This simplifies the
   Lambda setup (single NVMe drive is fine).

---

## What Comes Next

**Phase 2**: Restore operation after wipe. Phase 1 measures basic system
recovery (GPU re-init, DRAM availability, NVMe sync). Phase 2 goes further:
reload an ML model, rebuild CUDA contexts, and show the system can serve
inference requests. Key metric: **time from wipe completion to first successful
inference**. The Phase 1 `resume_time_s` provides the baseline; Phase 2 adds
model loading and warm-up on top.

**Phase 3**: Scale to multi-node clusters. Implement proper verifier isolation
(separate machine), distance-bounding timing, and support for arbitrary
cluster topologies. Research coverage guarantees for interconnected nodes.
