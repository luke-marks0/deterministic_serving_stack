# DPDK Egress Integrity Implementation Plan

**Goal:** Transmit the deterministic L2 frames that we already construct in
`pkg/networkdet/` out a real NIC via kernel bypass (DPDK), and verify that
the frames leave the node bit-for-bit unaltered.

**Status:** Phase 4 of the rollout described in ADR-0004.

**Prerequisite reading:** Before touching any code, read these files in order:

1. `docs/adr/0004-deterministic-userspace-networking.md` — the architecture
   decision that this plan implements.
2. `pkg/networkdet/backend_base.py` — the 4-method backend interface you will
   implement.
3. `pkg/networkdet/backend_sim.py` — the reference backend (37 lines). Your
   DPDK backend mirrors this structure.
4. `pkg/networkdet/__init__.py` — `DeterministicNetStack` facade and the
   `create_net_stack()` factory. This is where the backend is selected.
5. `pkg/networkdet/frame.py` — `DeterministicFrameBuilder`. Builds L2
   frames and records them in the capture ring. You do not modify this file.
6. `pkg/networkdet/capture.py` — `CaptureRing`. Pre-enqueue mirror buffer.
   You do not modify this file.
7. `pkg/networkdet/warden.py` — `ActiveWarden`. MRF normalizer. Understands
   TCP state, rewrites ISNs, strips options, recomputes checksums. In the
   DPDK path it serves as an assertion: if the warden changes any frame, the
   frame builder has a bug.
8. `cmd/runner/main.py` lines 370-530 — how the runner generates frames and
   writes the run bundle. You will modify this file.
9. `schemas/run_bundle.v1.schema.json` — the run bundle schema. You will
   extend it.
10. `tests/unit/test_networkdet_frame_builder.py` — the existing test
    patterns you should follow.

---

## 1. Domain Glossary

You will encounter these terms throughout the codebase. If you do not
understand a term, look it up before proceeding.

| Term | Meaning |
|------|---------|
| **MRF** | Minimal Requisite Fidelity. Every protocol field is pinned to the minimum entropy required for correct operation. This eliminates covert channels and ensures determinism. |
| **ISN** | Initial Sequence Number. TCP's starting sequence number. Normally random; we derive it from `sha256(run_id + conn_index)`. |
| **Capture ring** | An append-only buffer that records a copy of every L2 frame before it is transmitted. Non-perturbing: recording does not affect the frame or its ordering. |
| **Warden** | An inline normalizer that enforces MRF on arbitrary frames. Based on the Fisk et al. paper on active wardens. In our DPDK path, it is used as a **verifier** (assert frames are already MRF-compliant), not a mutator. |
| **DPDK** | Data Plane Development Kit. A set of C libraries that let userspace code drive a NIC directly, bypassing the kernel. |
| **PMD** | Poll-Mode Driver. DPDK's driver model. The NIC is polled in a tight loop instead of using interrupts. |
| **mlx5** | The DPDK PMD for NVIDIA/Mellanox ConnectX NICs. Supports bifurcated mode (DPDK fast path + kernel management). |
| **EAL** | Environment Abstraction Layer. DPDK's initialization subsystem (hugepages, CPU affinity, PCI device binding). |
| **mbuf** | Memory buffer. DPDK's packet buffer structure. Pre-allocated from a memory pool. |
| **TX completion** | Confirmation from the NIC that a frame has been DMA'd and transmitted. In DPDK, you poll for completions via `rte_eth_tx_done_cleanup()` or by checking mbuf refcounts. |
| **Run bundle** | The output artifact of a deterministic run. Contains tokens, logits, network frames, and attestation digests. |
| **Bifurcated mode** | mlx5-specific. The kernel retains the NIC for management traffic (SSH, ARP) while DPDK gets a fast path for your flows via flow steering rules. You do not lose SSH access. |
| **Hugepages** | 2MB or 1GB memory pages. DPDK requires these for DMA-safe memory pools. |
| **vfio-pci** | Kernel module that provides safe userspace access to PCI devices via IOMMU. Used by most DPDK PMDs (but NOT mlx5, which uses bifurcated mode). |
| **TxReport** | A data structure we will create. Contains the pre-enqueue digest, TX completion digest, and match status. |
| **Loopback verification** | Transmitting on one NIC port and receiving on another (or the same NIC in loopback mode) to verify frame integrity through the hardware path. |

---

## 2. Codebase Map

```
deterministic_serving_stack/
  pkg/networkdet/               <-- The deterministic networking stack
    __init__.py                 <-- DeterministicNetStack facade + create_net_stack() factory
    backend_base.py             <-- NetworkBackend ABC (4 methods)
    backend_sim.py              <-- SimulatedBackend (in-memory loopback)
    backend_dpdk.py             <-- YOU CREATE THIS (DPDK backend)
    capture.py                  <-- CaptureRing (pre-enqueue mirror)
    checksums.py                <-- RFC 1071 software checksums (IPv4 + TCP)
    config.py                   <-- NetStackConfig dataclass + parse_net_config()
    ethernet.py                 <-- build_ethernet_frame()
    frame.py                    <-- DeterministicFrameBuilder
    ip.py                       <-- DeterministicIPLayer
    tcp.py                      <-- DeterministicTCPConnection + deterministic_isn()
    warden.py                   <-- ActiveWarden (MRF normalizer)
    warden_config.py            <-- WardenConfig dataclass
    warden_service.py           <-- NFQUEUE-based warden daemon
    tx_report.py                <-- YOU CREATE THIS (TxReport dataclass)
  native/                       <-- YOU CREATE THIS DIRECTORY
    libnetdet/                  <-- C library wrapping DPDK
      CMakeLists.txt
      src/netdet.c              <-- EAL init, port setup, TX/RX, digest
      src/netdet.h              <-- Public API header
      tests/test_netdet.c       <-- C-level tests
  cmd/
    runner/main.py              <-- Runner CLI (you modify lines ~380-500)
    server/main.py              <-- Server proxy (reference for warden usage)
    capture/main.py             <-- Capture log converter
    verifier/main.py            <-- Run bundle verifier
  schemas/
    run_bundle.v1.schema.json   <-- Run bundle schema (you extend)
    manifest.v1.schema.json     <-- Manifest schema (read-only)
    lockfile.v1.schema.json     <-- Lockfile schema (read-only)
  tests/
    unit/
      test_networkdet_*.py      <-- Existing unit tests (your reference)
      test_tx_report.py         <-- YOU CREATE THIS
      test_backend_dpdk.py      <-- YOU CREATE THIS
    integration/
      test_dpdk_egress.py       <-- YOU CREATE THIS (requires DPDK hardware)
  docs/
    adr/0004-...                <-- ADR you update at the end
  Makefile                      <-- CI gate targets
```

---

## 3. Design

### 3.1 Frame pipeline with DPDK

```
Inference output
    |
    v
canonical_json_bytes(response)
    |
    v
DeterministicNetStack.process_response()
    |
    v
DeterministicFrameBuilder.build_response_frames()
    |  (segments data at MSS boundary, wraps in TCP/IP/Eth)
    |  (records each frame in CaptureRing)
    v
ActiveWarden.normalize(frame)   <-- ASSERT: must be a no-op
    |                               If it changes the frame, the builder
    |                               has a bug. Raise, don't silently fix.
    v
DPDKBackend.send_frame(frame)   <-- Copies frame bytes into an mbuf,
    |                               enqueues to DPDK TX ring
    v
DPDKBackend.flush() -> TxReport
    |  (calls rte_eth_tx_burst to push mbufs to NIC)
    |  (polls TX completions)
    |  (computes SHA-256 over confirmed-transmitted mbufs)
    v
TxReport:
    pre_enqueue_digest   = CaptureRing.digest()   (from Python)
    tx_completion_digest = libnetdet SHA-256       (from C, over DMA'd bytes)
    match                = (pre_enqueue == tx_completion)
```

### 3.2 Why DPDK, not AF_XDP

DPDK gives us the strongest guarantee: the frame bytes go from our userspace
buffer directly to the NIC's DMA engine with nothing in between. AF_XDP
still passes through the kernel's XDP layer (inside the driver), which is an
additional intermediary we cannot fully audit. For an egress integrity proof,
minimizing intermediaries is the whole point.

mlx5's bifurcated mode solves the practical problem: you keep SSH access
because the kernel still handles management traffic. DPDK only owns the
fast path.

### 3.3 `libnetdet.so` — the C library

A thin C library that wraps DPDK. It does NOT build frames — Python does
that. It only:

1. Initializes DPDK EAL and opens a port
2. Copies pre-built frame bytes into mbufs and transmits them
3. Computes SHA-256 over transmitted mbuf data for TX completion verification
4. Receives frames from an RX port (for loopback verification)
5. Cleans up

The API is deliberately small:

```c
// Opaque context
typedef struct netdet_ctx netdet_ctx;

// TX result: how many frames confirmed, SHA-256 digest
typedef struct {
    int confirmed;           // frames confirmed transmitted
    int submitted;           // frames submitted
    uint8_t digest[32];      // SHA-256 over confirmed frame bytes
} netdet_tx_result;

// RX result: received frames + digest
typedef struct {
    int count;               // frames received
    uint8_t** frames;        // array of frame pointers (caller frees)
    uint16_t* lengths;       // array of frame lengths
    uint8_t digest[32];      // SHA-256 over received frame bytes
} netdet_rx_result;

// Init: parse EAL args, set up port, allocate mempool
netdet_ctx* netdet_init(int argc, char** argv, uint16_t port_id);

// Transmit pre-built frames, wait for completion, return digest
netdet_tx_result netdet_send(netdet_ctx* ctx,
                             const uint8_t** frames,
                             const uint16_t* lengths,
                             int count);

// Receive frames (blocking up to timeout_ms)
netdet_rx_result netdet_recv(netdet_ctx* ctx, int timeout_ms);

// Free RX result buffers
void netdet_rx_free(netdet_rx_result* result);

// Teardown
void netdet_close(netdet_ctx* ctx);
```

### 3.4 Verification levels

**Level 1 — TX completion** (baseline, always runs with DPDK backend):

```
CaptureRing.digest() == libnetdet TX completion digest
```

Proves: the NIC's DMA engine received exactly the bytes we intended.

**Level 2 — Loopback** (optional, requires two ports or NIC loopback mode):

```
CaptureRing.digest() == TX completion digest == RX loopback digest
```

Proves: the bytes survived the entire hardware TX→wire→RX path.

### 3.5 Warden as assertion, not mutator

In the current server (`cmd/server/main.py` lines 728-731), the warden is
used as a real normalizer on frames coming from vLLM. That's a different
context: the server proxies frames from the kernel TCP stack, which may have
nondeterministic fields.

In the DPDK path, frames are built by our deterministic stack. They should
already be MRF-compliant. The warden is used as an **assertion**:

```python
normalized = warden.normalize(frame)
assert normalized == frame, (
    f"Frame builder produced non-MRF-compliant frame at index {i}. "
    f"Warden changed {_diff_frames(frame, normalized)} bytes."
)
```

If this assertion fires, the bug is in the frame builder, not the transmit
path. Fix the builder.

### 3.6 Run bundle schema extension

Add an optional `egress_verification` object to the run bundle:

```json
{
  "network_provenance": {
    "capture_path": "observables/network_egress.json",
    "capture_digest": "sha256:abc...",
    "frame_count": 42,
    "capture_mode": "userspace_pre_enqueue",
    "capture_isolation": "pre_enqueue_mirror",
    "capture_non_perturbing": true,
    "route_mode": "deterministic_userspace_stack",
    "egress_verification": {
      "backend": "dpdk",
      "level": "tx_completion",
      "pre_enqueue_digest": "sha256:abc...",
      "tx_completion_digest": "sha256:abc...",
      "rx_loopback_digest": "sha256:abc...",
      "frames_submitted": 42,
      "frames_confirmed": 42,
      "match": true
    }
  }
}
```

When the sim backend is used, `egress_verification` is absent (no real
transmission occurred).

---

## 4. Testing Rules

Follow these rules for every task. If you are unsure whether a test is
needed, write the test.

### 4.1 TDD cycle

1. Write a failing test first.
2. Write the minimum code to make it pass.
3. Refactor if the code smells. The test still passes.
4. Commit.

### 4.2 What to test

- **Every public method** gets at least one test.
- **Every branch** in error handling gets a test (invalid input, missing
  library, uninitialized state).
- **Determinism tests**: call the same function twice with the same inputs.
  Assert the outputs are identical. This pattern is already used in
  `test_networkdet_frame_builder.py::test_determinism_across_builders`.
- **Boundary tests**: empty input, single frame, exactly-MSS payload,
  one-byte-over-MSS payload.

### 4.3 What NOT to test

- Do not test DPDK internals (rte_* functions). Those are tested by DPDK.
- Do not test the frame builder or capture ring. Those already have tests.
- Do not write integration tests that require DPDK hardware in the unit test
  suite. Mark those with `@unittest.skipUnless(DPDK_AVAILABLE, "requires DPDK")`.

### 4.4 Test file naming

Follow the existing pattern:
- `tests/unit/test_<module_name>.py` for unit tests
- `tests/integration/test_<feature>.py` for integration tests

### 4.5 Test helpers

The existing tests use a `_test_config()` helper that returns a
`NetStackConfig` with sensible defaults. Reuse it. Do not create a new one.
If you need the helper in multiple test files, extract it to
`tests/unit/conftest.py` or a shared module — but only if you actually
reuse it in 3+ files. Until then, copy the 15-line function.

### 4.6 Running tests

```bash
# Run all unit tests
python -m pytest tests/unit/ -v

# Run a specific test file
python -m pytest tests/unit/test_tx_report.py -v

# Run with the CI gate (lint + schema + tests)
make ci-pr
```

---

## 5. Tasks

Each task has:
- **What**: a description of the deliverable
- **Files**: which files to create or modify
- **Tests**: what tests to write
- **Commit**: when to commit and a suggested message prefix
- **Do NOT**: explicit things to avoid

---

### Task 0: Identify NIC hardware on GH200

**What:** SSH into a Lambda Cloud GH200 instance and identify the network
interface. We need to know: NIC vendor/model, kernel driver, PCI address,
and whether mlx5 is available.

**Commands to run:**

```bash
# PCI devices — look for "Ethernet controller"
lspci | grep -i ethernet

# Driver in use
ethtool -i eth0    # or whatever the interface name is

# Detailed NIC info
lspci -vvv -s <pci_address>

# Check for mlx5 kernel module
lsmod | grep mlx5

# Check DPDK-compatible drivers
ls /sys/bus/pci/drivers/mlx5_core/
```

**Record:** Save the output in `.internal/recon/gh200-nic-info.txt`. We need:
- NIC model (e.g., ConnectX-7, BlueField-3)
- PCI address (e.g., `0000:01:00.0`)
- Current kernel driver (e.g., `mlx5_core`)
- Number of physical ports
- Whether the NIC supports loopback mode

**Commit:** `recon: identify NIC hardware on Lambda GH200`

**Do NOT:** Install anything yet. This is read-only reconnaissance.

---

### Task 1: `TxReport` data class

**What:** Create a frozen dataclass that holds the result of a DPDK
transmit operation. This is used by the DPDK backend and by the runner
to populate the run bundle.

**Files:**
- Create `pkg/networkdet/tx_report.py`
- Create `tests/unit/test_tx_report.py`

**Implementation:**

```python
# pkg/networkdet/tx_report.py
"""Transmission report for egress integrity verification."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TxReport:
    """Result of transmitting frames through a real backend.

    Both digests use the ``sha256:<hex>`` format used elsewhere in
    the codebase (see CaptureRing.digest()).
    """

    pre_enqueue_digest: str
    tx_completion_digest: str
    frames_submitted: int
    frames_confirmed: int

    @property
    def match(self) -> bool:
        return self.pre_enqueue_digest == self.tx_completion_digest

    @property
    def level(self) -> str:
        return "tx_completion"
```

**Tests** (`tests/unit/test_tx_report.py`):

Write these tests:

1. `test_match_when_digests_equal` — construct a TxReport with identical
   digests, assert `match` is True.
2. `test_no_match_when_digests_differ` — different digests, assert `match`
   is False.
3. `test_frozen` — assert that assigning to a field raises
   `FrozenInstanceError`.
4. `test_level_is_tx_completion` — assert the `level` property returns
   `"tx_completion"`.

**Commit:** `feat(networkdet): add TxReport dataclass for egress verification`

**Do NOT:**
- Add a `rx_loopback_digest` field yet. YAGNI — add it in Task 12 when
  you implement loopback.
- Add serialization methods. The runner will read the fields directly.

---

### Task 2: Extend `NetworkBackend` with `flush()`

**What:** Add a `flush()` method to `NetworkBackend` that returns an
optional `TxReport`. The sim backend returns `None` (no real transmission).
The DPDK backend (Task 7) will override it.

**Files:**
- Modify `pkg/networkdet/backend_base.py`
- Modify `pkg/networkdet/backend_sim.py`

**Changes to `backend_base.py`:**

Add one method to the ABC:

```python
def flush(self) -> TxReport | None:
    """Flush pending frames and return a transmission report.

    Returns None for backends that do not support egress verification
    (e.g., the simulated backend).
    """
    return None
```

This is a **concrete method with a default**, not an abstract method. The
sim backend inherits it unchanged.

Add the import: `from pkg.networkdet.tx_report import TxReport`

**Changes to `backend_sim.py`:** None. It inherits the default `flush()`.

**Tests:** The existing sim backend tests already cover `send_frame` and
`recv_frame`. Add one test to the existing test file (or create
`tests/unit/test_backend_sim.py` if one doesn't exist):

1. `test_flush_returns_none` — call `flush()` on an initialized
   SimulatedBackend, assert it returns None.

**Commit:** `feat(networkdet): add flush() to NetworkBackend interface`

**Do NOT:**
- Change `send_frame()` signature.
- Add any batching logic to the sim backend.

---

### Task 3: Add warden assertion to `DeterministicNetStack`

**What:** When the backend is not `sim`, run each frame through the warden
before transmitting. Assert that the warden does not change the frame. This
catches frame builder bugs before they reach the NIC.

**Files:**
- Modify `pkg/networkdet/__init__.py`

**Changes:**

Add the warden to `DeterministicNetStack.__init__()`:

```python
from pkg.networkdet.warden import ActiveWarden

class DeterministicNetStack:
    def __init__(self, config, *, run_id, backend=None, verify_mrf=False):
        # ... existing init ...
        self._verify_mrf = verify_mrf
        self._warden = ActiveWarden() if verify_mrf else None
```

Modify `process_response()`:

```python
def process_response(self, conn_index, response_bytes):
    builder = self._get_builder(conn_index)
    frames = builder.build_response_frames(response_bytes)
    for frame in frames:
        if self._warden is not None:
            normalized = self._warden.normalize(frame)
            if normalized != frame:
                raise RuntimeError(
                    f"Frame builder produced non-MRF-compliant frame "
                    f"(conn_index={conn_index}). The warden modified "
                    f"{sum(a != b for a, b in zip(frame, normalized))} bytes. "
                    f"This is a bug in the frame builder."
                )
        self._backend.send_frame(frame)
    return frames
```

Wire `verify_mrf=True` in `create_net_stack()` when `backend == "dpdk"`.

**Tests** (add to existing tests or a new file):

1. `test_mrf_verification_passes_for_valid_frames` — create a
   `DeterministicNetStack` with `verify_mrf=True` and a sim backend.
   Call `process_response()`. Should not raise.
2. `test_mrf_verification_catches_bad_frames` — mock the frame builder to
   produce a frame with a nonzero reserved bit. Assert `RuntimeError` is
   raised with "non-MRF-compliant" in the message.

**Commit:** `feat(networkdet): add MRF verification assertion to net stack`

**Do NOT:**
- Move `capture.record()` out of the frame builder. The existing capture
  ring records what the builder produced, and the warden asserts that is
  exactly what will be transmitted.
- Enable `verify_mrf` for the sim backend by default. It adds overhead that
  isn't needed for simulation-only runs.

---

### Task 4: `libnetdet.so` scaffolding

**What:** Create the C project directory structure with CMake, a header
file defining the public API, and a stub implementation that compiles
but does not link against DPDK.

**Files:**
- Create `native/libnetdet/CMakeLists.txt`
- Create `native/libnetdet/src/netdet.h`
- Create `native/libnetdet/src/netdet.c` (stubs)
- Create `native/libnetdet/tests/test_netdet.c`
- Add `make build-libnetdet` target to `Makefile`

**`CMakeLists.txt`:**

```cmake
cmake_minimum_required(VERSION 3.16)
project(libnetdet C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Werror -O2")

# DPDK discovery via pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(DPDK REQUIRED libdpdk)

# OpenSSL for SHA-256
find_package(OpenSSL REQUIRED)

add_library(netdet SHARED src/netdet.c)
target_include_directories(netdet PUBLIC src ${DPDK_INCLUDE_DIRS})
target_link_libraries(netdet ${DPDK_LINK_LIBRARIES} OpenSSL::Crypto)
target_compile_options(netdet PRIVATE ${DPDK_CFLAGS_OTHER})

# Install
install(TARGETS netdet LIBRARY DESTINATION lib)
install(FILES src/netdet.h DESTINATION include)
```

**`netdet.h`:** The API from Section 3.3 of this document. Copy it exactly.

**`netdet.c`:** Implement all functions as stubs that return error codes or
zero-initialized structs. Each stub should print a message to stderr:
`fprintf(stderr, "netdet: %s not yet implemented\n", __func__);`

**Test:** The C test file should compile and link. It calls `netdet_init()`
and asserts it returns NULL (since the stub doesn't initialize DPDK).
Use a simple `assert()` + `main()` pattern, not a C test framework.

**Makefile target:**

```makefile
build-libnetdet:
	mkdir -p native/libnetdet/build && \
	cd native/libnetdet/build && \
	cmake .. && \
	make -j$$(nproc)
```

**Commit:** `feat(native): scaffold libnetdet C library with DPDK CMake`

**Do NOT:**
- Implement real DPDK calls yet. This task is just scaffolding.
- Add the library to the Nix closure yet. That is Task 13.
- Link against DPDK on your Mac. This only builds on Linux with DPDK
  installed. Add a note to CMakeLists.txt explaining this.

---

### Task 5: `libnetdet.so` — EAL init + port setup

**What:** Implement `netdet_init()` and `netdet_close()`. This is the
first real DPDK code.

**Files:**
- Modify `native/libnetdet/src/netdet.c`

**Implementation details:**

`netdet_init()` must:

1. Call `rte_eal_init(argc, argv)` to initialize DPDK. EAL args are
   passed from Python (e.g., `--no-huge` for testing, `--socket-mem`,
   `-l` for core list).
2. Create a mempool: `rte_pktmbuf_pool_create("NETDET_POOL", 8191,
   250, 0, RTE_MBUF_DEFAULT_BUF_SIZE, rte_socket_id())`.
3. Configure the port:
   - 1 TX queue, 1 RX queue (matches `NetStackConfig.tx_queues`)
   - **Disable all offloads**: set `txmode.offloads = 0` and
     `rxmode.offloads = 0`. This is critical — any offload can mutate
     frames.
   - Set MTU to 1500.
4. Set up TX queue: `rte_eth_tx_queue_setup(port_id, 0, 512, ...)`.
5. Set up RX queue: `rte_eth_rx_queue_setup(port_id, 0, 512, ...)`.
6. Start the port: `rte_eth_dev_start(port_id)`.
7. Enable promiscuous mode (for loopback capture):
   `rte_eth_promiscuous_enable(port_id)`.
8. Return a heap-allocated `netdet_ctx` struct containing port_id and
   mempool pointer.

`netdet_close()` must:
1. Stop the port: `rte_eth_dev_stop(port_id)`.
2. Close the port: `rte_eth_dev_close(port_id)`.
3. Free the mempool (if DPDK supports it; some versions don't).
4. Call `rte_eal_cleanup()`.
5. Free the `netdet_ctx`.

**Critical: Disable offloads.**

This deserves its own section because getting it wrong silently defeats
the entire purpose.

When configuring the port, explicitly zero out all offload flags:

```c
struct rte_eth_conf port_conf = {
    .rxmode = { .mq_mode = RTE_ETH_MQ_RX_NONE, .offloads = 0 },
    .txmode = { .mq_mode = RTE_ETH_MQ_TX_NONE, .offloads = 0 },
};
```

After starting the port, verify offloads are actually disabled:

```c
struct rte_eth_dev_info dev_info;
rte_eth_dev_info_get(port_id, &dev_info);
if (dev_info.tx_offload_capa & actually_enabled_offloads) {
    // Log a warning — the NIC may force some offloads
}
```

Offloads that MUST be disabled:
- `RTE_ETH_TX_OFFLOAD_IPV4_CKSUM` — IP checksum offload
- `RTE_ETH_TX_OFFLOAD_TCP_CKSUM` — TCP checksum offload
- `RTE_ETH_TX_OFFLOAD_TCP_TSO` — TCP segmentation offload
- `RTE_ETH_TX_OFFLOAD_VLAN_INSERT` — VLAN tag insertion
- `RTE_ETH_TX_OFFLOAD_MULTI_SEGS` — scatter-gather

**Tests:** This task cannot be unit-tested without DPDK hardware. Write the
test in `native/libnetdet/tests/test_netdet.c` but guard it:

```c
int main(void) {
    // This test requires a DPDK-compatible NIC.
    // Run with: ./test_netdet --no-huge -l 0 -- 0
    // where the last arg is the port ID.
    if (argc < 4) {
        printf("SKIP: no DPDK args provided\n");
        return 0;
    }
    netdet_ctx* ctx = netdet_init(argc - 1, argv, atoi(argv[argc-1]));
    assert(ctx != NULL);
    netdet_close(ctx);
    printf("PASS: init/close\n");
    return 0;
}
```

**Commit:** `feat(native): implement DPDK EAL init and port setup`

**Do NOT:**
- Enable any offloads "for performance." Every offload is a potential
  source of frame mutation.
- Use `--no-huge` in production. Hugepages are required for real
  performance and correct DMA behavior.
- Assume port 0. Always take port_id as a parameter.

---

### Task 6: `libnetdet.so` — TX path

**What:** Implement `netdet_send()`. This is the core of the library.

**Files:**
- Modify `native/libnetdet/src/netdet.c`

**Implementation:**

```c
netdet_tx_result netdet_send(netdet_ctx* ctx,
                             const uint8_t** frames,
                             const uint16_t* lengths,
                             int count) {
    netdet_tx_result result = {0};
    SHA256_CTX sha_ctx;
    SHA256_Init(&sha_ctx);

    // Allocate mbufs and copy frame data
    struct rte_mbuf* mbufs[count];
    for (int i = 0; i < count; i++) {
        mbufs[i] = rte_pktmbuf_alloc(ctx->pool);
        if (!mbufs[i]) {
            // Handle allocation failure — free already allocated mbufs
            for (int j = 0; j < i; j++) rte_pktmbuf_free(mbufs[j]);
            result.submitted = i;
            result.confirmed = 0;
            return result;
        }
        // Copy frame bytes into mbuf
        char* data = rte_pktmbuf_append(mbufs[i], lengths[i]);
        memcpy(data, frames[i], lengths[i]);
    }

    result.submitted = count;

    // Transmit burst
    uint16_t sent = rte_eth_tx_burst(ctx->port_id, 0, mbufs, count);

    // Hash the bytes of the mbufs that were actually sent
    for (int i = 0; i < sent; i++) {
        SHA256_Update(&sha_ctx,
                      rte_pktmbuf_mtod(mbufs[i], uint8_t*),
                      rte_pktmbuf_pkt_len(mbufs[i]));
    }

    // Free unsent mbufs (sent mbufs are freed by the driver after TX)
    for (int i = sent; i < count; i++) {
        rte_pktmbuf_free(mbufs[i]);
    }

    result.confirmed = sent;
    SHA256_Final(result.digest, &sha_ctx);
    return result;
}
```

**Critical detail: when is the mbuf data valid?**

After `rte_eth_tx_burst()` returns, the mbufs that were accepted are
**owned by the driver**. The driver will DMA the data and then free the
mbufs. However, the data pointer (`rte_pktmbuf_mtod`) is valid until the
mbuf is freed. Since we hash the data immediately after `tx_burst` and
before the driver completes the DMA, we are hashing what we *submitted*
to the TX ring, not necessarily what the NIC *sent*. For mlx5 in
particular, the PMD does a memcpy into the WQE (work queue entry) during
`tx_burst`, so the mbuf data is the source of truth.

If you want stronger guarantees, you can use the **TX completion fence**
approach: set `RTE_ETH_TX_OFFLOAD_MBUF_FAST_FREE` off (already off since
we disable all offloads), and poll `rte_eth_tx_done_cleanup()` to wait
for the NIC to confirm the DMA completed. Then hash.

For the initial implementation, hashing immediately after `tx_burst` is
sufficient. Document this trade-off in a comment.

**Tests:** Same gated approach as Task 5. Add to the C test:

```c
// Construct a minimal valid Ethernet frame (60 bytes min)
uint8_t frame[60] = {0};
// ... fill in Ethernet header ...
const uint8_t* frames[] = {frame};
uint16_t lengths[] = {60};
netdet_tx_result res = netdet_send(ctx, frames, lengths, 1);
assert(res.submitted == 1);
assert(res.confirmed == 1);
printf("PASS: send 1 frame\n");
```

**Commit:** `feat(native): implement DPDK TX path with SHA-256 digest`

**Do NOT:**
- Set any mbuf offload flags (`ol_flags`). Leave them at 0. Setting
  `RTE_MBUF_F_TX_IP_CKSUM` would tell the NIC to compute the checksum,
  which would overwrite our software checksum.
- Use scatter-gather mbufs. One contiguous mbuf per frame. Simple.
- Allocate mbufs with variable-length arrays on the stack for large frame
  counts. For > 256 frames, heap-allocate the mbuf pointer array.

---

### Task 7: `libnetdet.so` — RX path (for loopback)

**What:** Implement `netdet_recv()` and `netdet_rx_free()`. These are
used only for loopback verification (Level 2).

**Files:**
- Modify `native/libnetdet/src/netdet.c`

**Implementation:**

```c
netdet_rx_result netdet_recv(netdet_ctx* ctx, int timeout_ms) {
    netdet_rx_result result = {0};
    SHA256_CTX sha_ctx;
    SHA256_Init(&sha_ctx);

    struct rte_mbuf* mbufs[256];
    int total = 0;
    uint64_t deadline = rte_get_timer_cycles()
                      + (uint64_t)timeout_ms * rte_get_timer_hz() / 1000;

    // Collect frames: array of (pointer, length) pairs
    // Start with capacity 256, grow if needed
    int capacity = 256;
    result.frames = malloc(capacity * sizeof(uint8_t*));
    result.lengths = malloc(capacity * sizeof(uint16_t));

    while (rte_get_timer_cycles() < deadline) {
        uint16_t nb = rte_eth_rx_burst(ctx->port_id, 0, mbufs, 256);
        for (int i = 0; i < nb; i++) {
            uint16_t len = rte_pktmbuf_pkt_len(mbufs[i]);
            uint8_t* copy = malloc(len);
            memcpy(copy, rte_pktmbuf_mtod(mbufs[i], uint8_t*), len);

            if (total >= capacity) {
                capacity *= 2;
                result.frames = realloc(result.frames, capacity * sizeof(uint8_t*));
                result.lengths = realloc(result.lengths, capacity * sizeof(uint16_t));
            }
            result.frames[total] = copy;
            result.lengths[total] = len;
            total++;

            SHA256_Update(&sha_ctx, copy, len);
            rte_pktmbuf_free(mbufs[i]);
        }
        if (nb == 0) {
            // Brief pause to avoid spinning too hard
            rte_delay_us_block(100);
        }
    }

    result.count = total;
    SHA256_Final(result.digest, &sha_ctx);
    return result;
}

void netdet_rx_free(netdet_rx_result* result) {
    for (int i = 0; i < result->count; i++) {
        free(result->frames[i]);
    }
    free(result->frames);
    free(result->lengths);
    result->frames = NULL;
    result->lengths = NULL;
    result->count = 0;
}
```

**Commit:** `feat(native): implement DPDK RX path for loopback verification`

**Do NOT:**
- Filter frames in C. Return everything received. Python will filter and
  match.
- Use a ring buffer or other fancy structure. A growable array of heap
  copies is fine for tens-to-hundreds of frames.

---

### Task 8: Python ctypes bindings

**What:** Create a Python module that loads `libnetdet.so` via ctypes and
exposes the C API as Python functions.

**Files:**
- Create `pkg/networkdet/libnetdet_ffi.py`
- Create `tests/unit/test_libnetdet_ffi.py`

**Implementation:**

```python
"""ctypes bindings for libnetdet.so (DPDK transmit/receive)."""
from __future__ import annotations

import ctypes
import hashlib
import os
from ctypes import (
    POINTER, Structure, c_char_p, c_int, c_uint8, c_uint16, c_void_p,
)
from pathlib import Path


class TxResult(Structure):
    _fields_ = [
        ("confirmed", c_int),
        ("submitted", c_int),
        ("digest", c_uint8 * 32),
    ]

    @property
    def digest_hex(self) -> str:
        return bytes(self.digest).hex()

    @property
    def digest_prefixed(self) -> str:
        return f"sha256:{self.digest_hex}"


class RxResult(Structure):
    _fields_ = [
        ("count", c_int),
        ("frames", POINTER(POINTER(c_uint8))),
        ("lengths", POINTER(c_uint16)),
        ("digest", c_uint8 * 32),
    ]

    @property
    def digest_prefixed(self) -> str:
        return f"sha256:{bytes(self.digest).hex()}"


def _find_library() -> str:
    """Find libnetdet.so. Search order:
    1. LIBNETDET_PATH environment variable
    2. native/libnetdet/build/libnetdet.so (development)
    3. /usr/local/lib/libnetdet.so (installed)
    """
    env_path = os.environ.get("LIBNETDET_PATH")
    if env_path:
        return env_path

    dev_path = Path(__file__).resolve().parents[2] / "native/libnetdet/build/libnetdet.so"
    if dev_path.exists():
        return str(dev_path)

    return "libnetdet.so"  # Let ctypes search LD_LIBRARY_PATH


class LibNetDet:
    """Wrapper around libnetdet.so."""

    def __init__(self, lib_path: str | None = None):
        path = lib_path or _find_library()
        self._lib = ctypes.CDLL(path)
        self._setup_signatures()

    def _setup_signatures(self):
        # netdet_init
        self._lib.netdet_init.argtypes = [c_int, POINTER(c_char_p), c_uint16]
        self._lib.netdet_init.restype = c_void_p

        # netdet_send
        self._lib.netdet_send.argtypes = [
            c_void_p,
            POINTER(POINTER(c_uint8)),
            POINTER(c_uint16),
            c_int,
        ]
        self._lib.netdet_send.restype = TxResult

        # netdet_recv
        self._lib.netdet_recv.argtypes = [c_void_p, c_int]
        self._lib.netdet_recv.restype = RxResult

        # netdet_rx_free
        self._lib.netdet_rx_free.argtypes = [POINTER(RxResult)]
        self._lib.netdet_rx_free.restype = None

        # netdet_close
        self._lib.netdet_close.argtypes = [c_void_p]
        self._lib.netdet_close.restype = None

    def init(self, eal_args: list[str], port_id: int) -> int:
        """Initialize DPDK. Returns an opaque context handle (as int)."""
        argc = len(eal_args) + 1
        argv_type = c_char_p * argc
        argv = argv_type(b"netdet", *[a.encode() for a in eal_args])
        ctx = self._lib.netdet_init(argc, argv, port_id)
        if not ctx:
            raise RuntimeError("netdet_init failed — check DPDK EAL args and port ID")
        return ctx

    def send(self, ctx: int, frames: list[bytes]) -> TxResult:
        """Send pre-built L2 frames. Returns TX result with digest."""
        count = len(frames)
        frame_ptrs = (POINTER(c_uint8) * count)()
        lengths = (c_uint16 * count)()
        # Keep references to prevent GC
        buffers = []
        for i, frame in enumerate(frames):
            buf = (c_uint8 * len(frame))(*frame)
            buffers.append(buf)
            frame_ptrs[i] = ctypes.cast(buf, POINTER(c_uint8))
            lengths[i] = len(frame)
        return self._lib.netdet_send(ctx, frame_ptrs, lengths, count)

    def recv(self, ctx: int, timeout_ms: int = 1000) -> RxResult:
        """Receive frames (for loopback verification)."""
        return self._lib.netdet_recv(ctx, timeout_ms)

    def rx_free(self, result: RxResult) -> None:
        """Free RX result buffers."""
        self._lib.netdet_rx_free(ctypes.byref(result))

    def close(self, ctx: int) -> None:
        """Shut down DPDK port and EAL."""
        self._lib.netdet_close(ctx)
```

**Tests** (`tests/unit/test_libnetdet_ffi.py`):

These tests run WITHOUT DPDK. They test the Python-side logic only.

1. `test_find_library_env_override` — set `LIBNETDET_PATH` env var, assert
   `_find_library()` returns it.
2. `test_find_library_fallback` — unset env var, assert it returns a path
   (don't assert it exists — that's a runtime concern).
3. `test_tx_result_digest_prefixed` — construct a `TxResult` with known
   digest bytes, assert `digest_prefixed` is correct.
4. `test_init_raises_on_null_ctx` — mock the C library to return NULL from
   `netdet_init`, assert `RuntimeError` is raised.

**Commit:** `feat(networkdet): add ctypes bindings for libnetdet.so`

**Do NOT:**
- Import this module at the top level of any `pkg/networkdet/` file. It
  should only be imported inside `BackendDPDK.__init__()`. If libnetdet.so
  is not installed, the sim backend must still work.
- Handle partial sends or retries in Python. The C library handles that.

---

### Task 9: `BackendDPDK` implementation

**What:** Create the DPDK backend implementing `NetworkBackend`. This is
the Python-side glue between the network stack and `libnetdet.so`.

**Files:**
- Create `pkg/networkdet/backend_dpdk.py`
- Create `tests/unit/test_backend_dpdk.py`

**Implementation:**

```python
"""DPDK network backend for real NIC transmission."""
from __future__ import annotations

from pkg.networkdet.backend_base import NetworkBackend
from pkg.networkdet.config import NetStackConfig
from pkg.networkdet.tx_report import TxReport


class DPDKBackend(NetworkBackend):
    """Kernel-bypass backend via DPDK + libnetdet.so.

    Frames are buffered in Python and flushed to the NIC in a single
    burst via flush(). The simulated backend sends immediately in
    send_frame(); this backend batches for efficiency and to compute
    a single TX completion digest over the entire run.
    """

    def __init__(self, *, port_id: int = 0, eal_args: list[str] | None = None):
        self._port_id = port_id
        self._eal_args = eal_args or []
        self._ctx: int | None = None
        self._lib = None
        self._tx_buffer: list[bytes] = []
        self._initialised = False

    def init(self, config: NetStackConfig) -> None:
        if config.tso or config.gso or config.checksum_offload:
            raise RuntimeError(
                "DPDK backend requires all offloads disabled. "
                "Got tso=%s gso=%s checksum_offload=%s"
                % (config.tso, config.gso, config.checksum_offload)
            )
        # Lazy import — only load libnetdet when DPDK is actually used
        from pkg.networkdet.libnetdet_ffi import LibNetDet
        self._lib = LibNetDet()
        self._ctx = self._lib.init(self._eal_args, self._port_id)
        self._tx_buffer.clear()
        self._initialised = True

    def send_frame(self, frame: bytes) -> None:
        if not self._initialised:
            raise RuntimeError("DPDKBackend not initialised")
        self._tx_buffer.append(bytes(frame))

    def recv_frame(self) -> bytes | None:
        # For loopback verification, use recv_loopback() instead.
        # This method exists to satisfy the interface.
        return None

    def flush(self) -> TxReport | None:
        """Transmit all buffered frames and return an egress report."""
        if not self._initialised or not self._tx_buffer:
            return None

        import hashlib
        # Compute pre-enqueue digest (same algorithm as CaptureRing.digest)
        h = hashlib.sha256()
        for frame in self._tx_buffer:
            h.update(frame)
        pre_enqueue = f"sha256:{h.hexdigest()}"

        result = self._lib.send(self._ctx, self._tx_buffer)
        self._tx_buffer.clear()

        return TxReport(
            pre_enqueue_digest=pre_enqueue,
            tx_completion_digest=result.digest_prefixed,
            frames_submitted=result.submitted,
            frames_confirmed=result.confirmed,
        )

    def recv_loopback(self, timeout_ms: int = 1000) -> tuple[list[bytes], str]:
        """Receive frames for loopback verification.

        Returns (frames, digest) where digest is sha256:<hex>.
        """
        if not self._initialised:
            raise RuntimeError("DPDKBackend not initialised")
        result = self._lib.recv(self._ctx, timeout_ms)
        # Extract frame bytes before freeing
        frames = []
        for i in range(result.count):
            length = result.lengths[i]
            frame = bytes(result.frames[i][:length])
            frames.append(frame)
        digest = result.digest_prefixed
        self._lib.rx_free(result)
        return frames, digest

    def close(self) -> None:
        if self._ctx is not None:
            self._lib.close(self._ctx)
            self._ctx = None
        self._tx_buffer.clear()
        self._initialised = False
```

**Tests** (`tests/unit/test_backend_dpdk.py`):

These tests mock `libnetdet_ffi.LibNetDet` to avoid requiring DPDK
hardware.

1. `test_init_rejects_tso` — pass a config with `tso=True`, assert
   `RuntimeError`.
2. `test_init_rejects_checksum_offload` — same for `checksum_offload=True`.
3. `test_send_frame_buffers` — mock the library, call `send_frame()` 3
   times, assert no library calls were made (frames are buffered).
4. `test_flush_sends_all_buffered_frames` — mock `lib.send()` to return a
   `TxResult` with matching digest. Call `send_frame()` 3 times then
   `flush()`. Assert `lib.send()` was called once with 3 frames.
5. `test_flush_returns_tx_report` — assert the returned `TxReport` has
   correct digests and counts.
6. `test_flush_clears_buffer` — after `flush()`, the buffer is empty.
   Calling `flush()` again returns None.
7. `test_send_before_init_raises` — call `send_frame()` without `init()`.
   Assert `RuntimeError`.
8. `test_close_cleans_up` — call `close()`, assert the library's close was
   called and the backend is no longer initialised.

**Commit:** `feat(networkdet): implement DPDKBackend with libnetdet.so`

**Do NOT:**
- Import `libnetdet_ffi` at module level.
- Add retry logic for failed transmissions. If `rte_eth_tx_burst` doesn't
  send all frames, that's a configuration or hardware issue. Report it in
  the TxReport and let the caller decide.
- Implement `recv_frame()` for real. It exists only to satisfy the ABC.
  Loopback uses the separate `recv_loopback()` method.

---

### Task 10: Wire the DPDK backend into the factory

**What:** Replace the `NotImplementedError` in `create_net_stack()` with
actual `DPDKBackend` creation.

**Files:**
- Modify `pkg/networkdet/__init__.py`

**Changes:**

Replace lines 164-167 (the `elif backend == "dpdk"` branch):

```python
elif backend == "dpdk":
    from pkg.networkdet.backend_dpdk import DPDKBackend
    dpdk_port = kwargs.get("dpdk_port", 0)
    dpdk_eal_args = kwargs.get("dpdk_eal_args", [])
    be = DPDKBackend(port_id=dpdk_port, eal_args=dpdk_eal_args)
```

Add `**kwargs` to `create_net_stack()` signature for DPDK-specific args.

Set `verify_mrf=True` when backend is "dpdk":

```python
return DeterministicNetStack(
    config, run_id=effective_run_id, backend=be,
    verify_mrf=(backend == "dpdk"),
)
```

**Tests:**

1. `test_create_net_stack_dpdk_raises_without_library` — call
   `create_net_stack(..., backend="dpdk")` without libnetdet.so installed.
   Assert it raises (either `OSError` from ctypes or `RuntimeError` from
   our code).
2. `test_create_net_stack_sim_still_works` — existing behavior unchanged.

**Commit:** `feat(networkdet): wire DPDKBackend into create_net_stack factory`

**Do NOT:**
- Change the default backend from "sim". Existing code must work unchanged.
- Add DPDK-specific args to `NetStackConfig`. Pass them through kwargs.

---

### Task 11: Runner CLI integration

**What:** Make the runner's `--network-backend dpdk` flag work end-to-end.
Add `--dpdk-port` and `--dpdk-eal-args` flags.

**Files:**
- Modify `cmd/runner/main.py`

**Changes:**

Add CLI arguments (near line 548):

```python
parser.add_argument("--dpdk-port", type=int, default=0,
                    help="DPDK port ID for NIC transmission")
parser.add_argument("--dpdk-eal-args", default="",
                    help="DPDK EAL arguments (space-separated)")
```

Modify the network stack creation (around line 387) to pass DPDK args:

```python
backend_kwargs = {}
if network_backend == "dpdk":
    backend_kwargs["dpdk_port"] = args.dpdk_port
    backend_kwargs["dpdk_eal_args"] = args.dpdk_eal_args.split() if args.dpdk_eal_args else []

net = create_net_stack(
    manifest_dict, lockfile,
    backend=network_backend,
    **backend_kwargs,
)
```

After processing all responses, call `flush()` and capture the `TxReport`:

```python
# After the frame generation loop
tx_report = None
if net is not None:
    tx_report = net.flush()  # Returns None for sim backend
    frames = net.capture_frames_hex()
    # ... existing code ...
```

You need to add a `flush()` method to `DeterministicNetStack` that
delegates to the backend:

```python
# In DeterministicNetStack
def flush(self) -> TxReport | None:
    return self._backend.flush()
```

Add the `TxReport` data to the run bundle's `network_provenance`:

```python
"network_provenance": {
    # ... existing fields ...
    **({"egress_verification": {
        "backend": network_backend,
        "level": tx_report.level,
        "pre_enqueue_digest": tx_report.pre_enqueue_digest,
        "tx_completion_digest": tx_report.tx_completion_digest,
        "frames_submitted": tx_report.frames_submitted,
        "frames_confirmed": tx_report.frames_confirmed,
        "match": tx_report.match,
    }} if tx_report is not None else {}),
},
```

**Tests:** Add a unit test that runs the runner in synthetic mode with a
mocked DPDK backend. Assert the run bundle contains `egress_verification`.

**Commit:** `feat(runner): integrate DPDK backend with --network-backend dpdk`

**Do NOT:**
- Change the default backend. `--network-backend sim` remains the default.
- Remove the `legacy` backend option.
- Add a `--verify-loopback` flag yet. That is Task 12.

---

### Task 12: Extend `TxReport` and runner for loopback verification

**What:** Add loopback verification (Level 2) as an optional runner flag.

**Files:**
- Modify `pkg/networkdet/tx_report.py`
- Modify `cmd/runner/main.py`

**Changes to `TxReport`:**

Add optional loopback fields:

```python
@dataclass(frozen=True)
class TxReport:
    pre_enqueue_digest: str
    tx_completion_digest: str
    frames_submitted: int
    frames_confirmed: int
    rx_loopback_digest: str | None = None
    rx_loopback_count: int | None = None

    @property
    def match(self) -> bool:
        base = self.pre_enqueue_digest == self.tx_completion_digest
        if self.rx_loopback_digest is not None:
            return base and self.tx_completion_digest == self.rx_loopback_digest
        return base

    @property
    def level(self) -> str:
        return "loopback" if self.rx_loopback_digest else "tx_completion"
```

**Changes to runner:**

Add `--dpdk-loopback-port` flag:

```python
parser.add_argument("--dpdk-loopback-port", type=int, default=None,
                    help="DPDK RX port for loopback verification (Level 2)")
```

After `flush()`, if loopback port is specified:

```python
if tx_report is not None and args.dpdk_loopback_port is not None:
    # The DPDK backend was initialized with the TX port.
    # For loopback, we need to receive on the loopback port.
    # This requires a second backend instance or multi-port support.
    # For now, use recv_loopback on the same backend if the NIC
    # supports internal loopback, or on a separate backend.
    rx_frames, rx_digest = dpdk_backend.recv_loopback(timeout_ms=2000)
    tx_report = TxReport(
        pre_enqueue_digest=tx_report.pre_enqueue_digest,
        tx_completion_digest=tx_report.tx_completion_digest,
        frames_submitted=tx_report.frames_submitted,
        frames_confirmed=tx_report.frames_confirmed,
        rx_loopback_digest=rx_digest,
        rx_loopback_count=len(rx_frames),
    )
```

**Tests:**

1. Update `test_tx_report.py`:
   - `test_loopback_match_all_three_digests` — all three equal, match is True.
   - `test_loopback_mismatch_rx` — pre/tx match but rx differs, match is False.
   - `test_level_is_loopback_when_rx_present` — assert level == "loopback".
   - `test_level_is_tx_completion_when_rx_absent` — assert level == "tx_completion".

**Commit:** `feat(networkdet): add loopback verification (Level 2) to TxReport and runner`

**Do NOT:**
- Make loopback mandatory. It is always opt-in via the flag.
- Block indefinitely waiting for RX frames. Use a timeout.

---

### Task 13: Run bundle schema update

**What:** Extend `schemas/run_bundle.v1.schema.json` to accept the
optional `egress_verification` field inside `network_provenance`.

**Files:**
- Modify `schemas/run_bundle.v1.schema.json`

**Changes:**

Add to the `network_provenance` properties:

```json
"egress_verification": {
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "backend": {
      "enum": ["dpdk"],
      "type": "string"
    },
    "level": {
      "enum": ["tx_completion", "loopback"],
      "type": "string"
    },
    "pre_enqueue_digest": {
      "pattern": "^sha256:[a-f0-9]{64}$",
      "type": "string"
    },
    "tx_completion_digest": {
      "pattern": "^sha256:[a-f0-9]{64}$",
      "type": "string"
    },
    "rx_loopback_digest": {
      "pattern": "^sha256:[a-f0-9]{64}$",
      "type": "string"
    },
    "frames_submitted": {
      "minimum": 0,
      "type": "integer"
    },
    "frames_confirmed": {
      "minimum": 0,
      "type": "integer"
    },
    "rx_loopback_count": {
      "minimum": 0,
      "type": "integer"
    },
    "match": {
      "type": "boolean"
    }
  },
  "required": [
    "backend", "level",
    "pre_enqueue_digest", "tx_completion_digest",
    "frames_submitted", "frames_confirmed", "match"
  ]
}
```

The field is **not required** in `network_provenance.required` — it is
only present when a real backend is used.

Also update `capture_mode` and `route_mode` enums to accept the new
values:

```json
"capture_mode": {
  "enum": ["userspace_pre_enqueue"],
  "type": "string"
},
"route_mode": {
  "enum": ["deterministic_userspace_stack", "dpdk_kernel_bypass"],
  "type": "string"
}
```

**Tests:** Run the schema gate:

```bash
make schema
```

Also validate a sample bundle with the new field against the schema:

```bash
python -c "
import json, jsonschema
schema = json.load(open('schemas/run_bundle.v1.schema.json'))
# Use an existing bundle and add egress_verification
bundle = json.load(open('path/to/existing/run_bundle.v1.json'))
bundle['network_provenance']['egress_verification'] = {
    'backend': 'dpdk', 'level': 'tx_completion',
    'pre_enqueue_digest': 'sha256:' + 'a' * 64,
    'tx_completion_digest': 'sha256:' + 'a' * 64,
    'frames_submitted': 42, 'frames_confirmed': 42, 'match': True,
}
jsonschema.validate(bundle, schema)
print('OK')
"
```

**Commit:** `schema: add egress_verification to run_bundle.v1`

**Do NOT:**
- Make `egress_verification` required. It is optional.
- Change existing required fields. This is a backward-compatible extension.

---

### Task 14: Container image with DPDK

**What:** Create a Dockerfile (or extend the Nix closure) that includes
DPDK 24.11 LTS, mlx5 PMD, and `libnetdet.so`.

**Files:**
- Create `nix/packages/dpdk.nix` (if using Nix) or
  `deploy/dpdk/Dockerfile` (if using Docker)
- Modify `Makefile` to add a build target

**Approach — choose one based on the existing Nix setup:**

**Option A: Nix overlay (preferred if the Nix closure already builds):**

```nix
# nix/packages/dpdk.nix
{ pkgs }: pkgs.dpdk.overrideAttrs (old: {
  version = "24.11";
  # Pin to LTS for stability
  # Enable mlx5 PMD
  mesonFlags = old.mesonFlags ++ [
    "-Ddisable_drivers=net/mlx4"
    "-Denable_driver_sdk=true"
  ];
  buildInputs = old.buildInputs ++ [ pkgs.rdma-core ];
})
```

Add `libnetdet` as a Nix package:

```nix
# nix/packages/libnetdet.nix
{ pkgs, dpdk }: pkgs.stdenv.mkDerivation {
  pname = "libnetdet";
  version = "0.1.0";
  src = ../../native/libnetdet;
  buildInputs = [ dpdk pkgs.openssl pkgs.pkg-config pkgs.cmake ];
}
```

**Option B: Docker (if Nix is not used for DPDK yet):**

```dockerfile
FROM ubuntu:22.04 AS dpdk-builder
RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config meson ninja-build \
    python3 python3-pyelftools libnuma-dev libssl-dev \
    libibverbs-dev libmlx5-1 rdma-core
# Build DPDK 24.11 LTS
RUN wget https://fast.dpdk.org/rel/dpdk-24.11.tar.xz && \
    tar xf dpdk-24.11.tar.xz && \
    cd dpdk-24.11 && \
    meson setup build -Ddefault_library=shared && \
    cd build && ninja && ninja install && ldconfig
# Build libnetdet
COPY native/libnetdet /src/libnetdet
RUN cd /src/libnetdet && mkdir build && cd build && cmake .. && make
```

**Tests:** Build the container image and verify `libnetdet.so` loads:

```bash
docker run --rm <image> python3 -c \
  "import ctypes; ctypes.CDLL('/usr/local/lib/libnetdet.so')"
```

**Commit:** `build: add DPDK 24.11 + libnetdet to container image`

**Do NOT:**
- Include DPDK debug symbols in the production image (use `-Dbuildtype=release`).
- Enable drivers we don't need. Only enable `net/mlx5` and `net/ring` (for
  testing).

---

### Task 15: E2E test on GH200

**What:** Run the full pipeline on a Lambda Cloud GH200 instance with
DPDK backend and verify egress integrity.

**Files:**
- Create `tests/integration/test_dpdk_egress.py`
- Create `deploy/dpdk/run_egress_test.sh`

**Test script (`deploy/dpdk/run_egress_test.sh`):**

```bash
#!/usr/bin/env bash
set -euo pipefail

# Prereqs: DPDK-enabled container running on GH200
# with hugepages allocated and NIC bound

# 1. Allocate hugepages
echo 1024 > /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
mkdir -p /dev/hugepages

# 2. Run the runner with DPDK backend
python3 cmd/runner/main.py \
  --manifest manifests/qwen3-1.7b.manifest.json \
  --lockfile /tmp/built.json \
  --out-dir /tmp/run \
  --mode synthetic \
  --network-backend dpdk \
  --dpdk-port 0 \
  --dpdk-eal-args "--socket-mem 256 -l 0"

# 3. Check the run bundle
python3 -c "
import json, sys
bundle = json.load(open('/tmp/run/run_bundle.v1.json'))
ev = bundle['network_provenance'].get('egress_verification')
if ev is None:
    print('FAIL: no egress_verification in bundle')
    sys.exit(1)
if not ev['match']:
    print(f'FAIL: digest mismatch')
    print(f'  pre_enqueue:    {ev[\"pre_enqueue_digest\"]}')
    print(f'  tx_completion:  {ev[\"tx_completion_digest\"]}')
    sys.exit(1)
print(f'PASS: {ev[\"frames_confirmed\"]}/{ev[\"frames_submitted\"]} frames verified')
print(f'  level: {ev[\"level\"]}')
print(f'  digest: {ev[\"tx_completion_digest\"]}')
"
```

**Python integration test** (`tests/integration/test_dpdk_egress.py`):

```python
"""Integration test for DPDK egress integrity.

Requires DPDK hardware — skipped in CI unless DPDK_TEST=1 is set.
"""
import json
import os
import subprocess
import unittest
from pathlib import Path


DPDK_TEST = os.environ.get("DPDK_TEST") == "1"


@unittest.skipUnless(DPDK_TEST, "requires DPDK hardware (set DPDK_TEST=1)")
class TestDPDKEgress(unittest.TestCase):

    def test_tx_completion_digest_matches(self):
        """Level 1: TX completion digest matches pre-enqueue digest."""
        # Run the runner with DPDK backend
        result = subprocess.run([
            "python3", "cmd/runner/main.py",
            "--manifest", "manifests/qwen3-1.7b.manifest.json",
            "--lockfile", "/tmp/built.json",
            "--out-dir", "/tmp/test_dpdk_egress",
            "--mode", "synthetic",
            "--network-backend", "dpdk",
            "--dpdk-port", "0",
            "--dpdk-eal-args", "--socket-mem 256 -l 0",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

        bundle = json.loads(
            (Path("/tmp/test_dpdk_egress") / "run_bundle.v1.json").read_text()
        )
        ev = bundle["network_provenance"]["egress_verification"]
        self.assertTrue(ev["match"])
        self.assertEqual(ev["frames_submitted"], ev["frames_confirmed"])
        self.assertEqual(ev["pre_enqueue_digest"], ev["tx_completion_digest"])
```

**Commit:** `test: add DPDK egress integrity E2E test`

**Do NOT:**
- Run this test in the CI gate. It requires hardware. Gate it behind
  `DPDK_TEST=1`.
- Assume hugepages are already allocated. The test script must set them up.

---

### Task 16: Update ADR-0004

**What:** Update the ADR to reflect that Phase 4 is implemented.

**Files:**
- Modify `docs/adr/0004-deterministic-userspace-networking.md`

**Changes:**
- Change status from "Proposed" to "Accepted"
- Update the phased rollout section to mark Phase 4 as complete
- Add a "Phase 4 Implementation Notes" section documenting:
  - NIC hardware identified (from Task 0)
  - `libnetdet.so` API surface
  - Verification levels (TX completion, loopback)
  - Warden assertion model
  - Known limitations

**Commit:** `docs: update ADR-0004 with Phase 4 implementation notes`

---

## 6. Task Dependency Graph

```
Task 0 (recon) ─────────────────────────────────────────┐
                                                         │
Task 1 (TxReport) ──┬── Task 2 (flush interface) ──┐    │
                     │                               │    │
                     │   Task 3 (warden assertion) ──┤    │
                     │                               │    │
                     └── Task 12 (loopback TxReport) │    │
                                                     │    │
Task 4 (C scaffolding) ── Task 5 (EAL init) ──┐     │    │
                                                │     │    │
                           Task 6 (TX path) ────┤     │    │
                                                │     │    │
                           Task 7 (RX path) ────┤     │    │
                                                │     │    │
                                   Task 8 (FFI) ┤     │    │
                                                │     │    │
                                   Task 9 (backend)───┤    │
                                                │     │    │
                                   Task 10 (factory)──┤    │
                                                      │    │
                                   Task 11 (runner) ──┤    │
                                                      │    │
                                   Task 13 (schema) ──┤    │
                                                      │    │
                                   Task 14 (container)─┤   │
                                                       │   │
                                   Task 15 (E2E) ──────┘   │
                                                            │
                                   Task 16 (ADR) ──────────┘
```

**Parallelizable work:**
- Tasks 1-3 (Python side) can proceed in parallel with Tasks 4-7 (C side)
- Task 0 (recon) can happen anytime but must be done before Task 15

---

## 7. Pitfalls

### 7.1 Offloads that silently mutate frames

The NIC may have default offloads enabled that you did not request. After
port setup, **read back the actual port config** and assert that no TX
offloads are active. Log the full `rte_eth_dev_info.tx_offload_capa` so
you can see what the NIC supports vs what you enabled.

### 7.2 mlx5 bifurcated mode gotchas

With mlx5, you do NOT bind the NIC to vfio-pci. The kernel driver
(`mlx5_core`) stays loaded. DPDK creates a secondary device path via
libibverbs. This means:
- You do NOT lose SSH access.
- You do NOT need to modprobe vfio-pci.
- You DO need `libibverbs-dev` and `rdma-core` installed.
- You DO need the same hugepages setup as any DPDK port.

### 7.3 Hugepage allocation failures

Hugepages must be allocated **before** EAL init. If the system is under
memory pressure, allocation may silently succeed with fewer pages than
requested. Check `/proc/meminfo` after allocation.

### 7.4 Frame size matters

The minimum Ethernet frame is 60 bytes (excluding FCS). DPDK's
`rte_eth_tx_burst` may pad shorter frames. Our frame builder already
pads to 60 bytes (see `ethernet.py` `MIN_ETHERNET_PAYLOAD`), so this
should not be an issue — but verify by comparing the TX completion
digest to the pre-enqueue digest. If they differ, check for padding.

### 7.5 mbuf lifecycle

After `rte_eth_tx_burst`, sent mbufs are owned by the driver. Do not
read or free them. Only free **unsent** mbufs (those beyond the return
value of `tx_burst`).

### 7.6 The warden assertion will fire if...

- The frame builder sets a nonzero reserved bit
- The frame builder sets a nonzero urgent pointer
- The frame builder includes TCP options beyond MSS in non-SYN packets
- The frame builder's checksum is wrong (the warden recomputes and
  compares)

All of these would be bugs in `pkg/networkdet/frame.py` or the
underlying layer modules. Fix them there, not in the warden.

### 7.7 Do not cargo-cult the server's warden usage

`cmd/server/main.py` lines 728-731 use the warden as a real normalizer
on frames from the kernel TCP stack. That is the correct use in that
context. In the DPDK path, the warden is an assertion, not a normalizer.
Do not copy the server's pattern into the runner.

### 7.8 Schema backward compatibility

The `egress_verification` field is optional. Existing run bundles without
it must still validate. Do not add it to `network_provenance.required`.

### 7.9 ctypes and GC

When passing frame data to C via ctypes, Python may garbage-collect the
underlying buffers before C reads them. Keep a reference to every buffer
you create until after `netdet_send()` returns. The FFI module in Task 8
handles this with the `buffers` list.
