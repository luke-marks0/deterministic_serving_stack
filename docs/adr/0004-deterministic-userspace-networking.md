# ADR-0004: Deterministic Userspace Networking Architecture

- Status: Proposed
- Date: 2026-03-17
- Owners: Platform
- Reviewers: Inference, Security
- Supersedes: None
- Superseded by: None

## Context

Spec section 9 (SPEC-9.1 through SPEC-9.4) requires bitwise-reproducible L2 egress frames through a deterministic userspace networking stack. The kernel TCP/IP stack introduces nondeterminism through random ISNs, timestamps, IP ID generation, ephemeral port selection, congestion window dynamics, and timer-driven retransmissions. Currently, `pkg/networkdet/` is an empty scaffold and all network observables are synthetic (hashed request content, not real protocol bytes).

No existing off-the-shelf tool (pf scrub, Snort normalize, Suricata IPS) provides deterministic networking. These tools apply heuristic normalization, not bitwise reproducibility. The paper "Eliminating Steganography in Internet Traffic with Active Wardens" (Fisk et al.) introduces the concept of Minimal Requisite Fidelity (MRF) — reducing every protocol field to the minimum entropy required by its semantics. We apply this principle to achieve deterministic frame construction.

## Decision

Build a custom minimal deterministic TCP/IP stack on DPDK with a simulated backend for testing without hardware. Apply the MRF principle: every protocol field is pinned to the minimum entropy needed for correct operation.

Architecture:

1. **Pure-Python protocol stack** (`pkg/networkdet/`): deterministic Ethernet, IPv4, and TCP frame construction with software checksums, fixed field values, and a deterministic TCP state machine.

2. **Backend abstraction**: `sim` backend (pure Python, in-memory loopback) for CI/testing, `dpdk` backend (C library via ctypes) for production with real NICs.

3. **Capture ring**: pre-enqueue mirror buffer satisfying SPEC-9.3 (capture without perturbation).

4. **MRF field policy**:
   - Ethernet: fixed src/dst MAC, ethertype 0x0800, no VLAN tags
   - IPv4: version=4, IHL=5, DSCP/ECN=0, ID=deterministic counter, DF=1, TTL=64, no options, software checksum
   - TCP: fixed ports, ISN=sha256(run_id+conn_index), no timestamps, no SACK, no window scaling, no Nagle, urgent_ptr=0, reserved bits=0, software checksum, MSS option only in SYN

5. **DPDK + mlx5 PMD** for production: full L2 control, ConnectX-7 support, pinned in Nix closure.

## Consequences

- `pkg/networkdet/` becomes the networking implementation package with ~10 modules.
- DPDK 24.11 LTS + mlx5 PMD added to Nix runtime closure (Phase 4).
- `libnetdet.so` native C library for DPDK interaction (Phase 4).
- Sim backend enables full frame-level determinism testing without hardware.
- Network egress golden fixtures must be regenerated (one-time breaking change).
- Runner and capture commands produce real L2 frame bytes instead of synthetic hex.

## Alternatives Considered

1. **F-Stack (FreeBSD TCP on DPDK)**: Rejected. FreeBSD's TCP stack has inherited nondeterminism (ISN generation via arc4random, congestion heuristics, timer-driven retransmissions, SACK/timestamp negotiation). Patching these out would require maintaining a fork of the FreeBSD network stack.

2. **AF_XDP (XDP sockets)**: Rejected. Still routes through kernel data structures (socket buffers, congestion state) that introduce nondeterminism. Does not provide full L2 frame control.

3. **Raw sockets**: Rejected. Requires root privileges, still subject to kernel IP ID assignment and ARP cache nondeterminism. No control over Ethernet padding.

4. **Existing normalizers (pf scrub, Snort, Suricata)**: Rejected. These are heuristic, not deterministic — they normalize known problematic fields but do not guarantee bitwise reproducibility.

## Rollout and Rollback Plan

Phased rollout:

1. Phase 2: Pure-Python stack with sim backend. No production impact. Runner gains `--network-backend` flag with `legacy` default preserving current behavior.
2. Phase 3: Pipeline integration. Runner/capture switch to real frames. `--network-backend legacy` preserves old behavior for one release cycle.
3. Phase 4: DPDK backend for production NIC use. Requires NIC hardware.
4. Phase 5: Conformance hardening and TLS modes.

Rollback: revert to `--network-backend legacy` at any phase.

## Conformance Impact

- Conformance IDs:
  - SPEC-9.1-NETWORK-USERSPACE-ROUTING: scaffolding -> implemented (Phase 3)
  - SPEC-9.1-NETWORK-EGRESS: scaffolding -> implemented (Phase 3)
  - SPEC-9.2-OFFLOADS-PINNED: scaffolding -> implemented (Phase 2)
  - SPEC-9.3-CAPTURE-NONPERTURBING: scaffolding -> implemented (Phase 2)
  - SPEC-9.4-SECURITY-MODE: already implemented (plaintext mode)
- CI gates affected:
  - PR gate: new unit tests for `pkg/networkdet/`
  - Main gate: D5 test updated to use real frames
  - Nightly gate: extended network determinism tests
  - Release gate: D5 with real frame validation
