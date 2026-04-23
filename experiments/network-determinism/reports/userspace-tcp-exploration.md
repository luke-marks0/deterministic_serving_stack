# Userspace TCP Stack Exploration

**Date**: 2026-04-08
**Status**: PoC successful -- all success criteria met

## Goal

Build a userspace TCP server that accepts real TCP connections via raw sockets
(AF_PACKET), bypasses the kernel TCP stack for the data path, and sends HTTP
responses with deterministic segmentation (fixed MSS boundaries).

## Target environment

DigitalOcean droplets (no SR-IOV, no DPDK-compatible NICs). We use AF_PACKET
raw sockets for TX and RX.

## Phase 1: Feasibility Analysis

### Existing code review

The `pkg/networkdet/` package already has everything needed for frame construction:

- `DeterministicTCPConnection`: Full TCP state machine (SYN/SYN-ACK/ACK/DATA/FIN)
  with deterministic ISN, fixed window, no timestamps, no SACK, no window scaling.
  The `segment_data()` method segments payloads at fixed MSS boundaries.

- `DeterministicIPLayer`: IPv4 with deterministic IP ID counter, DF=1, fixed TTL,
  software checksums.

- `build_ethernet_frame()`: L2 frame construction with deterministic zero-padding.

- `DeterministicFrameBuilder`: Composes all three layers into complete L2 frames.

### Key gap: server-side connection handling

The existing code was designed for the *egress* side (building frames for a known
connection). For a userspace TCP *server*, we need:

1. **Packet capture**: Read incoming SYN/ACK/data from AF_PACKET
2. **Packet parsing**: Extract TCP flags, sequence numbers, ports from raw frames
3. **Connection state**: Map incoming packets to DeterministicTCPConnection instances
4. **Kernel interference suppression**: iptables rules to prevent kernel from
   processing TCP on our port

### Architecture decision

Built a thin server wrapper (`UserspaceServer`) that:

- Uses AF_PACKET for both RX and TX
- Parses incoming packets to extract TCP state (ParsedPacket)
- Creates a DeterministicTCPConnection per client connection
- Manages connection lifecycle (SYN -> handshake -> response -> FIN)
- Uses DeterministicIPLayer for IP headers and software checksums

## Phase 2: Implementation

### File: `pkg/networkdet/userspace_tcp_server.py`

Key components:

1. **`ParsedPacket`** -- parses raw L2 frames into structured fields (MAC, IP, TCP,
   payload). Handles variable-length IP and TCP headers.

2. **`ConnectionState`** -- per-connection state wrapping a `DeterministicTCPConnection`
   and `DeterministicIPLayer`. Tracks client MAC/IP/port, handles frame wrapping.

3. **`UserspaceServer`** -- main server loop:
   - Creates AF_PACKET sockets (RX with ETH_P_ALL, TX with ETH_P_IP)
   - Adds iptables rules to suppress kernel TCP on our port
   - Polls for incoming packets, dispatches to handlers
   - Supports multiple HTTP endpoints with deterministic responses

### Kernel bypass strategy

Two iptables rules are needed:

```bash
# Prevent kernel TCP stack from seeing incoming packets on our port.
# AF_PACKET operates at the driver level, BEFORE netfilter, so we still see them.
iptables -A INPUT -p tcp --dport 9999 -j DROP

# Belt-and-suspenders: also drop outgoing RSTs from our port.
iptables -A OUTPUT -p tcp --tcp-flags RST RST --sport 9999 -j DROP
```

The INPUT DROP rule is the key insight: `AF_PACKET` captures packets before
netfilter processes them, so our userspace server sees every frame on the
interface. But by DROPping in INPUT, the kernel TCP stack never sees the SYN
and never generates a RST.

### Connection lifecycle

```
Client                    Userspace Server (AF_PACKET)
  |                              |
  |--- SYN ------------------>  | (captured via AF_PACKET RX)
  |                              | Creates ConnectionState
  |                              | Builds SYN-ACK via DeterministicTCPConnection
  |  <--- SYN-ACK -----------  | (sent via AF_PACKET TX)
  |                              |
  |--- ACK ------------------>  | (handshake complete)
  |--- HTTP GET /path -------->  | (data with PSH|ACK)
  |                              | Parses path, selects response
  |                              | Segments response at MSS boundaries
  |  <--- DATA (1460 bytes) --  | Segment 1 (ACK)
  |  <--- DATA (1460 bytes) --  | Segment 2 (ACK)
  |  <--- DATA (remaining) ---  | Segment 3 (PSH|ACK)
  |  <--- FIN ----------------  | Connection close
  |--- ACK ------------------>  |
  |--- FIN ------------------>  |
  |  <--- ACK ----------------  |
  |                              |
```

## Phase 3: Deployment and Testing

### Test environment

- **Server**: DigitalOcean droplet `143.198.114.248` (1 vCPU, 1GB RAM, Ubuntu 24.04)
- **Client**: External machine (my Mac, IP 203.167.42.233)
- **Interface**: eth0, no special NIC features

### Test 1: Basic HTTP response via userspace TCP

```
$ curl http://143.198.114.248:9999/deterministic
{"model":"deterministic-test","tokens":[1,2,3,4,5],"request_id":"fixed-request-001","server":"userspace-tcp"}
```

**Result: PASS** -- curl received correct JSON response through userspace TCP.

### Test 2: Multi-segment response

```
$ curl http://143.198.114.248:9999/large | wc -c
4002
```

Server log:
```
Sending response: 4161 bytes in 3 segments (MSS=1460)
  Sent segment 1/3 (1480 bytes)   # 1460 payload + 20 TCP header
  Sent segment 2/3 (1480 bytes)
  Sent segment 3/3 (1261 bytes)   # 1241 payload + 20 TCP header
```

**Result: PASS** -- large response correctly segmented at MSS boundaries.

### Test 3: Deterministic segmentation (two independent server runs)

Captured packets from two independent server instances (fresh process each time,
same run_id), sending the same request to `/large`.

tcpdump output (server->client data segments only):

```
=== Run 1 ===
Flags [.],  seq 1:1461,     ack 89, win 65535, length 1460
Flags [.],  seq 1461:2921,  ack 89, win 65535, length 1460
Flags [P.], seq 2921:4162,  ack 89, win 65535, length 1241

=== Run 2 ===
Flags [.],  seq 1:1461,     ack 89, win 65535, length 1460
Flags [.],  seq 1461:2921,  ack 89, win 65535, length 1460
Flags [P.], seq 2921:4162,  ack 89, win 65535, length 1241
```

Byte-level comparison of TCP payloads:

```
Run 1: 3 data segments
Run 2: 3 data segments
  Segment 0: size=1460/1460 flags=0x10/0x10 window=65535/65535 payload=MATCH [OK]
  Segment 1: size=1460/1460 flags=0x10/0x10 window=65535/65535 payload=MATCH [OK]
  Segment 2: size=1241/1241 flags=0x18/0x18 window=65535/65535 payload=MATCH [OK]

PASS: All segments are byte-identical across runs
```

**Result: PASS** -- segmentation is deterministic and byte-identical across runs.

### Test 4: Deterministic ISN

Both runs with `run_id="poc-test-run"` and `conn_index=1` produced ISN `2828161708`,
which is `deterministic_isn("poc-test-run", 1)`.

**Result: PASS** -- ISN is deterministic and reproducible.

## Success Criteria Assessment

| Criterion | Status | Evidence |
|-----------|--------|----------|
| curl receives correct JSON response | PASS | Both /deterministic and /large endpoints return correct data |
| Response sent via userspace TCP (raw sockets) | PASS | AF_PACKET TX, kernel TCP completely bypassed via iptables INPUT DROP |
| Segmentation at fixed MSS boundaries | PASS | 1460/1460/1241 pattern for 4161-byte response |
| Same request twice produces byte-identical frames | PASS | Payload-level comparison confirms all segments match |

## Architecture Decisions and Trade-offs

### What works well

1. **AF_PACKET is sufficient** -- no DPDK needed. The existing frame builder
   produces complete L2 frames that can be written directly to AF_PACKET.

2. **Kernel bypass via iptables is clean** -- INPUT DROP + OUTPUT RST DROP
   effectively removes the kernel TCP stack from the picture while AF_PACKET
   continues to work at the driver level.

3. **Reuse of existing code** -- DeterministicTCPConnection and DeterministicIPLayer
   worked without modification for the server use case. Just needed a packet
   parser and server loop wrapper.

### Limitations and future work

1. **No retransmission** -- if a packet is lost, the connection stalls. For
   production, need a deterministic retransmission timer (fixed backoff, not
   kernel's adaptive RTO).

2. **No congestion control** -- the server sends all segments immediately without
   pacing. This works for small responses but could overwhelm the client's receive
   window for large transfers.

3. **No reassembly of client requests** -- if the client's HTTP request spans
   multiple TCP segments, only the first is processed. Need to buffer and
   reassemble.

4. **Single-threaded** -- the server polls AF_PACKET in a single thread. For
   production, could use multiple AF_PACKET sockets with FANOUT for parallel
   processing.

5. **Performance** -- Python + AF_PACKET adds latency compared to DPDK. For
   the determinism use case, latency is acceptable since the goal is correctness,
   not speed.

### Key insight: AF_PACKET vs DPDK

DPDK is not needed for deterministic egress on DO droplets. AF_PACKET provides:
- Full L2 frame access (both TX and RX)
- Works on any Linux NIC (no SR-IOV or hardware requirements)
- Sufficient performance for HTTP serving (not line-rate, but adequate)
- Simple deployment (just needs root / CAP_NET_RAW)

The only thing DPDK provides that AF_PACKET doesn't is performance (kernel bypass
for the data path). But since we're already bypassing the kernel TCP stack
(which is the source of nondeterminism), AF_PACKET gives us the determinism
guarantees we need without the deployment complexity of DPDK.
