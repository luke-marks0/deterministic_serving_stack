# Memo: Network Determinism — How It Works and What It Doesn't Cover

## How the network output in the run bundle is generated

The run bundle includes a `network_egress` observable — a list of deterministic L2 Ethernet frames derived from the inference output. These frames are **not captured from real network traffic**. They are constructed by a simulated userspace network stack (`pkg/networkdet/`) that takes the application-layer response bytes and produces the exact frames that *would* be sent over the wire under controlled conditions.

The pipeline:

1. **Inference produces tokens + logits** (via vLLM or the synthetic runner)
2. **Response bytes are serialized** as canonical JSON: `{"id": "req-1", "tokens": [...], "logits": [...]}`
3. **The sim TCP stack** (`pkg/networkdet/tcp.py`) segments the bytes into MSS-sized chunks (default 1460 bytes), builds TCP headers with deterministic fields (sequence numbers, flags, checksums computed in software)
4. **The sim IP layer** (`pkg/networkdet/ip.py`) wraps each TCP segment in an IPv4 header with fixed fields (TTL=64, DF=1, no fragmentation, deterministic IP ID counter, software checksum)
5. **The sim Ethernet layer** (`pkg/networkdet/ethernet.py`) wraps each IP packet in an Ethernet frame with fixed MAC addresses and EtherType
6. **The capture ring** (`pkg/networkdet/capture.py`) records every frame and computes a SHA-256 digest over the ordered concatenation
7. **The frames are written** to `observables/network_egress.json` in the run bundle, and the digest goes into `network_provenance.capture_digest`

Every field in every header is either fixed by policy (TTL, flags, MAC addresses) or deterministically derived from the data (checksums, sequence numbers, IP ID). There are no random or timing-dependent values. Same input bytes → same frames → same digest, every time.

### Code locations

| Component | File | What it does |
|-----------|------|-------------|
| Entry point | `cmd/runner/main.py:388-405` | Creates net stack, feeds response bytes, collects frames |
| Stack factory | `pkg/networkdet/__init__.py:120` | `create_net_stack()` — builds the stack from config |
| Frame builder | `pkg/networkdet/frame.py:71` | Segments data → wraps in TCP/IP/Ethernet |
| TCP state machine | `pkg/networkdet/tcp.py:185` | `segment_data()` — splits payload at MSS boundaries |
| IP packet builder | `pkg/networkdet/ip.py:42` | `build_packet()` — fixed header, software checksum |
| Ethernet framing | `pkg/networkdet/ethernet.py` | Fixed MAC addresses, EtherType 0x0800 |
| Capture + digest | `pkg/networkdet/capture.py:39` | SHA-256 over ordered frame concatenation |
| MRF policy | `pkg/networkdet/frame.py` docstring | Documents which fields are fixed and why |

## How it's realistic

The frames are structurally valid. You could hand them to Wireshark and it would parse them correctly — valid Ethernet headers, valid IP headers with correct checksums, valid TCP headers with correct sequence numbers and checksums. The TCP state machine does a proper 3-way handshake (SYN → SYN-ACK → ACK) before sending data segments, and closes with FIN.

The sim stack implements the same layering as a real kernel TCP/IP stack:
- Application data → TCP segmentation at MSS boundaries → IP encapsulation → Ethernet framing
- Checksums are computed in software (no offload), which is what a production deterministic deployment would do (all offloads disabled)

## How it differs from production

| Aspect | Sim stack | Real deployment |
|--------|-----------|-----------------|
| **Who segments TCP** | Sim stack, at fixed MSS boundaries | Kernel TCP stack, influenced by timing, congestion window, Nagle's algorithm |
| **Who computes checksums** | Sim stack, in software | Either NIC (if offload enabled) or kernel |
| **Network timing** | None — frames are generated synchronously | ACKs, retransmits, window scaling all affect segment boundaries |
| **IP fragmentation** | Never — DF=1, MSS enforced | Could happen if MTU changes mid-path |
| **Actual transmission** | Frames are never sent over a wire | Frames go through the NIC, switch, cable |
| **Packet ordering** | Always in-order (sequential generation) | Could be reordered by the network |
| **Receiver influence** | None — no actual receiver | Receiver's window size affects sender's segmentation |

The key difference: **in production, the kernel's TCP stack decides segment boundaries based on runtime conditions, not just MSS.** Two identical servers sending identical data can produce different segment sizes due to CPU scheduling jitter, socket buffer fill levels, and congestion control state. The sim stack eliminates all of this by constructing frames directly from the data.

## The active warden

The active warden (`pkg/networkdet/warden.py`) is a separate component that sits on the **real** network path. It intercepts live packets via Linux NFQUEUE (netfilter) and normalizes their headers before forwarding:

- Forces TTL to a fixed value
- Zeros DSCP/ECN bits
- Forces DF=1, MF=0, fragment_offset=0 (prevents IP fragmentation)
- Strips IP options
- Normalizes TCP options (strips timestamps, keeps only MSS)
- Replaces IP ID with an encrypted deterministic permutation
- Recomputes checksums in software

The warden is a packet-level normalizer — it processes each packet independently without buffering or reassembly.

### Code location

| Component | File |
|-----------|------|
| Warden core | `pkg/networkdet/warden.py:140-380` |
| NFQUEUE service | `pkg/networkdet/warden_service.py` |
| Config | `pkg/networkdet/warden_config.py` |
| Tests | `tests/unit/test_networkdet_warden.py` (32 tests) |

## Fragmentation-based steganography

### What it is

A covert channel where the attacker encodes hidden information in how data is split into packets, rather than in the data itself. Two methods:

**IP fragmentation steganography:** Split a 1400-byte payload into fragments of 700+700 vs 800+600. The split point encodes a bit. Variations: fragment ordering, duplicate fragment IDs, overlapping fragments.

**TCP segmentation steganography:** Send "Hello World" as segments of 3+8 bytes vs 5+6 bytes. The segment boundaries encode bits. This is harder to detect because segment size variation is normal in TCP.

### Does the sim stack prevent it?

**Yes.** The sim stack constructs frames from application data with fixed MSS segmentation. There is no way for the inference code to influence segment boundaries — the same data always produces the same segments. An attacker would need to modify the sim stack itself to inject a covert channel.

### Does the warden prevent it?

**Partially.**

**IP fragmentation:** The warden forces DF=1, MF=0, fragment_offset=0 on every packet (line 186-188). This breaks any IP fragmentation — fragments are not reassembled, they're corrupted. This prevents the covert channel but also breaks legitimate fragmented traffic. The warden does not log fragmentation attempts.

**TCP segmentation:** The warden **does not** reassemble TCP streams or re-segment them. It normalizes individual packet headers (flags, options, checksums) but leaves payload boundaries untouched. A malicious sender that bypasses the sim stack and sends data through the real kernel TCP stack could encode information in segment sizes, and the warden would not detect or prevent it.

### What would be needed for full coverage

Defending against TCP segmentation steganography requires a **TCP-aware normalizer** (essentially a transparent TCP proxy):

1. Buffer incoming TCP segments
2. Reassemble the byte stream
3. Re-segment at fixed MSS boundaries
4. Forward the re-segmented packets

This is significantly more complex than the current warden, which operates on individual packets without buffering. It would add latency (must buffer at least MSS bytes before forwarding) and would need to handle connection state (SYN/FIN/RST), window management, and retransmits.

The sim stack sidesteps this entirely by never going through the kernel TCP stack in the first place. For the current demo, this is sufficient — the run bundle's `network_egress` proves what the frames *would* look like, and the determinism guarantee holds at the application layer (157/157 cross-server token matches across 3 models and 8.88 million tokens).
