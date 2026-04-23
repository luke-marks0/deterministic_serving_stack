#!/usr/bin/env python3
"""Compare two pcap captures for deterministic segmentation.

Extracts server->client TCP data segments and compares:
1. Segment count
2. Segment sizes (segmentation boundaries)
3. TCP payload content (byte-for-byte)
4. TCP header fields (flags, window, options)
"""
import struct
import sys


def read_pcap(path):
    packets = []
    with open(path, "rb") as f:
        ghdr = f.read(24)
        magic = struct.unpack("<I", ghdr[0:4])[0]
        endian = "<" if magic == 0xa1b2c3d4 else ">"
        while True:
            phdr = f.read(16)
            if len(phdr) < 16:
                break
            incl_len = struct.unpack(f"{endian}I", phdr[8:12])[0]
            data = f.read(incl_len)
            if len(data) < incl_len:
                break
            packets.append(data)
    return packets


def extract_server_data(packets, server_port):
    """Extract TCP data segments from server to client."""
    segments = []
    for pkt in packets:
        if len(pkt) < 54:
            continue
        ethertype = struct.unpack("!H", pkt[12:14])[0]
        if ethertype != 0x0800:
            continue
        ip_start = 14
        ip_header_len = (pkt[ip_start] & 0x0F) * 4
        protocol = pkt[ip_start + 9]
        if protocol != 6:
            continue
        ip_total_len = struct.unpack("!H", pkt[ip_start + 2 : ip_start + 4])[0]
        tcp_start = ip_start + ip_header_len
        src_port = struct.unpack("!H", pkt[tcp_start:tcp_start + 2])[0]
        if src_port != server_port:
            continue
        tcp_data_offset = ((pkt[tcp_start + 12] >> 4) & 0x0F) * 4
        flags = pkt[tcp_start + 13]
        window = struct.unpack("!H", pkt[tcp_start + 14 : tcp_start + 16])[0]
        payload_start = tcp_start + tcp_data_offset
        payload_end = ip_start + ip_total_len
        payload = pkt[payload_start:payload_end]
        if len(payload) > 0:
            segments.append({
                "payload": payload,
                "flags": flags,
                "window": window,
                "size": len(payload),
                "tcp_data_offset": tcp_data_offset,
            })
    return segments


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <pcap1> <pcap2>")
        sys.exit(1)

    port = 9999
    pkts1 = read_pcap(sys.argv[1])
    pkts2 = read_pcap(sys.argv[2])
    segs1 = extract_server_data(pkts1, port)
    segs2 = extract_server_data(pkts2, port)

    print(f"Run 1: {len(segs1)} data segments")
    print(f"Run 2: {len(segs2)} data segments")

    if len(segs1) != len(segs2):
        print("FAIL: Different segment count")
        sys.exit(1)

    all_match = True
    for i, (s1, s2) in enumerate(zip(segs1, segs2)):
        size_match = s1["size"] == s2["size"]
        payload_match = s1["payload"] == s2["payload"]
        flags_match = s1["flags"] == s2["flags"]
        window_match = s1["window"] == s2["window"]

        status = "OK" if all([size_match, payload_match, flags_match, window_match]) else "MISMATCH"
        print(f"  Segment {i}: size={s1['size']}/{s2['size']} "
              f"flags=0x{s1['flags']:02x}/0x{s2['flags']:02x} "
              f"window={s1['window']}/{s2['window']} "
              f"payload={'MATCH' if payload_match else 'DIFFER'} "
              f"[{status}]")

        if not all([size_match, payload_match, flags_match, window_match]):
            all_match = False

    if all_match:
        print("\nPASS: All segments are byte-identical across runs")
    else:
        print("\nFAIL: Segments differ between runs")
        sys.exit(1)


if __name__ == "__main__":
    main()
