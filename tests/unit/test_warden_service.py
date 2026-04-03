"""Unit tests for the inline warden NFQUEUE service wrapper.

These tests mock the netfilterqueue dependency so they run anywhere
(no root, no iptables, no Linux kernel needed).
"""
from __future__ import annotations

import os
import struct
import tempfile
import unittest
from unittest.mock import patch

from pkg.networkdet.checksums import ip_checksum, tcp_checksum
from pkg.networkdet.warden import ETH_HEADER_LEN
from pkg.networkdet.warden_config import WardenConfig, load_config
from pkg.networkdet.warden_service import FAKE_ETH, WardenService


# ---------------------------------------------------------------------------
# Helpers -- build raw IP+TCP packets (no Ethernet header, as NFQUEUE delivers L3)
# ---------------------------------------------------------------------------

def _build_ip_tcp_packet(
    *,
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    ip_id: int = 0x1234,
    ttl: int = 128,
    tos: int = 0,
    src_port: int = 12345,
    dst_port: int = 80,
    seq: int = 1000,
    ack: int = 0,
    tcp_flags: int = 0x02,  # SYN
    urgent_ptr: int = 0,
    tcp_options: bytes = b"",
    payload: bytes = b"",
) -> bytes:
    """Build a raw IP+TCP packet (L3, no Ethernet)."""
    import socket

    src_b = socket.inet_aton(src)
    dst_b = socket.inet_aton(dst)

    # Build TCP header.
    tcp_data_offset = (20 + len(tcp_options)) // 4
    tcp_hdr_no_cksum = struct.pack(
        "!HHIIBBHHH",
        src_port, dst_port, seq, ack,
        (tcp_data_offset << 4), tcp_flags, 65535, 0, urgent_ptr,
    ) + tcp_options + payload

    tcp_cksum = tcp_checksum(src_b, dst_b, tcp_hdr_no_cksum)
    tcp_segment = struct.pack(
        "!HHIIBBHHH",
        src_port, dst_port, seq, ack,
        (tcp_data_offset << 4), tcp_flags, 65535, tcp_cksum, urgent_ptr,
    ) + tcp_options + payload

    # Build IP header.
    total_length = 20 + len(tcp_segment)
    ip_hdr_no_cksum = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, tos, total_length, ip_id, 0x4000, ttl, 6, 0, src_b, dst_b,
    )
    ip_cksum = ip_checksum(ip_hdr_no_cksum)
    ip_hdr = struct.pack(
        "!BBHHHBBH4s4s",
        0x45, tos, total_length, ip_id, 0x4000, ttl, 6, ip_cksum, src_b, dst_b,
    )

    return ip_hdr + tcp_segment


def _parse_ip_from_raw(data: bytes) -> dict:
    """Parse IP fields from a raw L3 packet (no Ethernet header)."""
    return {
        "tos": data[1],
        "total_length": struct.unpack("!H", data[2:4])[0],
        "ip_id": struct.unpack("!H", data[4:6])[0],
        "ttl": data[8],
    }


def _parse_tcp_from_raw(data: bytes) -> dict:
    """Parse TCP fields from a raw L3 packet (no Ethernet header)."""
    tcp_start = (data[0] & 0x0F) * 4
    return {
        "seq": struct.unpack("!I", data[tcp_start + 4 : tcp_start + 8])[0],
        "flags": data[tcp_start + 13],
        "urgent_ptr": struct.unpack("!H", data[tcp_start + 18 : tcp_start + 20])[0],
    }


# ---------------------------------------------------------------------------
# Mock NFQUEUE packet
# ---------------------------------------------------------------------------

class MockNFPacket:
    """Simulates a netfilterqueue packet object."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._modified_payload: bytes | None = None
        self.accepted = False
        self.dropped = False

    def get_payload(self) -> bytes:
        return self._payload

    def set_payload(self, data: bytes) -> None:
        self._modified_payload = data

    def accept(self) -> None:
        self.accepted = True

    def drop(self) -> None:
        self.dropped = True

    @property
    def final_payload(self) -> bytes:
        return self._modified_payload if self._modified_payload is not None else self._payload


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWardenServiceConfig(unittest.TestCase):
    """Test configuration loading."""

    def test_default_config(self):
        cfg = WardenConfig()
        self.assertEqual(cfg.ttl, 64)
        self.assertEqual(cfg.queue_num, 0)
        self.assertEqual(cfg.chain, "FORWARD")
        self.assertEqual(cfg.log_level, "INFO")

    def test_env_override(self):
        env = {
            "WARDEN_SECRET": "my-test-secret",
            "WARDEN_TTL": "128",
            "WARDEN_QUEUE_NUM": "5",
            "WARDEN_CHAIN": "INPUT",
            "WARDEN_LOG_LEVEL": "debug",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = load_config("/nonexistent/path.yaml")

        self.assertEqual(cfg.secret, b"my-test-secret")
        self.assertEqual(cfg.ttl, 128)
        self.assertEqual(cfg.queue_num, 5)
        self.assertEqual(cfg.chain, "INPUT")
        self.assertEqual(cfg.log_level, "DEBUG")

    def test_yaml_config(self):
        """Test loading from a YAML file (if PyYAML is available)."""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not installed")

        content = """\
secret: yaml-secret-key
ttl: 32
queue_num: 7
stats_interval: 10
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(content)
            f.flush()

            # Remove WARDEN_ env vars temporarily to avoid override.
            old = {}
            for k in list(os.environ):
                if k.startswith("WARDEN_"):
                    old[k] = os.environ.pop(k)
            try:
                cfg = load_config(f.name)
            finally:
                os.environ.update(old)

        os.unlink(f.name)
        self.assertEqual(cfg.secret, b"yaml-secret-key")
        self.assertEqual(cfg.ttl, 32)
        self.assertEqual(cfg.queue_num, 7)
        self.assertEqual(cfg.stats_interval, 10.0)


class TestWardenServicePacketProcessing(unittest.TestCase):
    """Test the packet callback logic."""

    def _make_service(self, **kwargs):
        cfg = WardenConfig(**kwargs)
        return WardenService(cfg)

    def test_tcp_packet_accepted_and_normalized(self):
        """A valid TCP packet is normalized and accepted."""
        svc = self._make_service(secret=b"test-key")
        pkt = MockNFPacket(_build_ip_tcp_packet(
            tos=0xFF, ttl=200, ip_id=0xDEAD,
            tcp_flags=0x10, seq=5000, ack=1000,
            urgent_ptr=0xBEEF,
        ))

        svc._packet_callback(pkt)

        self.assertTrue(pkt.accepted)
        self.assertFalse(pkt.dropped)

        # Verify normalization happened.
        ip = _parse_ip_from_raw(pkt.final_payload)
        tcp = _parse_tcp_from_raw(pkt.final_payload)

        self.assertEqual(ip["tos"], 0, "TOS should be zeroed")
        self.assertEqual(ip["ttl"], 64, "TTL should be normalized to 64")
        self.assertNotEqual(ip["ip_id"], 0xDEAD, "IP ID should be rewritten")
        self.assertEqual(tcp["urgent_ptr"], 0, "Urgent ptr should be zeroed (URG not set)")

    def test_syn_isn_rewritten(self):
        """SYN packet ISN is rewritten."""
        svc = self._make_service(secret=b"isn-test")
        pkt = MockNFPacket(_build_ip_tcp_packet(
            tcp_flags=0x02, seq=0xDEADBEEF,
            src_port=50000, dst_port=80,
        ))

        svc._packet_callback(pkt)

        self.assertTrue(pkt.accepted)
        tcp = _parse_tcp_from_raw(pkt.final_payload)
        self.assertNotEqual(tcp["seq"], 0xDEADBEEF, "ISN should be rewritten")

    def test_rst_payload_stripped(self):
        """RST packet payload is removed."""
        svc = self._make_service()
        raw = _build_ip_tcp_packet(
            tcp_flags=0x14, seq=5000, ack=1000,
            payload=b"covert RST data",
        )
        pkt = MockNFPacket(raw)

        svc._packet_callback(pkt)

        self.assertTrue(pkt.accepted)
        ip = _parse_ip_from_raw(pkt.final_payload)
        # Total length should be IP header (20) + TCP header (20), no payload.
        self.assertEqual(ip["total_length"], 40)

    def test_non_ipv4_packet_dropped(self):
        """Non-IPv4 data is dropped (warden returns None for bad IP version)."""
        svc = self._make_service()
        # Craft a non-IPv4 packet (version=6 in IP header).
        bogus = bytearray(_build_ip_tcp_packet())
        bogus[0] = 0x60  # Version 6.
        pkt = MockNFPacket(bytes(bogus))

        svc._packet_callback(pkt)

        # The warden returns None for non-IPv4 -> service drops.
        self.assertTrue(pkt.dropped)

    def test_too_short_packet_dropped(self):
        """Packets too short to parse are dropped."""
        svc = self._make_service()
        pkt = MockNFPacket(b"\x45" + b"\x00" * 5)  # Too short for IP header.

        svc._packet_callback(pkt)

        self.assertTrue(pkt.dropped)

    def test_stats_accumulate(self):
        """Stats accumulate across multiple packets."""
        svc = self._make_service()
        for i in range(5):
            pkt = MockNFPacket(_build_ip_tcp_packet(
                seq=1000 + i, ack=2000,
                tcp_flags=0x10, tos=0xFF,
            ))
            svc._packet_callback(pkt)

        stats = svc.warden.stats.as_dict()
        self.assertEqual(stats["frames_processed"], 5)
        self.assertEqual(stats["frames_passed"], 5)
        self.assertEqual(stats["tos_zeroed"], 5)

    def test_custom_ttl(self):
        """Custom TTL is applied."""
        svc = self._make_service(secret=b"test", ttl=128)
        pkt = MockNFPacket(_build_ip_tcp_packet(ttl=64))

        svc._packet_callback(pkt)

        ip = _parse_ip_from_raw(pkt.final_payload)
        self.assertEqual(ip["ttl"], 128)

    def test_covert_ip_id_channel_destroyed(self):
        """Verify that IP-ID-based covert channel is destroyed through the service."""
        svc = self._make_service(secret=b"anti-steg")
        secret_chars = [ord("S"), ord("T"), ord("E"), ord("G")]
        recovered = []

        for char_val in secret_chars:
            pkt = MockNFPacket(_build_ip_tcp_packet(
                ip_id=char_val, tcp_flags=0x10, seq=5000, ack=1000,
            ))
            svc._packet_callback(pkt)
            ip = _parse_ip_from_raw(pkt.final_payload)
            recovered.append(ip["ip_id"])

        self.assertNotEqual(recovered, secret_chars,
                            "IP-ID covert channel should be destroyed")


class TestWardenServiceLifecycle(unittest.TestCase):
    """Test service start/stop without actual NFQUEUE."""

    def test_service_creates_warden(self):
        cfg = WardenConfig(secret=b"lifecycle-test", ttl=32)
        svc = WardenService(cfg)

        self.assertEqual(svc.warden._ttl, 32)
        self.assertEqual(svc.warden._secret, b"lifecycle-test")

    def test_stop_is_safe_without_start(self):
        """Calling stop() without run() should not raise."""
        cfg = WardenConfig()
        svc = WardenService(cfg)
        svc.stop()  # Should not raise.


class TestFakeEthHeader(unittest.TestCase):
    """Test the fake Ethernet header prepended for L3->L2 wrapping."""

    def test_fake_eth_header_is_ipv4(self):
        self.assertEqual(len(FAKE_ETH), 14)
        ethertype = struct.unpack("!H", FAKE_ETH[12:14])[0]
        self.assertEqual(ethertype, 0x0800)

    def test_roundtrip_preserves_ip_payload(self):
        """Prepending and stripping fake ETH header preserves IP data."""
        raw_ip = _build_ip_tcp_packet()
        frame = FAKE_ETH + raw_ip
        recovered = frame[ETH_HEADER_LEN:]
        self.assertEqual(recovered, raw_ip)


class TestWardenServiceCapture(unittest.TestCase):
    """Test capture ring and raw packet recording."""

    def _make_service(self, **kwargs):
        config = WardenConfig(**kwargs)
        return WardenService(config)

    def test_capture_records_normalized_frames(self):
        svc = self._make_service(secret=b"capture-test", skip_isn_rewrite=True)
        raw = _build_ip_tcp_packet(tcp_flags=0x10, seq=5000, ack=1000)
        pkt = MockNFPacket(raw)
        svc._packet_callback(pkt)
        self.assertEqual(svc.capture.frame_count, 1)
        self.assertTrue(svc.capture_digest().startswith("sha256:"))

    def test_capture_reset_clears(self):
        svc = self._make_service(skip_isn_rewrite=True)
        raw = _build_ip_tcp_packet(tcp_flags=0x10, seq=5000, ack=1000)
        svc._packet_callback(MockNFPacket(raw))
        svc.capture_reset()
        self.assertEqual(svc.capture.frame_count, 0)
        self.assertEqual(len(svc.raw_packets()), 0)

    def test_raw_packets_recorded(self):
        svc = self._make_service(skip_isn_rewrite=True)
        raw = _build_ip_tcp_packet(tcp_flags=0x10, seq=5000, ack=1000)
        svc._packet_callback(MockNFPacket(raw))
        self.assertEqual(len(svc.raw_packets()), 1)
        self.assertEqual(svc.raw_packets()[0], raw)

    def test_determinism_replay(self):
        """Replaying raw packets through a fresh warden produces identical output."""
        from pkg.networkdet.warden import ActiveWarden
        secret = b"replay-test"
        svc = self._make_service(secret=secret, skip_isn_rewrite=True)
        packets = [
            _build_ip_tcp_packet(tcp_flags=0x10, seq=5000 + i, ack=1000)
            for i in range(3)
        ]
        for raw in packets:
            svc._packet_callback(MockNFPacket(raw))

        captured = svc.capture.drain()
        fresh = ActiveWarden(secret=secret, skip_isn_rewrite=True)
        for raw, expected in zip(packets, captured):
            frame = FAKE_ETH + raw
            result = fresh.normalize(frame)
            self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
