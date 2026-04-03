"""Inline active warden service -- NFQUEUE-based packet normalizer.

Intercepts packets via iptables NFQUEUE, passes them through the
ActiveWarden MRF normalizer, and re-injects the scrubbed packets.

This module contains the service logic; the entry point is in
cmd/warden/main.py.
"""
from __future__ import annotations

import logging
import struct
import threading

from pkg.networkdet.capture import CaptureRing
from pkg.networkdet.warden import ActiveWarden, ETH_HEADER_LEN
from pkg.networkdet.warden_config import WardenConfig

logger = logging.getLogger("warden")

# Sentinel for clean shutdown.
_shutdown = threading.Event()


def _build_fake_eth_header() -> bytes:
    """Build a dummy Ethernet header for NFQUEUE packets.

    NFQUEUE delivers L3 (IP) payloads, but ActiveWarden expects L2 frames
    starting with an Ethernet header.  We prepend a minimal Ethernet header
    with ethertype 0x0800 (IPv4).
    """
    dst_mac = b"\x00\x00\x00\x00\x00\x00"
    src_mac = b"\x00\x00\x00\x00\x00\x00"
    return struct.pack("!6s6sH", dst_mac, src_mac, 0x0800)


FAKE_ETH = _build_fake_eth_header()


class WardenService:
    """Wraps ActiveWarden around an NFQUEUE binding."""

    def __init__(self, config: WardenConfig) -> None:
        self.config = config
        self.warden = ActiveWarden(
            secret=config.secret, ttl=config.ttl,
            skip_isn_rewrite=config.skip_isn_rewrite,
        )
        self._nfqueue = None
        self._stats_thread: threading.Thread | None = None
        self.capture = CaptureRing()
        self._raw_packets: list[bytes] = []

    def _packet_callback(self, pkt) -> None:
        """Called for every packet pulled from the NFQUEUE."""
        try:
            raw_ip = pkt.get_payload()

            # Wrap in a fake Ethernet frame for the warden.
            frame = FAKE_ETH + raw_ip

            normalized = self.warden.normalize(frame)

            if normalized is None:
                pkt.drop()
                return

            # Record for determinism verification.
            self.capture.record(normalized)
            self._raw_packets.append(bytes(raw_ip))

            # Strip the fake Ethernet header to get back to L3.
            normalized_ip = normalized[ETH_HEADER_LEN:]
            pkt.set_payload(normalized_ip)
            pkt.accept()

        except Exception:
            logger.exception("Error processing packet, accepting unchanged")
            try:
                pkt.accept()
            except Exception:
                pass

    def _log_stats_loop(self) -> None:
        """Periodically log warden statistics."""
        while not _shutdown.wait(self.config.stats_interval):
            stats = self.warden.stats.as_dict()
            logger.info("warden stats: %s", stats)

    def run(self) -> None:
        """Bind to NFQUEUE and start processing packets."""
        try:
            from netfilterqueue import NetfilterQueue
        except ImportError:
            logger.error(
                "netfilterqueue not installed. "
                "Install with: pip install NetfilterQueue"
            )
            import sys
            sys.exit(1)

        self._nfqueue = NetfilterQueue()
        self._nfqueue.bind(self.config.queue_num, self._packet_callback)

        try:
            self._nfqueue.maxlen = self.config.max_queue_len
        except AttributeError:
            pass  # Older versions may not support maxlen.

        # Start stats logging thread.
        self._stats_thread = threading.Thread(
            target=self._log_stats_loop, daemon=True
        )
        self._stats_thread.start()

        logger.info(
            "Warden started: queue=%d chain=%s ttl=%d stats_interval=%.0fs",
            self.config.queue_num,
            self.config.chain,
            self.config.ttl,
            self.config.stats_interval,
        )

        try:
            self._nfqueue.run()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def capture_digest(self) -> str:
        """SHA-256 digest over all captured normalized frames."""
        return self.capture.digest()

    def capture_reset(self) -> None:
        """Clear capture ring and raw packet buffer."""
        self.capture = CaptureRing()
        self._raw_packets.clear()

    def raw_packets(self) -> list[bytes]:
        """Return copies of raw L3 packets received before normalization."""
        return list(self._raw_packets)

    def stop(self) -> None:
        """Unbind from NFQUEUE and shut down."""
        _shutdown.set()
        if self._nfqueue is not None:
            try:
                self._nfqueue.unbind()
            except Exception:
                pass
        logger.info(
            "Warden stopped. Final stats: %s", self.warden.stats.as_dict()
        )
