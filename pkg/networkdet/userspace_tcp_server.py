"""Userspace TCP server using AF_PACKET raw sockets.

Accepts real TCP connections from clients (curl, urllib, etc.) by:
1. Capturing incoming SYN/ACK/data via AF_PACKET
2. Building response frames with DeterministicTCPConnection
3. Transmitting L2 frames via AF_PACKET

Requires:
- Root privileges (AF_PACKET needs CAP_NET_RAW)
- iptables rule to suppress kernel RST:
    iptables -A OUTPUT -p tcp --tcp-flags RST RST --sport <PORT> -j DROP
- Knowledge of local MAC, gateway MAC, and local IP

Usage:
    python -m pkg.networkdet.userspace_tcp_server --port 9999 --interface eth0
"""
from __future__ import annotations

import argparse
import json
import logging
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

# checksums are computed by DeterministicTCPConnection and DeterministicIPLayer
from pkg.networkdet.ethernet import (
    ETHERNET_HEADER_LEN,
    ETHERTYPE_IPV4,
    build_ethernet_frame,
    mac_to_bytes,
)
from pkg.networkdet.ip import IPV4_HEADER_LEN, PROTO_TCP, DeterministicIPLayer
from pkg.networkdet.tcp import (
    ACK,
    FIN,
    PSH,
    RST,
    SYN,
    TCP_HEADER_LEN,
    DeterministicTCPConnection,
    TCPState,
    deterministic_isn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# --- Fixed HTTP responses for determinism testing ---

# Small response (fits in one segment)
SMALL_BODY = json.dumps(
    {
        "model": "deterministic-test",
        "tokens": [1, 2, 3, 4, 5],
        "request_id": "fixed-request-001",
        "server": "userspace-tcp",
    },
    separators=(",", ":"),
).encode("utf-8")

# Large response (requires multiple MSS-sized segments)
# ~5000 bytes of deterministic JSON data
LARGE_BODY = json.dumps(
    {
        "model": "deterministic-test",
        "tokens": list(range(500)),
        "request_id": "fixed-request-002",
        "server": "userspace-tcp",
        "padding": "X" * 2000,
    },
    separators=(",", ":"),
).encode("utf-8")


def _build_response(body: bytes) -> bytes:
    """Build a complete HTTP response with deterministic headers."""
    headers = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: application/json\r\n"
        b"Content-Length: " + str(len(body)).encode() + b"\r\n"
        b"Connection: close\r\n"
        b"Server: DeterministicTCP/1.0\r\n"
        b"Date: Thu, 01 Jan 2026 00:00:00 GMT\r\n"
        b"\r\n"
    )
    return headers + body


RESPONSES: dict[str, bytes] = {
    "/deterministic": _build_response(SMALL_BODY),
    "/large": _build_response(LARGE_BODY),
}

# Default response for unknown paths
NOT_FOUND_RESPONSE = (
    b"HTTP/1.1 404 Not Found\r\n"
    b"Content-Length: 0\r\n"
    b"Connection: close\r\n"
    b"\r\n"
)

# Keep backward compat
FULL_RESPONSE = RESPONSES["/deterministic"]


@dataclass
class ParsedPacket:
    """Parsed fields from an incoming Ethernet/IP/TCP packet."""

    # Ethernet
    dst_mac: bytes
    src_mac: bytes
    ethertype: int

    # IP
    src_ip: bytes
    dst_ip: bytes
    ip_id: int
    ip_total_len: int
    protocol: int

    # TCP
    src_port: int
    dst_port: int
    seq: int
    ack: int
    data_offset: int  # in bytes
    flags: int
    window: int
    payload: bytes

    @property
    def src_ip_str(self) -> str:
        return socket.inet_ntoa(self.src_ip)

    @property
    def dst_ip_str(self) -> str:
        return socket.inet_ntoa(self.dst_ip)

    @property
    def is_syn(self) -> bool:
        return bool(self.flags & SYN) and not bool(self.flags & ACK)

    @property
    def is_ack(self) -> bool:
        return bool(self.flags & ACK) and not bool(self.flags & SYN) and not bool(self.flags & FIN)

    @property
    def is_fin(self) -> bool:
        return bool(self.flags & FIN)

    @property
    def is_rst(self) -> bool:
        return bool(self.flags & RST)

    @property
    def has_data(self) -> bool:
        return len(self.payload) > 0


def parse_packet(raw: bytes) -> Optional[ParsedPacket]:
    """Parse a raw L2 frame into structured fields. Returns None if not TCP/IPv4."""
    if len(raw) < ETHERNET_HEADER_LEN + IPV4_HEADER_LEN + TCP_HEADER_LEN:
        return None

    # Ethernet
    dst_mac = raw[0:6]
    src_mac = raw[6:12]
    ethertype = struct.unpack("!H", raw[12:14])[0]

    if ethertype != ETHERTYPE_IPV4:
        return None

    # IP
    ip_start = ETHERNET_HEADER_LEN
    ip_ver_ihl = raw[ip_start]
    if (ip_ver_ihl >> 4) != 4:
        return None

    ip_header_len = (ip_ver_ihl & 0x0F) * 4
    ip_total_len = struct.unpack("!H", raw[ip_start + 2 : ip_start + 4])[0]
    ip_id = struct.unpack("!H", raw[ip_start + 4 : ip_start + 6])[0]
    protocol = raw[ip_start + 9]

    if protocol != PROTO_TCP:
        return None

    src_ip = raw[ip_start + 12 : ip_start + 16]
    dst_ip = raw[ip_start + 16 : ip_start + 20]

    # TCP
    tcp_start = ip_start + ip_header_len
    if len(raw) < tcp_start + TCP_HEADER_LEN:
        return None

    src_port = struct.unpack("!H", raw[tcp_start : tcp_start + 2])[0]
    dst_port = struct.unpack("!H", raw[tcp_start + 2 : tcp_start + 4])[0]
    seq = struct.unpack("!I", raw[tcp_start + 4 : tcp_start + 8])[0]
    ack_num = struct.unpack("!I", raw[tcp_start + 8 : tcp_start + 12])[0]
    data_offset = ((raw[tcp_start + 12] >> 4) & 0x0F) * 4
    flags = raw[tcp_start + 13]
    window = struct.unpack("!H", raw[tcp_start + 14 : tcp_start + 16])[0]

    # TCP payload
    payload_start = tcp_start + data_offset
    payload_end = ip_start + ip_total_len
    payload = raw[payload_start:payload_end] if payload_end > payload_start else b""

    return ParsedPacket(
        dst_mac=dst_mac,
        src_mac=src_mac,
        ethertype=ethertype,
        src_ip=src_ip,
        dst_ip=dst_ip,
        ip_id=ip_id,
        ip_total_len=ip_total_len,
        protocol=protocol,
        src_port=src_port,
        dst_port=dst_port,
        seq=seq,
        ack=ack_num,
        data_offset=data_offset,
        flags=flags,
        window=window,
        payload=payload,
    )


class ConnectionState:
    """Tracks one userspace TCP connection."""

    def __init__(
        self,
        client_ip: str,
        client_port: int,
        client_mac: bytes,
        server_ip: str,
        server_port: int,
        server_mac: bytes,
        client_isn: int,
        run_id: str,
        conn_index: int,
        mss: int = 1460,
    ):
        self.client_ip = client_ip
        self.client_port = client_port
        self.client_mac = client_mac
        self.server_ip = server_ip
        self.server_port = server_port
        self.server_mac = server_mac
        self.client_isn = client_isn
        self.mss = mss

        # Build the deterministic TCP connection for our server->client direction
        isn = deterministic_isn(run_id, conn_index)
        self.ip_layer = DeterministicIPLayer(server_ip, client_ip)
        self.tcp = DeterministicTCPConnection(
            server_port,
            client_port,
            isn=isn,
            mss=mss,
            window=65535,
            src_ip=self.ip_layer.src_ip,
            dst_ip=self.ip_layer.dst_ip,
        )

        # Track what we expect from the client
        self.expected_client_seq = (client_isn + 1) & 0xFFFFFFFF
        self.response_sent = False
        self.fin_sent = False
        self.closed = False
        self.created_at = time.monotonic()

    def wrap_frame(self, tcp_segment: bytes) -> bytes:
        """Wrap a TCP segment in IP + Ethernet."""
        ip_packet = self.ip_layer.build_packet(PROTO_TCP, tcp_segment)
        return build_ethernet_frame(self.client_mac, self.server_mac, ip_packet)


class UserspaceServer:
    """Userspace TCP server using AF_PACKET."""

    def __init__(
        self,
        interface: str,
        port: int,
        local_ip: str,
        local_mac: str,
        gateway_mac: str,
        *,
        mss: int = 1460,
        run_id: str = "userspace-poc-run",
    ):
        self.interface = interface
        self.port = port
        self.local_ip = local_ip
        self.local_mac_bytes = mac_to_bytes(local_mac)
        self.local_mac_str = local_mac
        self.gateway_mac_bytes = mac_to_bytes(gateway_mac)
        self.gateway_mac_str = gateway_mac
        self.mss = mss
        self.run_id = run_id

        # Connection tracking: (client_ip, client_port) -> ConnectionState
        self.connections: dict[tuple[str, int], ConnectionState] = {}
        self.conn_counter = 0

        # Sockets
        self._rx_sock: Optional[socket.socket] = None
        self._tx_sock: Optional[socket.socket] = None

        # Stats
        self.stats = {
            "syns_received": 0,
            "handshakes_completed": 0,
            "responses_sent": 0,
            "fins_sent": 0,
            "packets_rx": 0,
            "packets_tx": 0,
        }

    def _setup_sockets(self) -> None:
        """Create AF_PACKET sockets for RX and TX."""
        # RX: capture all packets on the interface
        self._rx_sock = socket.socket(
            socket.AF_PACKET, socket.SOCK_RAW, socket.htons(0x0003)  # ETH_P_ALL
        )
        self._rx_sock.bind((self.interface, 0))
        self._rx_sock.settimeout(0.1)  # 100ms timeout for polling

        # TX: send raw L2 frames
        self._tx_sock = socket.socket(
            socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ETHERTYPE_IPV4)
        )
        self._tx_sock.bind((self.interface, 0))

    def _send_frame(self, frame: bytes) -> None:
        """Send a raw L2 frame."""
        assert self._tx_sock is not None
        self._tx_sock.send(frame)
        self.stats["packets_tx"] += 1

    def _handle_syn(self, pkt: ParsedPacket) -> None:
        """Handle an incoming SYN: send SYN-ACK."""
        key = (pkt.src_ip_str, pkt.src_port)

        # Determine the MAC to respond to.
        # For same-machine testing via public IP, the src_mac is the interface's own MAC.
        # For remote clients, src_mac is the gateway's MAC (since we're doing L2).
        # We just reply to whoever sent us the SYN.
        reply_mac = pkt.src_mac

        self.conn_counter += 1
        conn = ConnectionState(
            client_ip=pkt.src_ip_str,
            client_port=pkt.src_port,
            client_mac=reply_mac,
            server_ip=pkt.dst_ip_str,
            server_port=self.port,
            server_mac=self.local_mac_bytes,
            client_isn=pkt.seq,
            run_id=self.run_id,
            conn_index=self.conn_counter,
            mss=self.mss,
        )
        self.connections[key] = conn

        # Build and send SYN-ACK
        syn_ack_segment = conn.tcp.build_syn_ack(pkt.seq)
        frame = conn.wrap_frame(syn_ack_segment)
        self._send_frame(frame)

        self.stats["syns_received"] += 1
        log.info(
            "SYN from %s:%d (ISN=%d) -> SYN-ACK (our ISN=%d)",
            pkt.src_ip_str, pkt.src_port, pkt.seq,
            conn.tcp.seq - 1,  # seq was already incremented
        )

    def _handle_ack(self, pkt: ParsedPacket, conn: ConnectionState) -> None:
        """Handle ACK (handshake completion or data ACK)."""
        if conn.tcp.state == TCPState.SYN_RECEIVED:
            # Handshake complete
            conn.tcp.receive_ack(pkt.ack)
            self.stats["handshakes_completed"] += 1
            log.info(
                "Handshake complete with %s:%d",
                pkt.src_ip_str, pkt.src_port,
            )

        # If client sent data (HTTP request), send our response
        if pkt.has_data and not conn.response_sent:
            # Update our ack to reflect received data
            conn.tcp._ack = (pkt.seq + len(pkt.payload)) & 0xFFFFFFFF
            conn.expected_client_seq = conn.tcp._ack

            log.info(
                "Received %d bytes of request data from %s:%d",
                len(pkt.payload), pkt.src_ip_str, pkt.src_port,
            )

            # Parse HTTP request to find the path
            path = "/deterministic"  # default
            try:
                request_line = pkt.payload.split(b"\r\n")[0].decode("utf-8", errors="replace")
                parts = request_line.split(" ")
                if len(parts) >= 2:
                    path = parts[1]
            except Exception:
                pass

            # Select response based on path
            response_data = RESPONSES.get(path, NOT_FOUND_RESPONSE)

            # Send the deterministic HTTP response
            self._send_response(conn, response_data)

    def _send_response(self, conn: ConnectionState, response_data: bytes = FULL_RESPONSE) -> None:
        """Send the HTTP response as deterministically segmented frames."""
        # Segment the response at fixed MSS boundaries
        data_segments = conn.tcp.segment_data(response_data)
        log.info(
            "Sending response: %d bytes in %d segments (MSS=%d) to %s:%d",
            len(response_data), len(data_segments), self.mss,
            conn.client_ip, conn.client_port,
        )

        for i, segment in enumerate(data_segments):
            frame = conn.wrap_frame(segment)
            self._send_frame(frame)
            log.info("  Sent segment %d/%d (%d bytes)", i + 1, len(data_segments), len(segment))

        conn.response_sent = True
        self.stats["responses_sent"] += 1

        # Send FIN after response
        fin_segment = conn.tcp.build_fin()
        frame = conn.wrap_frame(fin_segment)
        self._send_frame(frame)
        conn.fin_sent = True
        self.stats["fins_sent"] += 1
        log.info("Sent FIN to %s:%d", conn.client_ip, conn.client_port)

    def _handle_fin(self, pkt: ParsedPacket, conn: ConnectionState) -> None:
        """Handle incoming FIN from client."""
        # ACK the FIN
        conn.tcp._ack = (pkt.seq + 1) & 0xFFFFFFFF
        ack_segment = conn.tcp.build_ack()
        frame = conn.wrap_frame(ack_segment)
        self._send_frame(frame)
        conn.closed = True
        log.info("Received FIN from %s:%d, sent ACK", pkt.src_ip_str, pkt.src_port)

    def _process_packet(self, raw: bytes) -> None:
        """Process one captured packet."""
        pkt = parse_packet(raw)
        if pkt is None:
            return

        # Only handle packets destined for our port
        if pkt.dst_port != self.port:
            return

        # Only handle packets destined for our IP
        if pkt.dst_ip_str != self.local_ip:
            return

        self.stats["packets_rx"] += 1
        key = (pkt.src_ip_str, pkt.src_port)

        if pkt.is_rst:
            # Client sent RST — clean up
            if key in self.connections:
                log.info("RST from %s:%d, closing connection", pkt.src_ip_str, pkt.src_port)
                del self.connections[key]
            return

        if pkt.is_syn:
            self._handle_syn(pkt)
            return

        conn = self.connections.get(key)
        if conn is None:
            # Unknown connection — ignore
            return

        if pkt.is_fin:
            self._handle_fin(pkt, conn)
            return

        if pkt.is_ack:
            self._handle_ack(pkt, conn)

    def _setup_iptables(self) -> None:
        """Add iptables rule to suppress kernel RST."""
        rule = [
            "iptables", "-A", "OUTPUT",
            "-p", "tcp",
            "--tcp-flags", "RST", "RST",
            "--sport", str(self.port),
            "-j", "DROP",
        ]
        try:
            subprocess.run(rule, check=True, capture_output=True)
            log.info("iptables RST suppression rule added for port %d", self.port)
        except subprocess.CalledProcessError as e:
            log.warning("Failed to add iptables rule: %s", e.stderr.decode())

    def _cleanup_iptables(self) -> None:
        """Remove iptables rule."""
        rule = [
            "iptables", "-D", "OUTPUT",
            "-p", "tcp",
            "--tcp-flags", "RST", "RST",
            "--sport", str(self.port),
            "-j", "DROP",
        ]
        try:
            subprocess.run(rule, check=True, capture_output=True)
            log.info("iptables RST suppression rule removed")
        except subprocess.CalledProcessError:
            pass

    def _cleanup_stale_connections(self) -> None:
        """Remove connections older than 10 seconds."""
        now = time.monotonic()
        stale = [k for k, v in self.connections.items() if now - v.created_at > 10]
        for k in stale:
            log.info("Cleaning up stale connection %s:%d", k[0], k[1])
            del self.connections[k]

    def serve_forever(self) -> None:
        """Main server loop."""
        self._setup_sockets()
        self._setup_iptables()

        log.info(
            "Userspace TCP server listening on %s:%d (interface=%s, MAC=%s)",
            self.local_ip, self.port, self.interface, self.local_mac_str,
        )
        log.info("Response: %d bytes, MSS=%d", len(FULL_RESPONSE), self.mss)

        try:
            cleanup_counter = 0
            while True:
                try:
                    raw = self._rx_sock.recv(65535)
                    self._process_packet(raw)
                except socket.timeout:
                    pass
                except OSError as e:
                    if e.errno == 100:  # Network down
                        log.warning("Network down, retrying...")
                        time.sleep(1)
                    else:
                        raise

                cleanup_counter += 1
                if cleanup_counter > 1000:
                    self._cleanup_stale_connections()
                    cleanup_counter = 0

        except KeyboardInterrupt:
            log.info("Shutting down...")
        finally:
            self._cleanup_iptables()
            if self._rx_sock:
                self._rx_sock.close()
            if self._tx_sock:
                self._tx_sock.close()
            log.info("Stats: %s", self.stats)


def get_interface_info(interface: str) -> tuple[str, str]:
    """Get IP and MAC address of an interface."""
    import fcntl

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Get IP
    ip_bytes = fcntl.ioctl(
        sock.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack("256s", interface.encode("utf-8")[:15]),
    )
    ip_addr = socket.inet_ntoa(ip_bytes[20:24])

    # Get MAC
    mac_bytes = fcntl.ioctl(
        sock.fileno(),
        0x8927,  # SIOCGIFHWADDR
        struct.pack("256s", interface.encode("utf-8")[:15]),
    )
    mac_addr = ":".join(f"{b:02x}" for b in mac_bytes[18:24])

    sock.close()
    return ip_addr, mac_addr


def get_gateway_mac(interface: str) -> str:
    """Get the default gateway's MAC address from the ARP table."""
    # Get default gateway IP
    result = subprocess.run(
        ["ip", "route", "show", "default", "dev", interface],
        capture_output=True, text=True,
    )
    gateway_ip = result.stdout.split()[2] if result.stdout else None
    if not gateway_ip:
        raise RuntimeError(f"No default gateway found for {interface}")

    # Get gateway MAC from ARP table
    result = subprocess.run(
        ["ip", "neigh", "show", gateway_ip, "dev", interface],
        capture_output=True, text=True,
    )
    parts = result.stdout.strip().split()
    for i, p in enumerate(parts):
        if p == "lladdr" and i + 1 < len(parts):
            return parts[i + 1]

    # If not in ARP cache, ping it first
    subprocess.run(["ping", "-c", "1", "-W", "1", gateway_ip], capture_output=True)
    result = subprocess.run(
        ["ip", "neigh", "show", gateway_ip, "dev", interface],
        capture_output=True, text=True,
    )
    parts = result.stdout.strip().split()
    for i, p in enumerate(parts):
        if p == "lladdr" and i + 1 < len(parts):
            return parts[i + 1]

    raise RuntimeError(f"Could not resolve gateway MAC for {gateway_ip}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Userspace TCP server")
    parser.add_argument("--port", type=int, default=9999, help="Listen port")
    parser.add_argument("--interface", default="eth0", help="Network interface")
    parser.add_argument("--mss", type=int, default=1460, help="Maximum segment size")
    parser.add_argument("--local-ip", help="Override local IP detection")
    parser.add_argument("--local-mac", help="Override local MAC detection")
    parser.add_argument("--gateway-mac", help="Override gateway MAC detection")
    parser.add_argument("--run-id", default="userspace-poc-run", help="Deterministic run ID")
    args = parser.parse_args()

    if args.local_ip and args.local_mac:
        local_ip = args.local_ip
        local_mac = args.local_mac
    else:
        local_ip, local_mac = get_interface_info(args.interface)
        if args.local_ip:
            local_ip = args.local_ip
        if args.local_mac:
            local_mac = args.local_mac

    if args.gateway_mac:
        gateway_mac = args.gateway_mac
    else:
        gateway_mac = get_gateway_mac(args.interface)

    log.info("Interface: %s", args.interface)
    log.info("Local IP: %s, MAC: %s", local_ip, local_mac)
    log.info("Gateway MAC: %s", gateway_mac)

    server = UserspaceServer(
        interface=args.interface,
        port=args.port,
        local_ip=local_ip,
        local_mac=local_mac,
        gateway_mac=gateway_mac,
        mss=args.mss,
        run_id=args.run_id,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
