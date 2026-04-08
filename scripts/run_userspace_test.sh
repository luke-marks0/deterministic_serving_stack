#!/bin/bash
set -e

cd /opt/deterministic_serving_stack

# Clean up any previous runs
pkill -f 'userspace_tcp_server' 2>/dev/null || true
killall tcpdump 2>/dev/null || true
sleep 1
iptables -D OUTPUT -p tcp --tcp-flags RST RST --sport 9999 -j DROP 2>/dev/null || true

# Get network info
GWIP=$(ip route show default | awk '{print $3}')
echo "Gateway IP: $GWIP"

# Ensure gateway is in ARP cache
ping -c 1 -W 1 "$GWIP" > /dev/null 2>&1 || true

# Parse gateway MAC from 'ip neigh' output
# Format: 143.198.112.1 lladdr fe:00:00:00:01:01 REACHABLE
GWMAC=$(ip neigh show "$GWIP" dev eth0 | grep -oP 'lladdr \K[0-9a-f:]+')
OURMAC=$(ip link show eth0 | grep -oP 'ether \K[0-9a-f:]+')
OURIP=$(ip addr show eth0 | grep -oP 'inet \K[0-9.]+' | head -1)

echo "Gateway MAC: $GWMAC"
echo "Our MAC: $OURMAC"
echo "Our IP: $OURIP"

if [ -z "$GWMAC" ]; then
    echo "ERROR: Could not resolve gateway MAC"
    ip neigh show dev eth0
    exit 1
fi

# Start the server in background
echo "Starting userspace TCP server..."
python3 -m pkg.networkdet.userspace_tcp_server \
    --port 9999 \
    --interface eth0 \
    --mss 1460 \
    --local-ip "$OURIP" \
    --local-mac "$OURMAC" \
    --gateway-mac "$GWMAC" \
    --run-id "poc-test-run" \
    > /tmp/userspace_server.log 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
sleep 3

# Check server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server failed to start"
    cat /tmp/userspace_server.log
    exit 1
fi

echo "Server is running. Server log so far:"
cat /tmp/userspace_server.log
echo ""

# For same-machine testing we need to work around the kernel
# claiming packets to its own IP. We use a network namespace
# to create a separate network stack that sends packets on the wire.
echo "=== Setting up test client network namespace ==="
ip netns del testclient 2>/dev/null || true

# Create a veth pair: veth-host <-> veth-client
ip link del veth-host 2>/dev/null || true
ip link add veth-host type veth peer name veth-client

# Put veth-client in the namespace
ip netns add testclient
ip link set veth-client netns testclient

# Configure host side
ip addr add 10.99.0.1/24 dev veth-host
ip link set veth-host up

# Configure client side (in namespace)
ip netns exec testclient ip addr add 10.99.0.2/24 dev veth-client
ip netns exec testclient ip link set veth-client up
ip netns exec testclient ip link set lo up

# Route traffic from the namespace to the real interface via NAT
# The client in the namespace sends to $OURIP:9999.
# We need to route it so it exits on eth0.
ip netns exec testclient ip route add default via 10.99.0.1
echo 1 > /proc/sys/net/ipv4/ip_forward

# But wait - this still goes through the kernel TCP stack on the host.
# The whole point is that our AF_PACKET server on eth0 needs to see the SYN.
# For same-machine testing, the easiest approach: use a raw socket client
# that sends a SYN directly via AF_PACKET on eth0, or just accept that
# same-machine testing requires special handling.
#
# Actually, the simplest approach for same-machine:
# Since our server listens on AF_PACKET on eth0, packets arriving
# from outside the machine ARE what we want. Let's just test externally.
# But we can also test same-machine by making the kernel NOT own port 9999
# (which we already do - no kernel socket on 9999) and adding iptables
# rules to prevent the kernel from interfering.

# Clean up the namespace approach - it adds complexity
ip netns del testclient 2>/dev/null || true
ip link del veth-host 2>/dev/null || true

echo ""
echo "=== Test: Same-machine curl (may not work due to kernel routing) ==="
echo "The server is ready for external testing on $OURIP:9999"
echo ""

# Try same-machine first - the kernel might send SYN on eth0 when
# connecting to its own public IP (DO uses the public IP on eth0 directly)
tcpdump -i eth0 -nn -w /tmp/test1.pcap "tcp port 9999" -c 30 &
TCPDUMP_PID=$!
sleep 1

echo "Sending curl request..."
CURL_OUTPUT=$(curl -v -s --max-time 10 --connect-timeout 5 "http://${OURIP}:9999/deterministic" 2>&1) || true
CURL_RC=$?
echo "curl exit code: $CURL_RC"
echo "curl output (first 500 chars):"
echo "$CURL_OUTPUT" | head -30

sleep 3
kill $TCPDUMP_PID 2>/dev/null || true
wait $TCPDUMP_PID 2>/dev/null || true

echo ""
echo "=== Packet capture ==="
tcpdump -nn -r /tmp/test1.pcap 2>/dev/null || echo "No pcap"

echo ""
echo "=== Server log ==="
cat /tmp/userspace_server.log

# Leave server running for external testing
echo ""
echo "============================================="
echo "Server still running on $OURIP:9999 (PID $SERVER_PID)"
echo "Test externally: curl http://$OURIP:9999/deterministic"
echo "Kill with: kill $SERVER_PID"
echo "============================================="
