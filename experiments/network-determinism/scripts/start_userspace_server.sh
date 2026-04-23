#!/bin/bash
# Start the userspace TCP server on the droplet.
# Run this, then test from an external machine:
#   curl http://143.198.114.248:9999/deterministic
set -e

cd /opt/deterministic_serving_stack

# Clean up any previous runs
pkill -f 'userspace_tcp_server' 2>/dev/null || true
killall tcpdump 2>/dev/null || true
sleep 1

# Remove old iptables rules (ignore errors)
iptables -D OUTPUT -p tcp --tcp-flags RST RST --sport 9999 -j DROP 2>/dev/null || true

# Also need to prevent the kernel from sending RST for incoming SYNs
# to a port it doesn't own. We drop outgoing RSTs from our port.
# Additionally, we need to prevent the kernel from processing incoming
# TCP packets on port 9999 at all. We can use iptables to DROP incoming
# TCP on port 9999 in the INPUT chain - this prevents the kernel TCP
# stack from seeing it, but AF_PACKET still captures it (AF_PACKET
# operates before iptables/netfilter).
iptables -D INPUT -p tcp --dport 9999 -j DROP 2>/dev/null || true

# Add the rules
iptables -A OUTPUT -p tcp --tcp-flags RST RST --sport 9999 -j DROP
iptables -A INPUT -p tcp --dport 9999 -j DROP

echo "iptables rules set:"
iptables -L -n | grep 9999

# Get network info
GWIP=$(ip route show default | awk '{print $3}')
GWMAC=$(ip neigh show "$GWIP" dev eth0 | grep -oP 'lladdr \K[0-9a-f:]+')
OURMAC=$(ip link show eth0 | grep -oP 'ether \K[0-9a-f:]+')
OURIP=$(ip addr show eth0 | grep -oP 'inet \K[0-9.]+' | head -1)

echo "Interface: eth0"
echo "Our IP: $OURIP, MAC: $OURMAC"
echo "Gateway MAC: $GWMAC"

if [ -z "$GWMAC" ]; then
    echo "ERROR: Could not resolve gateway MAC"
    ping -c 1 -W 1 "$GWIP" > /dev/null 2>&1 || true
    GWMAC=$(ip neigh show "$GWIP" dev eth0 | grep -oP 'lladdr \K[0-9a-f:]+')
    echo "Retried, Gateway MAC: $GWMAC"
    if [ -z "$GWMAC" ]; then
        echo "FATAL: Still no gateway MAC"
        exit 1
    fi
fi

# Start tcpdump in background for analysis
tcpdump -i eth0 -nn -w /tmp/userspace_tcp.pcap "tcp port 9999" -c 100 &
TCPDUMP_PID=$!
echo "tcpdump PID: $TCPDUMP_PID"

# Start the server (foreground for log visibility)
echo ""
echo "Starting userspace TCP server on $OURIP:9999 ..."
echo "Test from external machine: curl http://$OURIP:9999/deterministic"
echo "Press Ctrl+C to stop."
echo ""

python3 -m pkg.networkdet.userspace_tcp_server \
    --port 9999 \
    --interface eth0 \
    --mss 1460 \
    --local-ip "$OURIP" \
    --local-mac "$OURMAC" \
    --gateway-mac "$GWMAC" \
    --run-id "poc-test-run"

# Cleanup on exit
kill $TCPDUMP_PID 2>/dev/null || true
iptables -D OUTPUT -p tcp --tcp-flags RST RST --sport 9999 -j DROP 2>/dev/null || true
iptables -D INPUT -p tcp --dport 9999 -j DROP 2>/dev/null || true
