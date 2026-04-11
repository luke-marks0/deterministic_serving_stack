#!/usr/bin/env bash
# Orchestrates the userspace vs kernel TCP benchmark across two DO droplets.
#
# Usage: bash tests/benchmark/run_benchmark.sh
#
# Droplet A (server): runs both userspace and kernel servers
# Droplet B (client): sends requests to A
set -euo pipefail

SSH_KEY="$HOME/.ssh/id_ed25519_surveyor"
SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o ConnectTimeout=10"

SERVER_IP="$1"  # Droplet A public IP
CLIENT_IP="$2"  # Droplet B public IP
BRANCH="${3:-dpdk-egress-integrity}"

RESULTS_DIR="tests/benchmark/results"
mkdir -p "$RESULTS_DIR"

sshA() { ssh $SSH_OPTS "root@$SERVER_IP" "$@"; }
sshB() { ssh $SSH_OPTS "root@$CLIENT_IP" "$@"; }
scpTo() { scp $SSH_OPTS -r "$1" "root@$2:$3"; }
scpFrom() { scp $SSH_OPTS "root@$1:$2" "$3"; }

echo "=== Deploying code to both droplets ==="

# Deploy to both in parallel
deploy_one() {
    local ip=$1
    local name=$2
    ssh $SSH_OPTS "root@$ip" bash <<'DEPLOY'
set -e
apt-get update -qq && apt-get install -y -qq python3 python3-pip git > /dev/null 2>&1 || true
rm -rf /root/stack
mkdir -p /root/stack
DEPLOY
    # Use git archive to ship just the branch content
    git archive "$BRANCH" | ssh $SSH_OPTS "root@$ip" "tar xf - -C /root/stack"
    echo "  Deployed to $name ($ip)"
}

deploy_one "$SERVER_IP" "server" &
deploy_one "$CLIENT_IP" "client" &
wait

echo
echo "=== Getting server network info ==="
SERVER_INFO=$(sshA "ip -4 addr show eth0 | grep inet | awk '{print \$2}' | cut -d/ -f1; ip link show eth0 | grep ether | awk '{print \$2}'; ip neigh show dev eth0 | grep router | head -1 | awk '{print \$3}'")
# Fallback: try to get gateway MAC from default route
SERVER_PRIVATE_IP=$(echo "$SERVER_INFO" | sed -n '1p')
SERVER_MAC=$(echo "$SERVER_INFO" | sed -n '2p')
SERVER_GW_MAC=$(sshA "ip neigh show dev eth0 | head -1 | awk '{for(i=1;i<=NF;i++) if(\$i==\"lladdr\") print \$(i+1)}'")

if [ -z "$SERVER_GW_MAC" ]; then
    # Ping gateway to populate ARP
    GW_IP=$(sshA "ip route show default dev eth0 | awk '{print \$3}'")
    sshA "ping -c1 -W1 $GW_IP > /dev/null 2>&1 || true"
    sleep 1
    SERVER_GW_MAC=$(sshA "ip neigh show dev eth0 | head -1 | awk '{for(i=1;i<=NF;i++) if(\$i==\"lladdr\") print \$(i+1)}'")
fi

echo "  Server IP: $SERVER_IP (private: $SERVER_PRIVATE_IP)"
echo "  Server MAC: $SERVER_MAC"
echo "  Gateway MAC: $SERVER_GW_MAC"

# ---- Kernel TCP benchmark ----
echo
echo "================================================================"
echo "  KERNEL TCP BENCHMARK"
echo "================================================================"

# Start kernel server on A
sshA "cd /root/stack && nohup python3 -m tests.benchmark.bench_userspace_vs_kernel server --mode kernel --port 9999 > /tmp/kernel_server.log 2>&1 &"
sleep 2

# Run client on B
sshB "cd /root/stack && python3 -m tests.benchmark.bench_userspace_vs_kernel client --server-ip $SERVER_IP --port 9999 --output /tmp/kernel_results.json" | tee "$RESULTS_DIR/kernel_output.txt"

# Stop kernel server
sshA "pkill -f 'bench_userspace_vs_kernel.*kernel' || true"
sleep 1

# Fetch results
scpFrom "$CLIENT_IP" "/tmp/kernel_results.json" "$RESULTS_DIR/kernel_results.json"

# ---- Userspace TCP benchmark ----
echo
echo "================================================================"
echo "  USERSPACE TCP BENCHMARK"
echo "================================================================"

# Start userspace server on A (needs root for AF_PACKET)
sshA "cd /root/stack && nohup python3 -m tests.benchmark.bench_userspace_vs_kernel server --mode userspace --port 9999 --ip $SERVER_IP --mac $SERVER_MAC --gw-mac $SERVER_GW_MAC > /tmp/userspace_server.log 2>&1 &"
sleep 3

# Run client on B
sshB "cd /root/stack && python3 -m tests.benchmark.bench_userspace_vs_kernel client --server-ip $SERVER_IP --port 9999 --output /tmp/userspace_results.json" | tee "$RESULTS_DIR/userspace_output.txt"

# Stop userspace server
sshA "pkill -f 'bench_userspace_vs_kernel.*userspace' || true"
sleep 1

# Fetch results
scpFrom "$CLIENT_IP" "/tmp/userspace_results.json" "$RESULTS_DIR/userspace_results.json"

# Fetch server logs
scpFrom "$SERVER_IP" "/tmp/kernel_server.log" "$RESULTS_DIR/kernel_server.log"
scpFrom "$SERVER_IP" "/tmp/userspace_server.log" "$RESULTS_DIR/userspace_server.log"

echo
echo "================================================================"
echo "  DONE - results in $RESULTS_DIR/"
echo "================================================================"
