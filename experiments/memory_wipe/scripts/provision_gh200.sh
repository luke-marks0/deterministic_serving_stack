#!/usr/bin/env bash
set -euo pipefail

# Poll Lambda Cloud for a GH200 instance and launch when capacity appears.
# Usage: ./scripts/provision_gh200.sh
# Requires: LAMBDALABS_API_KEY env var, jq

API="https://cloud.lambdalabs.com/api/v1"
AUTH="-u ${LAMBDALABS_API_KEY:?Set LAMBDALABS_API_KEY}:"
INSTANCE_TYPE="gpu_1x_gh200"
REGION="us-east-3"
SSH_KEY="macbook 2025"
NAME="pose-gh200"
POLL_INTERVAL=30

check_existing() {
    local instances
    instances=$(curl -s $AUTH "$API/instances")
    echo "$instances" | jq -e '.data[] | select(.instance_type.name == "'"$INSTANCE_TYPE"'")' 2>/dev/null && return 0
    return 1
}

echo "Checking for existing GH200 instance..."
if check_existing; then
    echo "GH200 instance already running:"
    curl -s $AUTH "$API/instances" | jq '.data[] | select(.instance_type.name == "'"$INSTANCE_TYPE"'") | {id, name, ip: .ip, status}'
    exit 0
fi

echo "No GH200 running. Polling for capacity every ${POLL_INTERVAL}s..."
attempt=0
while true; do
    attempt=$((attempt + 1))
    result=$(curl -s $AUTH -X POST "$API/instance-operations/launch" \
        -H 'Content-Type: application/json' \
        -d '{
            "region_name": "'"$REGION"'",
            "instance_type_name": "'"$INSTANCE_TYPE"'",
            "ssh_key_names": ["'"$SSH_KEY"'"],
            "name": "'"$NAME"'"
        }')

    if echo "$result" | jq -e '.data.instance_ids[0]' 2>/dev/null; then
        instance_id=$(echo "$result" | jq -r '.data.instance_ids[0]')
        echo ""
        echo "Launched! Instance ID: $instance_id"
        echo "Waiting for IP assignment..."

        while true; do
            info=$(curl -s $AUTH "$API/instances/$instance_id")
            ip=$(echo "$info" | jq -r '.data.ip // empty')
            status=$(echo "$info" | jq -r '.data.status // empty')
            if [ -n "$ip" ] && [ "$ip" != "null" ]; then
                echo "Instance ready: ssh ubuntu@$ip"
                echo "$ip" > /tmp/gh200_ip.txt
                exit 0
            fi
            echo "  Status: $status — waiting..."
            sleep 10
        done
    fi

    error=$(echo "$result" | jq -r '.error.code // empty')
    printf "\r[attempt %d] %s — retrying in %ds..." "$attempt" "$error" "$POLL_INTERVAL"
    sleep "$POLL_INTERVAL"
done
