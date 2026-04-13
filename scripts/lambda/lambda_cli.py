#!/usr/bin/env python3
"""Lambda Cloud helper for the D6 rollout.

One place for every Lambda API interaction we need. DRY.

Subcommands:
    list                 List running instances (id, type, ip, status).
    keys                 List registered SSH keys.
    add-key NAME PUBKEY  Register an SSH public key.
    types                List instance types and current capacity.
    poll TYPE [opts]     Poll for capacity, launch when available.
    terminate ID         Terminate an instance.
    terminate-all        Terminate every running instance (with confirmation).

Auth: reads LAMBDALABS_API_KEY from env. No fallback; we don't want surprises.

Run with:  python3 scripts/lambda/lambda_cli.py <subcommand> [args...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from base64 import b64encode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_BASE = "https://cloud.lambdalabs.com/api/v1"


def _auth_header() -> dict[str, str]:
    key = os.environ.get("LAMBDALABS_API_KEY")
    if not key:
        sys.exit("LAMBDALABS_API_KEY is not set")
    creds = b64encode(f"{key}:".encode()).decode()
    return {
        "Authorization": f"Basic {creds}",
        "Content-Type": "application/json",
        "User-Agent": "d6-lambda-cli/1.0",
    }


def api(method: str, path: str, body: dict | None = None) -> dict:
    """Single chokepoint for every Lambda API call."""
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = Request(url, data=data, method=method, headers=_auth_header())
    try:
        with urlopen(req) as resp:
            return json.loads(resp.read())
    except HTTPError as e:
        sys.exit(f"{method} {path} -> HTTP {e.code}: {e.read().decode()[:500]}")
    except URLError as e:
        sys.exit(f"{method} {path} -> network error: {e}")


# ───────────────────────── subcommands ─────────────────────────

def cmd_list(_args: argparse.Namespace) -> None:
    data = api("GET", "/instances")["data"]
    if not data:
        print("(no instances)")
        return
    for d in data:
        print(
            f"{d['id']:24s} "
            f"{d['instance_type']['name']:24s} "
            f"{d.get('ip', '-'):16s} "
            f"{d.get('status', '?')}"
        )


def cmd_keys(_args: argparse.Namespace) -> None:
    for k in api("GET", "/ssh-keys")["data"]:
        print(f"{k['id']}\t{k['name']}")


def cmd_add_key(args: argparse.Namespace) -> None:
    pubkey = open(args.pubkey_file).read().strip()
    out = api("POST", "/ssh-keys", {"name": args.name, "public_key": pubkey})
    print(json.dumps(out, indent=2))


def cmd_types(_args: argparse.Namespace) -> None:
    data = api("GET", "/instance-types")["data"]
    for name in sorted(data.keys()):
        info = data[name]
        regions = [r["name"] for r in info["regions_with_capacity_available"]]
        price = info["instance_type"]["price_cents_per_hour"] / 100
        gpus = info["instance_type"]["specs"]["gpus"]
        avail = ",".join(regions) if regions else "-"
        print(f"{name:30s} ${price:6.2f}/hr  gpus={gpus}  available_in: {avail}")


def cmd_terminate(args: argparse.Namespace) -> None:
    out = api(
        "POST",
        "/instance-operations/terminate",
        {"instance_ids": [args.instance_id]},
    )
    print(json.dumps(out, indent=2))


def cmd_terminate_all(_args: argparse.Namespace) -> None:
    data = api("GET", "/instances")["data"]
    if not data:
        print("(nothing to terminate)")
        return
    ids = [d["id"] for d in data]
    print("about to terminate:")
    for d in data:
        print(f"  {d['id']}  {d['instance_type']['name']}  {d.get('ip','-')}")
    if input("type 'yes' to confirm: ").strip() != "yes":
        sys.exit("aborted")
    out = api(
        "POST",
        "/instance-operations/terminate",
        {"instance_ids": ids},
    )
    print(json.dumps(out, indent=2))


def cmd_poll(args: argparse.Namespace) -> None:
    """Poll for capacity, launch one at a time."""
    target = args.count
    interval = args.interval
    region_filter = args.region
    instance_type = args.type
    ssh_key = args.ssh_key
    name_prefix = args.name_prefix

    print(
        f"polling for {target}× {instance_type} "
        f"(interval={interval}s, region={region_filter or 'any'})"
    )
    launched: list[str] = []
    iteration = 0
    while len(launched) < target:
        iteration += 1
        ts = time.strftime("%H:%M:%S")
        types = api("GET", "/instance-types")["data"]
        if instance_type not in types:
            sys.exit(f"unknown instance type: {instance_type}")
        regions = [
            r["name"]
            for r in types[instance_type]["regions_with_capacity_available"]
        ]
        if region_filter:
            regions = [r for r in regions if r == region_filter]
        if not regions:
            print(f"  [{ts}] iter {iteration}: no capacity ({len(launched)}/{target} launched)")
            time.sleep(interval)
            continue

        region = regions[0]
        body = {
            "region_name": region,
            "instance_type_name": instance_type,
            "ssh_key_names": [ssh_key],
            "name": f"{name_prefix}-{len(launched) + 1}",
        }
        try:
            out = api("POST", "/instance-operations/launch", body)
        except SystemExit as e:
            # Capacity can vanish between the check and the launch. Try again.
            print(f"  [{ts}] launch failed in {region}: {e}")
            time.sleep(interval)
            continue
        ids = out["data"]["instance_ids"]
        launched.extend(ids)
        print(f"  [{ts}] launched {ids} in {region}  ({len(launched)}/{target})")

    print(f"all {target} launched: {launched}")


# ───────────────────────── argument parser ─────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(func=cmd_list)
    sub.add_parser("keys").set_defaults(func=cmd_keys)
    sub.add_parser("types").set_defaults(func=cmd_types)

    p_addkey = sub.add_parser("add-key")
    p_addkey.add_argument("name")
    p_addkey.add_argument("pubkey_file")
    p_addkey.set_defaults(func=cmd_add_key)

    p_term = sub.add_parser("terminate")
    p_term.add_argument("instance_id")
    p_term.set_defaults(func=cmd_terminate)

    sub.add_parser("terminate-all").set_defaults(func=cmd_terminate_all)

    p_poll = sub.add_parser("poll")
    p_poll.add_argument("type", help="e.g. gpu_1x_h100_sxm5")
    p_poll.add_argument("--count", type=int, default=1)
    p_poll.add_argument("--region", help="restrict to one region")
    p_poll.add_argument("--interval", type=int, default=30)
    p_poll.add_argument("--ssh-key", default="d6-rollout")
    p_poll.add_argument("--name-prefix", default="d6-node")
    p_poll.set_defaults(func=cmd_poll)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
