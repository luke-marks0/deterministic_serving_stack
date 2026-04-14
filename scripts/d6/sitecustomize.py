"""sitecustomize hook that forces torch ProcessGroupGloo to bind to a specific IP.

Problem: on a dual-stack network interface (Linux eno1 with BOTH an RFC 1918
private address and a routable public address), torch's
`ProcessGroupGloo::createDeviceForInterface` → `gloo::lookupAddrForIface` walks
`getifaddrs()` and picks whatever address happens to come first. Inside Lambda
VMs this is the private IP, which is NOT reachable from peers in a different
Lambda region. The full-mesh gloo rendezvous then fails with
`SO_ERROR: Connection refused, remote=[<public>]:<port>` — the listener never
bound to the public IP it advertised.

Fix: swap `ProcessGroupGloo.createDefaultDevice` for one that calls
`createDeviceForHostname(<literal IP>)`, which bypasses getifaddrs entirely
and `bind()`s to the IP literal. Source the literal from VLLM_HOST_IP (which
we already set per-node via docker -e for vLLM's own driver-IP resolution).

This file must be on PYTHONPATH in every rank process (driver + all Ray
workers). Python auto-imports `sitecustomize` at interpreter startup, before
any user code, so by the time vLLM's `init_world_group` runs, the patch is
live. Propagating PYTHONPATH to Ray workers is automatic when the env var is
set on the ray start process (which inherits from the container's
`docker run -e` flags).

Silent on success, silent on expected failures (missing torch, missing
VLLM_HOST_IP). Errors are intentionally swallowed: a broken import hook
here would break every Python process on the node.
"""
from __future__ import annotations

import os
import sys


def _apply_gloo_bind_patch() -> None:
    host_ip = os.environ.get("VLLM_HOST_IP")
    if not host_ip:
        return

    try:
        from torch.distributed import ProcessGroupGloo  # type: ignore
    except Exception:
        return

    if not hasattr(ProcessGroupGloo, "createDeviceForHostname"):
        return

    try:
        def _make_device():  # noqa: ANN202
            return ProcessGroupGloo.createDeviceForHostname(host_ip)

        ProcessGroupGloo.createDefaultDevice = staticmethod(_make_device)
        print(
            f"[sitecustomize] patched ProcessGroupGloo.createDefaultDevice "
            f"to bind {host_ip}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as exc:
        print(
            f"[sitecustomize] failed to patch ProcessGroupGloo: {exc}",
            file=sys.stderr,
            flush=True,
        )


_apply_gloo_bind_patch()
