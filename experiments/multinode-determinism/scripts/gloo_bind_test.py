"""Two-rank torch.distributed init_process_group gloo test, single-machine.

Spawns 2 subprocess ranks, both init_process_group at world_size=2 gloo
backend, using the sitecustomize-patched create_default_device. Exits 0
if both ranks reach the post-init barrier, nonzero otherwise.

Usage:
  VLLM_HOST_IP=<local public IP> PYTHONPATH=/d6 \\
      python3 scripts/d6/gloo_bind_test.py <master_port>

The point of this test is to exercise the gloo ProcessGroup *listener bind*
and *connectFullMesh* path that vLLM's init_world_group hits. On a single
machine both ranks bind to the same IP (the patched VLLM_HOST_IP), so
connectFullMesh has to succeed if the bind/advertise addresses match.
If the patch is broken and gloo binds to a different IP than it advertises,
connectFullMesh fails with "SO_ERROR: Connection refused" — the same
Phase 3 symptom.
"""
from __future__ import annotations

import argparse
import os
import sys
import time


def rank_main(rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    import torch.distributed as dist  # noqa: E402

    print(f"[rank {rank}] calling init_process_group(gloo)", flush=True)
    t0 = time.time()
    dist.init_process_group(backend="gloo", timeout=__import__("datetime").timedelta(seconds=30))
    print(f"[rank {rank}] init_process_group ok in {time.time()-t0:.2f}s", flush=True)

    dist.barrier()
    print(f"[rank {rank}] barrier ok", flush=True)

    dist.destroy_process_group()
    print(f"[rank {rank}] destroyed", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("master_port", type=int)
    p.add_argument("--rank", type=int, default=None, help="internal: child worker rank")
    args = p.parse_args()

    master_addr = os.environ.get("VLLM_HOST_IP", "127.0.0.1")
    world_size = 2

    if args.rank is not None:
        rank_main(args.rank, world_size, master_addr, args.master_port)
        return 0

    import subprocess
    procs = []
    for r in range(world_size):
        procs.append(
            subprocess.Popen(
                [sys.executable, __file__, str(args.master_port), "--rank", str(r)],
                env=os.environ.copy(),
            )
        )
    rcs = [p.wait() for p in procs]
    print(f"[main] rank exit codes: {rcs}", flush=True)
    return 0 if all(rc == 0 for rc in rcs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
