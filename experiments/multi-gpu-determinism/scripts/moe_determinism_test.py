#!/usr/bin/env python3
"""Test MoE same-config determinism: run identical config twice, compare."""
import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_PROTO"] = "Simple"
os.environ["NCCL_DEBUG"] = "WARN"

MOE_DIR = Path("/tmp/d4-boi-results/moe")
manifest = str(MOE_DIR / "manifest_a.json")

# Run A2: second identical run
a2_dir = MOE_DIR / "a2"
a2_dir.mkdir(parents=True, exist_ok=True)

print("Resolving...")
subprocess.run(["python3", "cmd/resolver/main.py", "--manifest", manifest,
                 "--lockfile-out", str(a2_dir / "resolved.lock.json")], check=True)
print("Building...")
subprocess.run(["python3", "cmd/builder/main.py", "--lockfile", str(a2_dir / "resolved.lock.json"),
                 "--lockfile-out", str(a2_dir / "built.lock.json")], check=True)
print("Running vLLM inference (100 requests)...")
subprocess.run(["python3", "cmd/runner/main.py", "--manifest", manifest,
                 "--lockfile", str(a2_dir / "built.lock.json"),
                 "--out-dir", str(a2_dir / "run"), "--mode", "vllm", "--replica-id", "replica-0"], check=True)

# Compare A vs A2
ta = json.loads((MOE_DIR / "a/run/observables/tokens.json").read_text())
tb = json.loads((a2_dir / "run/observables/tokens.json").read_text())
a_by_id = {r["id"]: r["tokens"] for r in ta}
b_by_id = {r["id"]: r["tokens"] for r in tb}
matches = sum(1 for rid in a_by_id if a_by_id[rid] == b_by_id.get(rid, []))
total = len(a_by_id)
total_tokens = sum(len(r["tokens"]) for r in ta)
print(f"\nMoE same-config determinism: {matches}/{total} match ({total_tokens} total tokens)")
if matches < total:
    for rid in sorted(a_by_id):
        if a_by_id[rid] != b_by_id.get(rid, []):
            ta_r, tb_r = a_by_id[rid], b_by_id.get(rid, [])
            pos = next((i for i, (x, y) in enumerate(zip(ta_r, tb_r)) if x != y), -1)
            print(f"  {rid}: diverges at token {pos}")
            if pos >= 0:
                break
else:
    print("ALL MATCH — MoE TP determinism confirmed")
