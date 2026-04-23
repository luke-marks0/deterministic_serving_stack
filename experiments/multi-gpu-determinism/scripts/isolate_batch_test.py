#!/usr/bin/env python3
"""Isolate batch size vs order as the source of non-determinism.

Compares:
  A vs C: same order, batch=64 vs batch=16 (isolates batch size effect)
  A vs D: shuffled order, same batch=64 (isolates order effect)

Uses already-generated Run A from d4_batch_order_invariance.sh.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
os.environ["NCCL_ALGO"] = "Ring"
os.environ["NCCL_PROTO"] = "Simple"
os.environ["NCCL_DEBUG"] = "WARN"

DENSE_DIR = Path("/tmp/d4-boi-results/dense")


def run_cmd(args):
    print(f"  Running: {' '.join(args[:4])}...")
    subprocess.run(args, check=True, capture_output=True)


def run_pipeline(manifest_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["python3", "cmd/resolver/main.py", "--manifest", str(manifest_path),
             "--lockfile-out", str(out_dir / "resolved.lock.json")])
    run_cmd(["python3", "cmd/builder/main.py", "--lockfile", str(out_dir / "resolved.lock.json"),
             "--lockfile-out", str(out_dir / "built.lock.json")])
    run_cmd(["python3", "cmd/runner/main.py", "--manifest", str(manifest_path),
             "--lockfile", str(out_dir / "built.lock.json"),
             "--out-dir", str(out_dir / "run"), "--mode", "vllm", "--replica-id", "replica-0"])


def compare(label, dir_a, dir_b):
    tokens_a = json.loads((dir_a / "run/observables/tokens.json").read_text())
    tokens_b = json.loads((dir_b / "run/observables/tokens.json").read_text())
    a_by_id = {r["id"]: r["tokens"] for r in tokens_a}
    b_by_id = {r["id"]: r["tokens"] for r in tokens_b}
    matches = sum(1 for rid in a_by_id if a_by_id[rid] == b_by_id.get(rid, []))
    total = len(a_by_id)
    status = "PASS" if matches == total else "FAIL"
    print(f"  {label}: {matches}/{total} match [{status}]")
    if matches < total:
        for rid in sorted(a_by_id):
            ta, tb = a_by_id[rid], b_by_id.get(rid, [])
            if ta != tb:
                pos = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
                print(f"    {rid}: diverges at token {pos}")
                break  # just show first
    return matches, total


import random

# Load manifest A
manifest_a = json.loads((DENSE_DIR / "manifest_a.json").read_text())

# --- Test C: same order, different batch size ---
print("\n=== Test C: same order, batch=64 vs batch=16 ===")
manifest_c = json.loads(json.dumps(manifest_a))
manifest_c["run_id"] = "qwen2.5-32b-boi-ordered-batch16"
manifest_c["runtime"]["serving_engine"]["max_num_seqs"] = 16
c_path = DENSE_DIR / "manifest_c.json"
c_path.write_text(json.dumps(manifest_c, sort_keys=True, separators=(",", ":")) + "\n")
c_dir = DENSE_DIR / "c"
run_pipeline(c_path, c_dir)
compare("Same order, batch 64→16", DENSE_DIR / "a", c_dir)

# --- Test D: shuffled order, same batch size ---
print("\n=== Test D: shuffled order, same batch=64 ===")
rng = random.Random(12345)
shuffled_reqs = list(manifest_a["requests"])
rng.shuffle(shuffled_reqs)
manifest_d = json.loads(json.dumps(manifest_a))
manifest_d["run_id"] = "qwen2.5-32b-boi-shuffled-batch64"
manifest_d["requests"] = shuffled_reqs
# Keep batch=64 (same as A)
d_path = DENSE_DIR / "manifest_d.json"
d_path.write_text(json.dumps(manifest_d, sort_keys=True, separators=(",", ":")) + "\n")
d_dir = DENSE_DIR / "d"
run_pipeline(d_path, d_dir)
compare("Shuffled order, same batch=64", DENSE_DIR / "a", d_dir)

print("\nDone.")
