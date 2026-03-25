#!/usr/bin/env python3
"""
Million-Token Determinism Experiment
-------------------------------------

Two test modes per model:
  repeated: same prompt × 20 runs (tests KV cache / memory determinism)
  diverse:  34 unique prompts × 1 run each (tests across input distributions)

Run one set on each server, compare S1 vs S2.

Usage:
  # Run a set on a server:
  python3 experiments/million_token_determinism.py \
      --model Qwen/Qwen3-8B \
      --mode repeated \
      --server-id s1 \
      --out-dir /tmp/determinism-experiment

  python3 experiments/million_token_determinism.py \
      --model Qwen/Qwen3-8B \
      --mode diverse \
      --server-id s1 \
      --out-dir /tmp/determinism-experiment

  # Compare two servers:
  python3 experiments/million_token_determinism.py \
      --compare /tmp/results/s1/repeated /tmp/results/s2/repeated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


TOKENS_PER_REQUEST = 30_000
SERVER_URL = "http://localhost:8000/v1/chat/completions"

REPEATED_PROMPT = "Tell me a very long, detailed story with many characters, plot twists, and dialogue. Do not stop writing."

DIVERSE_PROMPTS = [
    "Tell me a very long, detailed story with many characters, plot twists, and dialogue. Do not stop writing.",
    "Write a comprehensive textbook chapter on the history of mathematics from ancient civilizations to modern day.",
    "Write a detailed technical manual for building a space station, covering engineering, life support, and power systems.",
    "Write an epic fantasy novel with world-building, magic systems, political intrigue, and multiple POV characters.",
    "Write a thorough analysis of every major economic theory from mercantilism to modern monetary theory with examples.",
    "Write a complete guide to quantum mechanics including mathematical derivations and experimental evidence.",
    "Write a detailed history of computing from Babbage to modern AI, covering hardware, software, and social impact.",
    "Write an extensive cookbook with 100 recipes from around the world, including techniques and cultural context.",
    "Write a comprehensive guide to marine biology covering every major phylum and ecosystem interaction.",
    "Write a detailed legal textbook covering constitutional law, criminal law, and civil procedure with case studies.",
    "Write an exhaustive guide to music theory, composition, orchestration, and the history of musical traditions.",
    "Write a thorough exploration of philosophy of mind, consciousness, free will, and personal identity.",
    "Write a complete reference manual for organic chemistry covering reaction mechanisms and synthesis strategies.",
    "Write an extensive treatise on architecture from ancient temples to modern skyscrapers.",
    "Write a comprehensive analysis of geopolitics covering every major region and alliance structure.",
    "Write a detailed guide to neuroscience covering neural circuits, brain imaging, and neuroplasticity.",
    "Write an exhaustive history of warfare from ancient battles to modern cyber warfare.",
    "Write a thorough guide to astrophysics covering stellar evolution, galactic dynamics, and cosmology.",
    "Write a complete textbook on molecular biology covering DNA replication and gene expression.",
    "Write a comprehensive exploration of world religions, their histories, philosophies, and rituals.",
    "Write a detailed analysis of climate science covering atmospheric physics and ocean circulation.",
    "Write an extensive guide to machine learning covering supervised, unsupervised, and reinforcement learning.",
    "Write a thorough history of art from cave paintings to digital art, covering movements and techniques.",
    "Write a complete reference on pharmacology covering drug mechanisms and clinical trials.",
    "Write a comprehensive guide to linguistics covering phonology, morphology, syntax, and semantics.",
    "Write a detailed exploration of number theory including prime distribution and elliptic curves.",
    "Write an exhaustive guide to ecology covering population dynamics and biodiversity conservation.",
    "Write a thorough analysis of behavioral economics covering cognitive biases and prospect theory.",
    "Write a complete guide to materials science covering crystallography, polymers, and semiconductors.",
    "Write a comprehensive history of exploration from Polynesian navigation to Mars rovers.",
    "Write a detailed guide to epidemiology covering study design, biostatistics, and disease modeling.",
    "Write an extensive treatise on ethics covering virtue ethics, deontology, and consequentialism.",
    "Write a thorough guide to robotics covering kinematics, control theory, and computer vision.",
    "Write a complete analysis of the global financial system covering central banking and derivatives.",
]


def generate_chunk(model: str, prompt: str, chunk_idx: int) -> dict:
    """Generate one 30K token chunk and return metadata."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": TOKENS_PER_REQUEST,
        "min_tokens": TOKENS_PER_REQUEST,
        "temperature": 0.0,
        "seed": 42,
    }).encode()

    req = urllib.request.Request(
        SERVER_URL,
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=1800)
    elapsed = time.time() - t0
    data = json.loads(resp.read())

    content = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["completion_tokens"]
    finish = data["choices"][0].get("finish_reason", "unknown")
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    return {
        "content": content,
        "tokens": tokens,
        "elapsed": round(elapsed, 1),
        "finish_reason": finish,
        "hash": content_hash,
        "chunk_idx": chunk_idx,
    }


def run_repeated(model: str, server_id: str, out_dir: Path, n_runs: int = 20) -> None:
    """Same prompt × N runs."""
    run_dir = out_dir / server_id / "repeated"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  === REPEATED MODE ===")
    print(f"  Model: {model}")
    print(f"  Server: {server_id}")
    print(f"  Prompt: \"{REPEATED_PROMPT[:60]}...\"")
    print(f"  Runs: {n_runs} × {TOKENS_PER_REQUEST:,} tokens")
    print()

    # Resume from existing progress
    rolling_hash = hashlib.sha256()
    total_tokens = 0
    chunks = []
    start_idx = 0

    chunks_file = run_dir / "chunks.json"
    if chunks_file.exists():
        chunks = json.loads(chunks_file.read_text())
        start_idx = len(chunks)
        total_tokens = sum(c["tokens"] for c in chunks)
        # Rebuild rolling hash from saved chunks
        for c in chunks:
            content = (run_dir / f"chunk_{c['chunk_idx']:04d}.txt").read_text(encoding="utf-8")
            rolling_hash.update(content.encode())
        if start_idx > 0:
            print(f"  Resuming from chunk {start_idx}/{n_runs} ({total_tokens:,} tokens already done)")
            print()

    if start_idx >= n_runs:
        print(f"  Already complete ({n_runs}/{n_runs} runs)")
        return

    for i in range(start_idx, n_runs):
        result = generate_chunk(model, REPEATED_PROMPT, i)

        rolling_hash.update(result["content"].encode())
        total_tokens += result["tokens"]

        # Save content
        (run_dir / f"chunk_{i:04d}.txt").write_text(result["content"], encoding="utf-8")

        match = "MATCH" if i == 0 or result["hash"] == chunks[0]["hash"] else "MISMATCH"
        tps = result["tokens"] / result["elapsed"] if result["elapsed"] > 0 else 0
        print(
            f"  Run {i+1:3d}/{n_runs}: {result['tokens']:5d} tok  {result['elapsed']:6.1f}s  "
            f"{tps:5.0f} tok/s  total={total_tokens:>9,}  "
            f"sha256:{result['hash'][:12]}...  {match}",
            flush=True,
        )

        chunks.append({
            "chunk_idx": i,
            "tokens": result["tokens"],
            "elapsed": result["elapsed"],
            "finish_reason": result["finish_reason"],
            "hash": result["hash"],
            "rolling_hash": rolling_hash.hexdigest(),
        })

        # Save progress after each chunk
        (run_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    meta = {
        "mode": "repeated",
        "model": model,
        "server_id": server_id,
        "hostname": os.uname().nodename,
        "n_runs": n_runs,
        "total_tokens": total_tokens,
        "final_rolling_hash": rolling_hash.hexdigest(),
        "all_identical": len(set(c["hash"] for c in chunks)) == 1,
        "started_at": chunks[0].get("started_at", ""),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"\n  Total: {total_tokens:,} tokens")
    print(f"  All identical: {meta['all_identical']}")
    print(f"  Rolling hash: sha256:{rolling_hash.hexdigest()}")


def run_diverse(model: str, server_id: str, out_dir: Path) -> None:
    """34 unique prompts × 1 run each."""
    run_dir = out_dir / server_id / "diverse"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  === DIVERSE MODE ===")
    print(f"  Model: {model}")
    print(f"  Server: {server_id}")
    print(f"  Prompts: {len(DIVERSE_PROMPTS)} unique")
    print(f"  Tokens per prompt: {TOKENS_PER_REQUEST:,}")
    print()

    # Resume from existing progress
    rolling_hash = hashlib.sha256()
    total_tokens = 0
    chunks = []
    start_idx = 0

    chunks_file = run_dir / "chunks.json"
    if chunks_file.exists():
        chunks = json.loads(chunks_file.read_text())
        start_idx = len(chunks)
        total_tokens = sum(c["tokens"] for c in chunks)
        for c in chunks:
            content = (run_dir / f"chunk_{c['chunk_idx']:04d}.txt").read_text(encoding="utf-8")
            rolling_hash.update(content.encode())
        if start_idx > 0:
            print(f"  Resuming from prompt {start_idx}/{len(DIVERSE_PROMPTS)} ({total_tokens:,} tokens already done)")
            print()

    if start_idx >= len(DIVERSE_PROMPTS):
        print(f"  Already complete ({len(DIVERSE_PROMPTS)}/{len(DIVERSE_PROMPTS)} prompts)")
        return

    for i, prompt in enumerate(DIVERSE_PROMPTS):
        if i < start_idx:
            continue
        result = generate_chunk(model, prompt, i)

        rolling_hash.update(result["content"].encode())
        total_tokens += result["tokens"]

        (run_dir / f"chunk_{i:04d}.txt").write_text(result["content"], encoding="utf-8")

        tps = result["tokens"] / result["elapsed"] if result["elapsed"] > 0 else 0
        print(
            f"  Prompt {i+1:3d}/{len(DIVERSE_PROMPTS)}: {result['tokens']:5d} tok  "
            f"{result['elapsed']:6.1f}s  {tps:5.0f} tok/s  "
            f"total={total_tokens:>9,}  sha256:{result['hash'][:12]}...  "
            f"\"{prompt[:40]}...\"",
            flush=True,
        )

        chunks.append({
            "chunk_idx": i,
            "prompt": prompt,
            "tokens": result["tokens"],
            "elapsed": result["elapsed"],
            "finish_reason": result["finish_reason"],
            "hash": result["hash"],
            "rolling_hash": rolling_hash.hexdigest(),
        })

        # Save progress after each chunk
        (run_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    meta = {
        "mode": "diverse",
        "model": model,
        "server_id": server_id,
        "hostname": os.uname().nodename,
        "n_prompts": len(DIVERSE_PROMPTS),
        "total_tokens": total_tokens,
        "final_rolling_hash": rolling_hash.hexdigest(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (run_dir / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"\n  Total: {total_tokens:,} tokens")
    print(f"  Rolling hash: sha256:{rolling_hash.hexdigest()}")


def compare_runs(dir_a: Path, dir_b: Path) -> None:
    """Compare two run directories chunk-by-chunk."""
    meta_a = json.loads((dir_a / "meta.json").read_text())
    meta_b = json.loads((dir_b / "meta.json").read_text())
    chunks_a = json.loads((dir_a / "chunks.json").read_text())
    chunks_b = json.loads((dir_b / "chunks.json").read_text())

    mode = meta_a.get("mode", "unknown")
    print(f"\n  === COMPARISON ({mode}) ===")
    print(f"  A: {meta_a.get('server_id','?')} on {meta_a.get('hostname','?')} ({meta_a['total_tokens']:,} tokens)")
    print(f"  B: {meta_b.get('server_id','?')} on {meta_b.get('hostname','?')} ({meta_b['total_tokens']:,} tokens)")
    print()

    min_chunks = min(len(chunks_a), len(chunks_b))
    mismatches = 0
    first_divergence = None

    for i in range(min_chunks):
        a, b = chunks_a[i], chunks_b[i]
        match = a["hash"] == b["hash"]
        if not match:
            mismatches += 1
            if first_divergence is None:
                first_divergence = i
            status = "MISMATCH"
        else:
            status = "MATCH   "

        label = f"Prompt {i+1}" if mode == "diverse" else f"Run {i+1}"
        print(
            f"  {label:12s}: {status}  "
            f"A=sha256:{a['hash'][:12]}...  "
            f"B=sha256:{b['hash'][:12]}...",
            flush=True,
        )

    print(f"\n  Chunks compared: {min_chunks}")
    print(f"  Matching: {min_chunks - mismatches}/{min_chunks}")
    print(f"  Mismatches: {mismatches}")
    if first_divergence is not None:
        print(f"  First divergence: chunk {first_divergence}")
    print(f"  Rolling hash A: sha256:{meta_a['final_rolling_hash']}")
    print(f"  Rolling hash B: sha256:{meta_b['final_rolling_hash']}")
    if meta_a["final_rolling_hash"] == meta_b["final_rolling_hash"]:
        print(f"  VERDICT: IDENTICAL")
    else:
        print(f"  VERDICT: DIVERGENT")


def main() -> None:
    parser = argparse.ArgumentParser(description="Million-token determinism experiment")
    parser.add_argument("--model", help="Model to test")
    parser.add_argument("--mode", choices=["repeated", "diverse"], help="Test mode")
    parser.add_argument("--server-id", help="Server identifier (e.g. s1, s2)")
    parser.add_argument("--out-dir", help="Output directory")
    parser.add_argument("--n-runs", type=int, default=20, help="Number of repeated runs")
    parser.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"),
                        help="Compare two run directories")
    args = parser.parse_args()

    if args.compare:
        compare_runs(Path(args.compare[0]), Path(args.compare[1]))
    elif args.model and args.mode and args.server_id and args.out_dir:
        model_short = args.model.split("/")[-1]
        base = Path(args.out_dir) / model_short
        if args.mode == "repeated":
            run_repeated(args.model, args.server_id, base, n_runs=args.n_runs)
        else:
            run_diverse(args.model, args.server_id, base)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
