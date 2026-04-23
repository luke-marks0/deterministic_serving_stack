#!/usr/bin/env python3
"""Generate Run A and Run B manifest variants for the 1M-token determinism experiment.

Usage: gen_workload.py <base_manifest.json> <out_dir> <tag>

Produces:
  <out_dir>/manifest_a.json  (1000 reqs, original order, max_num_seqs=64)
  <out_dir>/manifest_b.json  (same 1000 reqs, shuffled, max_num_seqs=16)

Both manifests share the same request set; only order and batch size differ.
Per-request bitwise token equality across A and B proves batch + order invariance.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

TEMPLATES = [
    "Explain in detail how {} works, covering the underlying mechanisms, key tradeoffs, and common pitfalls:",
    "Write a comprehensive technical overview of {}, including its history, design rationale, and modern variants:",
    "Describe the historical evolution and current state of {}, with emphasis on the key turning points:",
    "Compare and contrast {} with at least three serious alternatives, explaining the practical tradeoffs:",
    "What are the deepest open challenges in {}? Discuss why each is hard and what approaches show promise:",
    "Walk through {} step by step as if teaching it to a strong CS undergraduate, with concrete examples:",
    "Give a critical analysis of {}: what works, what doesn't, and what most practitioners get wrong:",
    "Trace the engineering decisions behind {} and explain what would change if we redesigned it today:",
]

SUBJECTS = [
    "TCP congestion control", "the BGP routing protocol", "DNS resolution and caching",
    "TLS 1.3 handshake design", "QUIC and HTTP/3", "the HTTP/2 multiplexing model",
    "RSA encryption and its weaknesses", "elliptic curve cryptography",
    "post-quantum lattice-based cryptography", "zero-knowledge proofs",
    "Merkle trees in distributed systems", "Paxos consensus", "Raft consensus",
    "the CAP theorem in practice", "vector clocks and CRDTs",
    "two-phase commit and its alternatives", "MapReduce and its descendants",
    "Apache Kafka's storage model", "log-structured merge trees",
    "B-tree vs LSM-tree database indexes", "PostgreSQL's MVCC",
    "InnoDB undo and redo logs", "Linux kernel scheduling",
    "Linux virtual memory and page faults", "Linux io_uring",
    "Linux cgroups and namespaces", "container image layers",
    "the eBPF virtual machine", "garbage collection in the JVM",
    "Go's concurrent garbage collector", "Rust's borrow checker",
    "the LLVM optimizer", "compiler register allocation",
    "SSA form and dataflow analysis", "loop vectorization in compilers",
    "GPU shader programming", "CUDA memory hierarchy",
    "tensor parallelism in large language models", "pipeline parallelism in neural networks",
    "the Transformer attention mechanism", "FlashAttention and its variants",
    "speculative decoding for LLM inference", "KV cache management in vLLM",
    "mixture-of-experts routing", "RLHF and its critiques",
    "the WebAssembly virtual machine", "JIT compilation in V8",
    "binary search trees and balancing strategies", "skip lists",
    "consistent hashing and its applications", "Bloom filters",
    "RAID levels and failure modes", "ZFS data integrity guarantees",
    "ext4 journaling", "filesystems on flash storage",
    "the X.509 certificate ecosystem", "OAuth 2.0 and OpenID Connect",
    "cross-site request forgery defenses", "SQL injection prevention",
    "the Spectre and Meltdown vulnerabilities",
]


def make_requests(seed: int, count: int) -> list[dict]:
    rng = random.Random(seed)
    # Length distribution sums to ~1.04M tokens over 1000 reqs
    length_pool = (
        [256] * 150
        + [512] * 250
        + [1024] * 350
        + [2048] * 250
    )
    rng.shuffle(length_pool)
    assert len(length_pool) >= count, f"need {count}, have {len(length_pool)}"

    reqs = []
    for i in range(count):
        template = TEMPLATES[i % len(TEMPLATES)]
        subject = SUBJECTS[i % len(SUBJECTS)]
        salt = i // (len(TEMPLATES) * len(SUBJECTS))
        prompt = template.format(subject)
        if salt:
            prompt = f"[v{salt}] " + prompt
        reqs.append({
            "id": f"req-{i:04d}",
            "prompt": prompt,
            "max_new_tokens": length_pool[i],
            "temperature": 0,
        })
    return reqs


def main() -> None:
    if len(sys.argv) != 4:
        sys.exit("usage: gen_workload.py <base.json> <out_dir> <tag>")
    base_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    tag = sys.argv[3]

    base = json.loads(base_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    reqs = make_requests(seed=20260414, count=1000)
    total_max = sum(r["max_new_tokens"] for r in reqs)
    print(f"[{tag}] generated {len(reqs)} requests, max_new_tokens sum = {total_max}")

    # Run A: original order, batch=64
    a = json.loads(json.dumps(base))
    a["run_id"] = f"{tag}-1m-A-ordered"
    a["requests"] = reqs
    a["runtime"]["serving_engine"]["max_num_seqs"] = 64
    (out_dir / "manifest_a.json").write_text(
        json.dumps(a, sort_keys=True, separators=(",", ":")) + "\n"
    )

    # Run B: shuffled, batch=16
    rng_b = random.Random(12345)
    shuffled = list(reqs)
    rng_b.shuffle(shuffled)
    b = json.loads(json.dumps(base))
    b["run_id"] = f"{tag}-1m-B-shuffled"
    b["requests"] = shuffled
    b["runtime"]["serving_engine"]["max_num_seqs"] = 16
    (out_dir / "manifest_b.json").write_text(
        json.dumps(b, sort_keys=True, separators=(",", ":")) + "\n"
    )

    print(f"[{tag}] wrote {out_dir/'manifest_a.json'} and {out_dir/'manifest_b.json'}")


if __name__ == "__main__":
    main()
