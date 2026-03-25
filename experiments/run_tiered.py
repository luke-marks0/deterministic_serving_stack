#!/usr/bin/env python3
"""
Tiered Determinism Experiment
------------------------------

Three stages that build on each other (results are cumulative and resumable):

  smoke  (~10-15 min)  2 repeated × 2K tok + 3 diverse × 2K tok, per model
  medium (~3 hours)    10 repeated × 10K tok + 10 diverse × 10K tok, per model
  full   (~12+ hours)  20 repeated × 30K tok + 34 diverse × 30K tok, per model

Each tier picks up where the previous left off. Results live in:
  <out-dir>/<ModelName>/<server-id>/{repeated,diverse}/

Progress is written to:
  <out-dir>/progress.json       (machine-readable, updated after every chunk)
  <out-dir>/progress.txt        (human-readable summary)

Usage:
  # Smoke test on one server:
  python3 experiments/run_tiered.py --tier smoke --server-id s1

  # Continue to medium (reuses smoke results):
  python3 experiments/run_tiered.py --tier medium --server-id s1

  # Full experiment:
  python3 experiments/run_tiered.py --tier full --server-id s1

  # Check progress from anywhere:
  cat /tmp/determinism-experiment/progress.txt

  # Compare two servers:
  python3 experiments/run_tiered.py --compare s1 s2 --out-dir /tmp/determinism-experiment
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER_URL = "http://localhost:8000/v1/chat/completions"

MODELS = [
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-30B-A3B",
]

TIERS = {
    "smoke": {
        "n_repeated": 2,
        "n_diverse": 3,
        "tokens_per_request": 2_000,
        "description": "Quick sanity check (~10-15 min)",
    },
    "medium": {
        "n_repeated": 10,
        "n_diverse": 10,
        "tokens_per_request": 10_000,
        "description": "Thorough test (~3 hours)",
    },
    "full": {
        "n_repeated": 20,
        "n_diverse": 34,
        "tokens_per_request": 30_000,
        "description": "Full million-token experiment (~12+ hours)",
    },
}

REPEATED_PROMPT = (
    "Tell me a very long, detailed story with many characters, "
    "plot twists, and dialogue. Do not stop writing."
)

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


# ── Progress tracking ──────────────────────────────────────────────────────

class Progress:
    """Writes machine-readable progress.json and human-readable progress.txt."""

    def __init__(self, out_dir: Path, server_id: str, tier: str):
        self.out_dir = out_dir
        self.server_id = server_id
        self.tier = tier
        self.json_path = out_dir / "progress.json"
        self.txt_path = out_dir / "progress.txt"
        self.state: dict = self._load()

    def _load(self) -> dict:
        if self.json_path.exists():
            return json.loads(self.json_path.read_text())
        return {
            "server_id": self.server_id,
            "tier": self.tier,
            "started_at": _now(),
            "models": {},
        }

    def update(
        self,
        model: str,
        mode: str,
        done: int,
        total: int,
        tokens: int,
        last_hash: str = "",
        status: str = "running",
    ):
        model_short = model.split("/")[-1]
        key = f"{model_short}/{mode}"
        self.state.setdefault("models", {})[key] = {
            "done": done,
            "total": total,
            "tokens": tokens,
            "last_hash": last_hash,
            "status": status,
            "updated_at": _now(),
        }
        self.state["updated_at"] = _now()
        self._write()

    def mark_complete(self, model: str, mode: str):
        model_short = model.split("/")[-1]
        key = f"{model_short}/{mode}"
        if key in self.state.get("models", {}):
            self.state["models"][key]["status"] = "complete"
        self._write()

    def mark_error(self, model: str, mode: str, error: str):
        model_short = model.split("/")[-1]
        key = f"{model_short}/{mode}"
        self.state.setdefault("models", {})[key] = {
            **self.state.get("models", {}).get(key, {}),
            "status": f"error: {error}",
            "updated_at": _now(),
        }
        self._write()

    def _write(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps(self.state, indent=2))

        # Human-readable summary
        lines = [
            f"Determinism Experiment — {self.tier} tier",
            f"Server: {self.server_id}",
            f"Updated: {self.state.get('updated_at', '?')}",
            "",
        ]
        total_tokens = 0
        for key, info in self.state.get("models", {}).items():
            pct = (
                f"{info['done']}/{info['total']}"
                if info["total"] > 0
                else "?"
            )
            tok = info.get("tokens", 0)
            total_tokens += tok
            status = info.get("status", "?")
            lines.append(f"  {key:40s}  {pct:>8s}  {tok:>10,} tok  {status}")
        lines.append("")
        lines.append(f"  Total tokens: {total_tokens:,}")
        self.txt_path.write_text("\n".join(lines) + "\n")


# ── Generation ─────────────────────────────────────────────────────────────

def generate_chunk(model: str, prompt: str, max_tokens: int, chunk_idx: int) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 42,
    }).encode()

    req = urllib.request.Request(
        SERVER_URL, data=body, headers={"Content-Type": "application/json"},
    )

    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=3600)
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


def run_mode(
    model: str,
    mode: str,
    server_id: str,
    out_dir: Path,
    n_chunks: int,
    tokens_per_request: int,
    progress: Progress,
) -> None:
    model_short = model.split("/")[-1]
    run_dir = out_dir / model_short / server_id / mode
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = (
        [REPEATED_PROMPT] * n_chunks
        if mode == "repeated"
        else DIVERSE_PROMPTS[:n_chunks]
    )

    label = "Run" if mode == "repeated" else "Prompt"

    # Resume
    chunks_file = run_dir / "chunks.json"
    chunks: list[dict] = []
    rolling_hash = hashlib.sha256()
    total_tokens = 0
    start_idx = 0

    if chunks_file.exists():
        chunks = json.loads(chunks_file.read_text())
        start_idx = len(chunks)
        total_tokens = sum(c["tokens"] for c in chunks)
        for c in chunks:
            txt = (run_dir / f"chunk_{c['chunk_idx']:04d}.txt").read_text(encoding="utf-8")
            rolling_hash.update(txt.encode())

    if start_idx >= n_chunks:
        print(f"    Already complete ({n_chunks}/{n_chunks})")
        progress.update(model, mode, n_chunks, n_chunks, total_tokens, status="complete")
        progress.mark_complete(model, mode)
        return

    if start_idx > 0:
        print(f"    Resuming from {label.lower()} {start_idx + 1}/{n_chunks} ({total_tokens:,} tokens done)")

    progress.update(model, mode, start_idx, n_chunks, total_tokens, status="running")

    for i in range(start_idx, n_chunks):
        prompt = prompts[i]
        result = generate_chunk(model, prompt, tokens_per_request, i)

        rolling_hash.update(result["content"].encode())
        total_tokens += result["tokens"]

        (run_dir / f"chunk_{i:04d}.txt").write_text(result["content"], encoding="utf-8")

        # Match check for repeated mode
        suffix = ""
        if mode == "repeated" and i > 0:
            suffix = " MATCH" if result["hash"] == chunks[0]["hash"] else " MISMATCH"

        tps = result["tokens"] / result["elapsed"] if result["elapsed"] > 0 else 0
        print(
            f"    {label} {i + 1:3d}/{n_chunks}: "
            f"{result['tokens']:5d} tok  {result['elapsed']:6.1f}s  {tps:5.0f} tok/s  "
            f"sha256:{result['hash'][:12]}...{suffix}",
            flush=True,
        )

        chunks.append({
            "chunk_idx": i,
            "tokens": result["tokens"],
            "elapsed": result["elapsed"],
            "finish_reason": result["finish_reason"],
            "hash": result["hash"],
            "rolling_hash": rolling_hash.hexdigest(),
            **({"prompt": prompt} if mode == "diverse" else {}),
        })

        chunks_file.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
        progress.update(model, mode, i + 1, n_chunks, total_tokens, result["hash"])

    # Write meta
    meta = {
        "mode": mode,
        "model": model,
        "server_id": server_id,
        "hostname": os.uname().nodename,
        "n_chunks": n_chunks,
        "tokens_per_request": tokens_per_request,
        "total_tokens": total_tokens,
        "final_rolling_hash": rolling_hash.hexdigest(),
        "all_identical": len(set(c["hash"] for c in chunks)) == 1 if mode == "repeated" else None,
        "finished_at": _now(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    progress.mark_complete(model, mode)
    print(f"    Done: {total_tokens:,} tokens, hash={rolling_hash.hexdigest()[:16]}...")


# ── Server management ──────────────────────────────────────────────────────

def wait_for_server(timeout: int = 300) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = urllib.request.urlopen(f"{SERVER_URL.rsplit('/', 1)[0]}/models", timeout=5)
            if resp.status == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def server_is_up() -> bool:
    try:
        resp = urllib.request.urlopen(
            SERVER_URL.replace("/v1/chat/completions", "/health"), timeout=5
        )
        return resp.status == 200
    except Exception:
        return False


# ── Compare ────────────────────────────────────────────────────────────────

def compare_servers(out_dir: Path, sid_a: str, sid_b: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  Comparing {sid_a} vs {sid_b}")
    print(f"{'=' * 70}\n")

    for model_dir in sorted(out_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for mode in ["repeated", "diverse"]:
            dir_a = model_dir / sid_a / mode
            dir_b = model_dir / sid_b / mode

            if not (dir_a / "chunks.json").exists() or not (dir_b / "chunks.json").exists():
                continue

            chunks_a = json.loads((dir_a / "chunks.json").read_text())
            chunks_b = json.loads((dir_b / "chunks.json").read_text())

            n = min(len(chunks_a), len(chunks_b))
            matches = sum(1 for i in range(n) if chunks_a[i]["hash"] == chunks_b[i]["hash"])

            verdict = "IDENTICAL" if matches == n else f"DIVERGENT (first mismatch at chunk {next(i for i in range(n) if chunks_a[i]['hash'] != chunks_b[i]['hash'])})"

            print(f"  {model_name:25s} {mode:10s}  {matches}/{n} match  {verdict}")

    print()


# ── Main ───────────────────────────────────────────────────────────────────

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tiered determinism experiment (smoke → medium → full)"
    )
    parser.add_argument(
        "--tier",
        choices=["smoke", "medium", "full"],
        help="Test tier to run",
    )
    parser.add_argument("--server-id", default="s1", help="Server identifier")
    parser.add_argument(
        "--out-dir",
        default="/tmp/determinism-experiment",
        help="Output directory (default: /tmp/determinism-experiment)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Override model list (default: all 3)",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("S1", "S2"),
        help="Compare two server results",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print current progress and exit",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # Progress check
    if args.progress:
        txt = out_dir / "progress.txt"
        if txt.exists():
            print(txt.read_text())
        else:
            print("No progress file found.")
        return 0

    # Compare mode
    if args.compare:
        compare_servers(out_dir, args.compare[0], args.compare[1])
        return 0

    # Run mode
    if not args.tier:
        parser.print_help()
        return 1

    tier = TIERS[args.tier]
    models = args.models or MODELS

    print(f"\n{'=' * 70}")
    print(f"  Determinism Experiment — {args.tier} tier")
    print(f"  {tier['description']}")
    print(f"  Server: {args.server_id}")
    print(f"  Models: {', '.join(m.split('/')[-1] for m in models)}")
    print(f"  Repeated: {tier['n_repeated']} × {tier['tokens_per_request']:,} tokens")
    print(f"  Diverse:  {tier['n_diverse']} × {tier['tokens_per_request']:,} tokens")
    print(f"  Output:   {out_dir}")
    print(f"{'=' * 70}\n")

    progress = Progress(out_dir, args.server_id, args.tier)

    if not server_is_up():
        print("ERROR: vLLM server is not running at", SERVER_URL)
        print("Start it first, then re-run this script.")
        return 1

    for model in models:
        model_short = model.split("/")[-1]
        print(f"\n  ── {model_short} ──")

        # Check the server is serving this model
        try:
            resp = urllib.request.urlopen(
                SERVER_URL.replace("/v1/chat/completions", "/v1/models"), timeout=10
            )
            served_models = json.loads(resp.read())
            served_ids = [m["id"] for m in served_models.get("data", [])]
            if model not in served_ids:
                print(f"    SKIP: server is serving {served_ids}, not {model}")
                print(f"    Restart the server with --model {model} and re-run.")
                progress.mark_error(model, "repeated", f"wrong model: {served_ids}")
                continue
        except Exception as e:
            print(f"    WARNING: could not check served model: {e}")

        print(f"\n    --- Repeated ({tier['n_repeated']} runs) ---")
        try:
            run_mode(
                model=model,
                mode="repeated",
                server_id=args.server_id,
                out_dir=out_dir,
                n_chunks=tier["n_repeated"],
                tokens_per_request=tier["tokens_per_request"],
                progress=progress,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            progress.mark_error(model, "repeated", str(e))

        print(f"\n    --- Diverse ({tier['n_diverse']} prompts) ---")
        try:
            run_mode(
                model=model,
                mode="diverse",
                server_id=args.server_id,
                out_dir=out_dir,
                n_chunks=tier["n_diverse"],
                tokens_per_request=tier["tokens_per_request"],
                progress=progress,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            progress.mark_error(model, "diverse", str(e))

    print(f"\n{'=' * 70}")
    print(f"  {args.tier} tier complete")
    print(f"  Progress: {out_dir / 'progress.txt'}")
    print(f"{'=' * 70}\n")

    # Print summary
    print((out_dir / "progress.txt").read_text())

    return 0


if __name__ == "__main__":
    sys.exit(main())
