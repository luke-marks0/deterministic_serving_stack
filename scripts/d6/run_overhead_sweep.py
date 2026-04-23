"""Run the full determinism overhead sweep and produce a summary table.

Usage:
  python3 scripts/d6/run_overhead_sweep.py --output results/overhead.jsonl

Iterates all (model, config, batch_size, max_tokens) combos, calling
benchmark_determinism_overhead.py as a subprocess for each so the LLM
object is fully deallocated between runs (avoids CUDA context leaks).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
CONFIGS = ["c0", "c1", "c2", "c3"]
BATCH_SIZES = [1, 4, 16, 64, 128]
MAX_TOKENS = [16, 128, 512, 2048]

SCRIPT = Path(__file__).parent / "benchmark_determinism_overhead.py"


def run_one(model: str, config: str, batch: int, max_tok: int) -> dict | None:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        sys.executable, str(SCRIPT),
        "--model", model,
        "--config", config,
        "--batch-size", str(batch),
        "--max-tokens", str(max_tok),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, env=env)
        if r.returncode != 0:
            print(f"  FAIL: {model} {config} b={batch} t={max_tok}: {r.stderr[-500:]}", file=sys.stderr)
            return None
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        print(f"  NO JSON: {model} {config} b={batch} t={max_tok}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT: {model} {config} b={batch} t={max_tok}", file=sys.stderr)
        return None


def generate_table(results: list[dict], output_path: Path) -> None:
    md = Path(str(output_path).replace(".jsonl", ".md"))
    lines = ["# Determinism Overhead Benchmark\n"]

    for model in MODELS:
        model_results = [r for r in results if r["model"] == model]
        if not model_results:
            continue
        short_name = model.split("/")[-1]
        lines.append(f"\n## {short_name}\n")

        for batch in BATCH_SIZES:
            batch_results = [r for r in model_results if r["batch_size"] == batch]
            if not batch_results:
                continue
            lines.append(f"\n### batch={batch}\n")
            header = "| Config |"
            sep = "|--------|"
            for mt in MAX_TOKENS:
                header += f" max_tok={mt} |"
                sep += "------------|"
            lines.append(header)
            lines.append(sep)

            baseline = {}
            for r in batch_results:
                if r["config"] == "c0":
                    baseline[r["max_tokens"]] = r["tok_per_s"]

            for config in CONFIGS:
                row = f"| {config:10s} |"
                for mt in MAX_TOKENS:
                    match = [r for r in batch_results if r["config"] == config and r["max_tokens"] == mt]
                    if match:
                        tps = match[0]["tok_per_s"]
                        base = baseline.get(mt)
                        if base and config != "c0" and base > 0:
                            pct = (tps - base) / base * 100
                            row += f" {tps:.0f} t/s ({pct:+.1f}%) |"
                        else:
                            row += f" {tps:.0f} t/s |"
                    else:
                        row += " — |"
                lines.append(row)

    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/overhead.jsonl")
    p.add_argument("--model", help="Run only this model (skip the other)")
    p.add_argument("--config", help="Run only this config")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else MODELS
    configs = [args.config] if args.config else CONFIGS

    results: list[dict] = []
    if out.exists():
        results = [json.loads(l) for l in out.read_text().strip().split("\n") if l.strip()]
        print(f"loaded {len(results)} existing results from {out}")

    total = len(models) * len(configs) * len(BATCH_SIZES) * len(MAX_TOKENS)
    done = 0

    for model in models:
        for config in configs:
            for batch in BATCH_SIZES:
                for max_tok in MAX_TOKENS:
                    done += 1
                    existing = [
                        r for r in results
                        if r["model"] == model and r["config"] == config
                        and r["batch_size"] == batch and r["max_tokens"] == max_tok
                    ]
                    if existing:
                        print(f"[{done}/{total}] skip {model.split('/')[-1]} {config} b={batch} t={max_tok} (exists)")
                        continue

                    print(f"[{done}/{total}] {model.split('/')[-1]} {config} b={batch} t={max_tok} ...", end=" ", flush=True)
                    r = run_one(model, config, batch, max_tok)
                    if r:
                        results.append(r)
                        with open(out, "a") as f:
                            f.write(json.dumps(r) + "\n")
                        print(f"{r['tok_per_s']} t/s ({r['wall_s']}s)")
                    else:
                        print("FAILED")

    generate_table(results, out)
    print(f"\n{len(results)} total results in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
