"""Run latency sweep across all configs, models, prompt lengths, and output sizes.

Usage:
  python3 scripts/d6/run_latency_sweep.py --output results/latency.jsonl
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
PROMPT_LENGTHS = ["short", "medium", "long"]
MAX_TOKENS = [16, 128, 512]
REPS = 5

SCRIPT = Path(__file__).parent / "benchmark_latency.py"


def run_one(model, config, prompt_len, max_tok) -> dict | None:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        sys.executable, str(SCRIPT),
        "--model", model,
        "--config", config,
        "--max-tokens", str(max_tok),
        "--prompt-length", prompt_len,
        "--reps", str(REPS),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if r.returncode != 0:
            print(f"  FAIL: {r.stderr[-300:]}", file=sys.stderr)
            return None
        for line in r.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return None
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT", file=sys.stderr)
        return None


def generate_report(results: list[dict], output_path: Path) -> None:
    md = Path(str(output_path).replace(".jsonl", "_report.md"))
    lines = ["# Latency Benchmark — Per-Request Determinism Overhead", ""]

    models = list(dict.fromkeys(r["model"] for r in results))
    model_short = {m: m.split("/")[-1] for m in models}

    lines.append("## Configs (cumulative)")
    lines.append("")
    lines.append("| Config | What it adds |")
    lines.append("|--------|-------------|")
    lines.append("| **c0** | Baseline (all optimizations) |")
    lines.append("| **c1** | + `enforce_eager=True` |")
    lines.append("| **c2** | + `CUBLAS_WORKSPACE_CONFIG=:4096:8` |")
    lines.append("| **c3** | + `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` |")
    lines.append("")
    lines.append(f"Each measurement: batch_size=1, {REPS} repetitions after 2 warmup runs.")
    lines.append("")

    for model in models:
        short = model_short[model]
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"# {short}")
        lines.append(f"")

        for prompt_len in PROMPT_LENGTHS:
            lines.append(f"## prompt={prompt_len}")
            lines.append(f"")

            # Request latency table
            lines.append("### Request latency (ms) — total time for one request, batch=1")
            lines.append("")
            header = "| Config |"
            sep = "|--------|"
            for mt in MAX_TOKENS:
                header += f" {mt} tok (median) | {mt} tok (p95) |"
                sep += "----------------|---------------|"
            lines.append(header)
            lines.append(sep)

            baseline_medians = {}
            for config in CONFIGS:
                row = f"| **{config}** |"
                for mt in MAX_TOKENS:
                    match = [r for r in results if r["model"] == model and r["config"] == config
                             and r["prompt_length"] == prompt_len and r["max_tokens"] == mt]
                    if match:
                        lat = match[0]["request_latency_ms"]
                        med = lat["median"]
                        p95 = lat["p95"]
                        if config == "c0":
                            baseline_medians[(prompt_len, mt)] = med
                            row += f" {med:.0f} | {p95:.0f} |"
                        else:
                            base = baseline_medians.get((prompt_len, mt))
                            if base and base > 0:
                                pct = (med - base) / base * 100
                                row += f" {med:.0f} ({pct:+.0f}%) | {p95:.0f} |"
                            else:
                                row += f" {med:.0f} | {p95:.0f} |"
                    else:
                        row += " — | — |"
                lines.append(row)
            lines.append("")

            # Per-token latency
            lines.append("### Per-token latency (ms/tok) — request_time / tokens_generated")
            lines.append("")
            header2 = "| Config |"
            sep2 = "|--------|"
            for mt in MAX_TOKENS:
                header2 += f" {mt} tok |"
                sep2 += "---------|"
            lines.append(header2)
            lines.append(sep2)

            for config in CONFIGS:
                row = f"| **{config}** |"
                for mt in MAX_TOKENS:
                    match = [r for r in results if r["model"] == model and r["config"] == config
                             and r["prompt_length"] == prompt_len and r["max_tokens"] == mt]
                    if match:
                        ptl = match[0]["per_token_latency_ms"]
                        row += f" {ptl['median']:.2f} |"
                    else:
                        row += " — |"
                lines.append(row)
            lines.append("")

    # Summary
    lines.append("---")
    lines.append("")
    lines.append("# Summary — median request latency overhead (c0 → c3)")
    lines.append("")
    lines.append("| Model | Prompt | 16 tok | 128 tok | 512 tok |")
    lines.append("|-------|--------|--------|---------|---------|")
    for model in models:
        short = model_short[model]
        for pl in PROMPT_LENGTHS:
            row = f"| {short} | {pl} |"
            for mt in MAX_TOKENS:
                c0 = [r for r in results if r["model"] == model and r["config"] == "c0"
                       and r["prompt_length"] == pl and r["max_tokens"] == mt]
                c3 = [r for r in results if r["model"] == model and r["config"] == "c3"
                       and r["prompt_length"] == pl and r["max_tokens"] == mt]
                if c0 and c3:
                    m0 = c0[0]["request_latency_ms"]["median"]
                    m3 = c3[0]["request_latency_ms"]["median"]
                    pct = (m3 - m0) / m0 * 100 if m0 > 0 else 0
                    row += f" {m0:.0f}→{m3:.0f}ms ({pct:+.0f}%) |"
                else:
                    row += " — |"
            lines.append(row)
    lines.append("")

    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {md}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/latency.jsonl")
    p.add_argument("--model", help="Run only this model")
    args = p.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    models = [args.model] if args.model else MODELS
    results: list[dict] = []
    if out.exists():
        results = [json.loads(l) for l in out.read_text().strip().split("\n") if l.strip()]
        print(f"loaded {len(results)} existing results")

    total = len(models) * len(CONFIGS) * len(PROMPT_LENGTHS) * len(MAX_TOKENS)
    done = 0

    for model in models:
        for config in CONFIGS:
            for pl in PROMPT_LENGTHS:
                for mt in MAX_TOKENS:
                    done += 1
                    existing = [r for r in results if r["model"] == model and r["config"] == config
                                and r["prompt_length"] == pl and r["max_tokens"] == mt]
                    if existing:
                        print(f"[{done}/{total}] skip (exists)")
                        continue
                    short = model.split("/")[-1]
                    print(f"[{done}/{total}] {short} {config} prompt={pl} tok={mt} ...", end=" ", flush=True)
                    r = run_one(model, config, pl, mt)
                    if r:
                        results.append(r)
                        with open(out, "a") as f:
                            f.write(json.dumps(r) + "\n")
                        med = r["request_latency_ms"]["median"]
                        ptl = r["per_token_latency_ms"]["median"]
                        print(f"{med:.0f}ms total, {ptl:.2f}ms/tok")
                    else:
                        print("FAILED")

    generate_report(results, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
