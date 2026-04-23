"""Run TTFT sweep and produce a report."""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

MODELS = ["Qwen/Qwen2.5-1.5B-Instruct", "mistralai/Mistral-7B-Instruct-v0.3"]
CONFIGS = ["c0", "c1", "c2", "c3"]
PROMPT_LENGTHS = ["short", "medium", "long"]
REPS = 10
SCRIPT = Path(__file__).parent / "benchmark_ttft.py"

def run_one(model, config, pl):
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib64:/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [sys.executable, str(SCRIPT), "--model", model, "--config", config,
           "--prompt-length", pl, "--reps", str(REPS)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        if r.returncode != 0:
            print(f"  FAIL: {r.stderr[-300:]}", file=sys.stderr); return None
        for line in r.stdout.strip().split("\n"):
            if line.strip().startswith("{"): return json.loads(line.strip())
        return None
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", file=sys.stderr); return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/ttft.jsonl")
    args = p.parse_args()
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    if out.exists():
        results = [json.loads(l) for l in out.read_text().strip().split("\n") if l.strip()]
    total = len(MODELS) * len(CONFIGS) * len(PROMPT_LENGTHS)
    done = 0
    for model in MODELS:
        for config in CONFIGS:
            for pl in PROMPT_LENGTHS:
                done += 1
                existing = [r for r in results if r["model"]==model and r["config"]==config and r["prompt_length"]==pl]
                if existing:
                    print(f"[{done}/{total}] skip (exists)"); continue
                short = model.split("/")[-1]
                print(f"[{done}/{total}] {short} {config} prompt={pl} ...", end=" ", flush=True)
                r = run_one(model, config, pl)
                if r:
                    results.append(r)
                    with open(out, "a") as f: f.write(json.dumps(r)+"\n")
                    print(f"TTFT={r['ttft_ms']:.1f}ms  decode_step={r['decode_step_ms']:.1f}ms  t1={r['t1_ms']['median']:.1f}ms")
                else:
                    print("FAILED")

    # Generate report
    md = Path(str(out).replace(".jsonl", "_report.md"))
    lines = ["# TTFT Benchmark — Time To First Token", "",
             "TTFT = wall_time(max_tokens=1) - one_decode_step", "",
             "where decode_step = median(wall_time(max_tokens=2)) - median(wall_time(max_tokens=1))", "",
             f"Each measurement: batch_size=1, {REPS} reps after 3 warmup.", ""]
    model_short = {m: m.split("/")[-1] for m in MODELS}
    for model in MODELS:
        short = model_short[model]
        lines += [f"---", f"", f"# {short}", ""]
        lines += ["| Config | short prompt | medium prompt | long prompt |",
                   "|--------|-------------|---------------|-------------|"]
        baseline = {}
        for config in CONFIGS:
            row = f"| **{config}** |"
            for pl in PROMPT_LENGTHS:
                match = [r for r in results if r["model"]==model and r["config"]==config and r["prompt_length"]==pl]
                if match:
                    ttft = match[0]["ttft_ms"]
                    if config == "c0":
                        baseline[pl] = ttft
                        row += f" {ttft:.1f}ms |"
                    else:
                        base = baseline.get(pl)
                        pct = (ttft - base)/base*100 if base and base > 0 else 0
                        row += f" {ttft:.1f}ms ({pct:+.0f}%) |"
                else:
                    row += " — |"
            lines.append(row)
        lines += ["", "**Decode step (ms/tok) and t1 (prefill + 1 decode):**", "",
                   "| Config | short t1 | short decode | medium t1 | medium decode | long t1 | long decode |",
                   "|--------|----------|-------------|-----------|--------------|---------|------------|"]
        for config in CONFIGS:
            row = f"| **{config}** |"
            for pl in PROMPT_LENGTHS:
                match = [r for r in results if r["model"]==model and r["config"]==config and r["prompt_length"]==pl]
                if match:
                    row += f" {match[0]['t1_ms']['median']:.1f}ms | {match[0]['decode_step_ms']:.1f}ms |"
                else:
                    row += " — | — |"
            lines.append(row)
        lines.append("")

    # Summary
    lines += ["---", "", "# Summary — TTFT overhead (c0 → c3)", "",
              "| Model | short | medium | long |",
              "|-------|-------|--------|------|"]
    for model in MODELS:
        short = model_short[model]
        row = f"| {short} |"
        for pl in PROMPT_LENGTHS:
            c0 = [r for r in results if r["model"]==model and r["config"]=="c0" and r["prompt_length"]==pl]
            c3 = [r for r in results if r["model"]==model and r["config"]=="c3" and r["prompt_length"]==pl]
            if c0 and c3:
                t0, t3 = c0[0]["ttft_ms"], c3[0]["ttft_ms"]
                pct = (t3-t0)/t0*100 if t0 > 0 else 0
                row += f" {t0:.1f}→{t3:.1f}ms ({pct:+.0f}%) |"
            else:
                row += " — |"
        lines.append(row)
    lines.append("")
    md.write_text("\n".join(lines)+"\n")
    print(f"wrote {md}")

if __name__ == "__main__":
    main()
