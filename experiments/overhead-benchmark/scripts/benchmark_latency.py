"""Benchmark per-request latency (TTFT + total) for each determinism config.

Uses vLLM's offline LLM API with a single request at a time to isolate
latency without batching effects. For each (model, config, prompt_length,
max_tokens) combo, runs N repetitions and reports min/median/p95/max.

Usage:
  python3 scripts/d6/benchmark_latency.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --config c0 \
    --max-tokens 128 \
    --prompt-length short \
    --reps 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

SHORT_PROMPT = "What is 2 + 2?"
MEDIUM_PROMPT = "Explain how photosynthesis works in one paragraph. Cover the light-dependent reactions, the Calvin cycle, and the role of chlorophyll."
LONG_PROMPT = (
    "Write a detailed essay about the development of human civilization "
    "from the agricultural revolution to the modern era. Cover at least "
    "three major turning points and explain their lasting consequences. "
    "Include discussion of technological, social, and political changes. "
    "Consider how geography influenced the spread of ideas and empires. "
    "Discuss the role of trade routes, religious movements, and scientific "
    "discoveries in shaping the world we live in today."
)

PROMPTS = {
    "short": SHORT_PROMPT,
    "medium": MEDIUM_PROMPT,
    "long": LONG_PROMPT,
}


def apply_config(config: str) -> dict:
    kwargs: dict = {}
    if config in ("c1", "c2", "c3"):
        kwargs["enforce_eager"] = True
    if config in ("c2", "c3"):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    if config == "c3":
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        kwargs["attention_backend"] = "FLASH_ATTN"
    else:
        os.environ.pop("VLLM_BATCH_INVARIANT", None)
    return kwargs


def run_latency_bench(
    model: str,
    config: str,
    prompt_length: str,
    max_tokens: int,
    reps: int,
    seed: int = 42,
) -> dict:
    extra_kwargs = apply_config(config)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model,
        seed=seed,
        dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
        **extra_kwargs,
    )

    prompt = PROMPTS[prompt_length]
    params = SamplingParams(temperature=0, max_tokens=max_tokens)

    # Warmup (2 runs, discarded)
    for _ in range(2):
        _ = llm.generate([prompt], params)

    # Timed runs — one request at a time, measure wall time per request
    wall_times = []
    token_counts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], params)
        t1 = time.perf_counter()
        wall_times.append(t1 - t0)
        token_counts.append(len(outputs[0].outputs[0].token_ids))

    del llm
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # Compute per-token latency (inter-token latency proxy)
    per_token = [w / t if t > 0 else 0 for w, t in zip(wall_times, token_counts)]

    def stats(vals):
        s = sorted(vals)
        return {
            "min": round(s[0] * 1000, 2),
            "median": round(statistics.median(s) * 1000, 2),
            "p95": round(s[int(len(s) * 0.95)] * 1000, 2) if len(s) >= 5 else round(s[-1] * 1000, 2),
            "max": round(s[-1] * 1000, 2),
            "mean": round(statistics.mean(s) * 1000, 2),
        }

    return {
        "config": config,
        "model": model,
        "prompt_length": prompt_length,
        "max_tokens": max_tokens,
        "reps": reps,
        "tokens_generated": token_counts[0],
        "request_latency_ms": stats(wall_times),
        "per_token_latency_ms": stats(per_token),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True, choices=["c0", "c1", "c2", "c3"])
    p.add_argument("--max-tokens", type=int, required=True)
    p.add_argument("--prompt-length", required=True, choices=["short", "medium", "long"])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.environ["PYTHONHASHSEED"] = "0"

    result = run_latency_bench(
        model=args.model,
        config=args.config,
        prompt_length=args.prompt_length,
        max_tokens=args.max_tokens,
        reps=args.reps,
        seed=args.seed,
    )
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
