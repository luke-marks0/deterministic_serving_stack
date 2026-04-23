"""Benchmark Time To First Token (TTFT) for each determinism config.

Measures wall time of llm.generate() with max_tokens=1, which is
prefill + exactly one decode step. Since decode-per-token is known from
the latency benchmark, TTFT ≈ wall_time(max_tokens=1) - decode_per_token.

Also measures max_tokens=2 so we can compute decode_step = t2 - t1,
giving us a clean TTFT = t1 - decode_step.

Usage:
  python3 scripts/d6/benchmark_ttft.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --config c0 --prompt-length short --reps 10
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

PROMPTS = {
    "short": "What is 2 + 2?",
    "medium": "Explain how photosynthesis works in one paragraph. Cover the light-dependent reactions, the Calvin cycle, and the role of chlorophyll.",
    "long": (
        "Write a detailed essay about the development of human civilization "
        "from the agricultural revolution to the modern era. Cover at least "
        "three major turning points and explain their lasting consequences. "
        "Include discussion of technological, social, and political changes. "
        "Consider how geography influenced the spread of ideas and empires. "
        "Discuss the role of trade routes, religious movements, and scientific "
        "discoveries in shaping the world we live in today."
    ),
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


def run_ttft_bench(model, config, prompt_length, reps, seed=42):
    extra_kwargs = apply_config(config)
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model, seed=seed, dtype="auto",
        gpu_memory_utilization=0.90, max_model_len=4096,
        trust_remote_code=True, **extra_kwargs,
    )

    prompt = PROMPTS[prompt_length]

    # Warmup
    for _ in range(3):
        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=1))

    # Measure max_tokens=1 (prefill + 1 decode)
    t1_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=1))
        t1_times.append(time.perf_counter() - t0)

    # Measure max_tokens=2 (prefill + 2 decode)
    t2_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        llm.generate([prompt], SamplingParams(temperature=0, max_tokens=2))
        t2_times.append(time.perf_counter() - t0)

    del llm
    import torch; torch.cuda.empty_cache()
    import gc; gc.collect()

    # decode_step = median(t2) - median(t1)
    med_t1 = statistics.median(t1_times)
    med_t2 = statistics.median(t2_times)
    decode_step = max(med_t2 - med_t1, 0.0001)
    ttft = max(med_t1 - decode_step, 0)

    def ms_stats(vals):
        s = sorted(vals)
        return {
            "min": round(s[0]*1000, 2),
            "median": round(statistics.median(s)*1000, 2),
            "p95": round(s[int(len(s)*0.95)]*1000, 2) if len(s) >= 10 else round(s[-1]*1000, 2),
            "max": round(s[-1]*1000, 2),
        }

    return {
        "config": config,
        "model": model,
        "prompt_length": prompt_length,
        "reps": reps,
        "t1_ms": ms_stats(t1_times),
        "t2_ms": ms_stats(t2_times),
        "decode_step_ms": round(decode_step * 1000, 2),
        "ttft_ms": round(ttft * 1000, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--config", required=True, choices=["c0","c1","c2","c3"])
    p.add_argument("--prompt-length", required=True, choices=["short","medium","long"])
    p.add_argument("--reps", type=int, default=10)
    args = p.parse_args()
    os.environ["PYTHONHASHSEED"] = "0"
    result = run_ttft_bench(args.model, args.config, args.prompt_length, args.reps)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
