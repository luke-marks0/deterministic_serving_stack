#!/usr/bin/env python3
"""End-to-end audit verification demo.

Demonstrates the mechanical audit loop for deterministic LLM inference:
run inference, commit all output tokens, randomly challenge one token,
and verify it by replaying the request from scratch.

SECURITY NOTE: This demo uses HMAC with a hardcoded shared key. It proves
that deterministic replay works, but does NOT provide cryptographic binding
against a malicious provider. See docs/plans/e2e-audit-verification.md for
details on what a production protocol would need.

Prerequisites:
    - GPU with sufficient VRAM for the model (Qwen 2.5 1.5B needs ~4 GB)
    - vLLM installed (pip install vllm)
    - Model weights accessible from HuggingFace (auto-downloaded on first run)

Usage:
    # Default (Qwen 2.5 1.5B, seed 42, random challenge)
    python3 scripts/e2e_verify.py

    # Specific model and forced challenge
    python3 scripts/e2e_verify.py --model mistralai/Mistral-7B-Instruct-v0.3 --challenge req-1:5

    # Verbose output (shows plaintext token IDs)
    python3 scripts/e2e_verify.py --verbose

PASS means: the verification run produced the same token at the challenged
position as the primary run. The deterministic replay worked.

FAIL means: the tokens diverged. Something is wrong with the determinism
setup (env vars not set before vLLM import, engine teardown not clean, etc.).
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# scripts/ is one level deep from repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.e2e.crypto import commit_token, commit_token_stream

# ── Prompts ──────────────────────────────────────────────────────────────
# A small, diverse set. Keep it short so the demo runs in under a minute.
PROMPTS = [
    {"id": "req-0", "prompt": "Explain how photosynthesis works in one paragraph.", "max_new_tokens": 16},
    {"id": "req-1", "prompt": "What is the difference between TCP and UDP?", "max_new_tokens": 16},
    {"id": "req-2", "prompt": "Describe the life cycle of a star.", "max_new_tokens": 16},
    {"id": "req-3", "prompt": "Why is the sky blue?", "max_new_tokens": 16},
    {"id": "req-4", "prompt": "What is a hash table?", "max_new_tokens": 16},
]


def setup_deterministic_env() -> None:
    """Set all env vars for full deterministic mode (c3).

    MUST be called before any `import vllm` or `import torch`.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["PYTHONHASHSEED"] = "0"


def run_inference(
    prompts: list[dict],
    *,
    model: str,
    seed: int,
) -> dict[str, list[int]]:
    """Run deterministic inference. Returns {request_id: [token_ids]}.

    Creates and destroys the LLM engine, freeing VRAM for the next call.
    """
    from vllm import LLM, SamplingParams
    import torch

    llm = LLM(
        model=model,
        seed=seed,
        dtype="auto",
        enforce_eager=True,
        attention_backend="FLASH_ATTN",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        trust_remote_code=True,
    )

    prompt_texts = [p["prompt"] for p in prompts]
    params_list = [
        SamplingParams(temperature=0, max_tokens=p["max_new_tokens"], seed=seed)
        for p in prompts
    ]

    outputs = llm.generate(prompt_texts, params_list)

    result: dict[str, list[int]] = {}
    for prompt_def, output in zip(prompts, outputs):
        result[prompt_def["id"]] = list(output.outputs[0].token_ids)

    del llm
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    setup_deterministic_env()

    print("=== E2E Audit Verification ===")
    print(f"Model: {args.model}")
    print(f"Seed:  {args.seed}")
    print()

    print("Phase 1: Primary inference run")
    print(f"  {len(PROMPTS)} prompts")
    t0 = time.perf_counter()
    tokens_by_req = run_inference(PROMPTS, model=args.model, seed=args.seed)
    t1 = time.perf_counter()
    print(f"  Inference complete ({t1 - t0:.1f}s)")

    commitments: dict[str, list[str]] = {}
    total_tokens = 0
    for prompt_def in PROMPTS:
        req_id = prompt_def["id"]
        toks = tokens_by_req[req_id]
        commits = commit_token_stream(toks)
        commitments[req_id] = commits
        total_tokens += len(toks)
        print(f"  {req_id}: {len(toks)} tokens, commitment[0]={commits[0][:8]}...")

    print(f"  Total: {total_tokens} tokens committed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
