#!/usr/bin/env python3
"""vLLM execution backend for the deterministic runner.

Loads a model via vLLM's offline LLM class with batch invariance,
executes requests, and returns structured observables.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any


def _set_deterministic_env(knobs: dict[str, Any]) -> dict[str, str]:
    """Set environment variables for deterministic execution. Returns the env snapshot."""
    env = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_LAUNCH_BLOCKING": str(int(knobs.get("cuda_launch_blocking", True))),
        "PYTHONHASHSEED": "0",
    }
    for key, value in env.items():
        os.environ[key] = value
    return env


def _resolve_model_path(manifest: dict[str, Any], lockfile: dict[str, Any]) -> str:
    """Determine the model path/name for vLLM.

    Prefers a local model directory if RUNNER_MODEL_PATH is set,
    otherwise uses the HF model ID from the manifest source field.
    """
    env_path = os.getenv("RUNNER_MODEL_PATH")
    if env_path and Path(env_path).is_dir():
        return env_path

    source = manifest["model"]["source"]
    if source.startswith("hf://"):
        return source[len("hf://"):]
    return source


def run_vllm(
    manifest: dict[str, Any],
    lockfile: dict[str, Any],
) -> dict[str, Any]:
    """Execute vLLM inference and return observables.

    Returns dict with keys: request_outputs, engine_events, frames, env_info
    """
    # These imports are deferred so the module can be imported on machines without vLLM
    # (e.g. for schema validation or synthetic mode).
    import torch
    from vllm import LLM, SamplingParams

    runtime = manifest["runtime"]
    knobs = runtime["deterministic_knobs"]
    batch_inv = runtime.get("batch_invariance", {})

    resolved_env = _set_deterministic_env(knobs)

    if knobs.get("torch_deterministic", False):
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model_path = _resolve_model_path(manifest, lockfile)
    seed = knobs.get("seed", 42)

    # Build vLLM engine args
    engine_kwargs: dict[str, Any] = {
        "model": model_path,
        "seed": seed,
        "dtype": "auto",
        "trust_remote_code": bool(manifest["model"].get("trust_remote_code", False)),
        "gpu_memory_utilization": float(os.getenv("RUNNER_GPU_MEM_UTIL", "0.90")),
    }

    if batch_inv.get("enforce_eager", False):
        engine_kwargs["enforce_eager"] = True

    max_model_len = os.getenv("RUNNER_MAX_MODEL_LEN")
    if max_model_len:
        engine_kwargs["max_model_len"] = int(max_model_len)

    # vLLM batch invariance (requires vLLM >= 0.8.x with batch invariance support)
    if batch_inv.get("enabled", False):
        engine_kwargs["enable_batch_invariance"] = True

    llm = LLM(**engine_kwargs)

    # Prepare requests
    prompts = []
    sampling_params_list = []
    request_ids = []
    for req in manifest["requests"]:
        prompts.append(req["prompt"])
        request_ids.append(req["id"])
        sampling_params_list.append(
            SamplingParams(
                temperature=req["temperature"],
                max_tokens=req["max_new_tokens"],
                logprobs=20,
                seed=seed,
            )
        )

    # Execute inference
    t0 = time.monotonic()
    outputs = llm.generate(prompts, sampling_params_list)
    inference_time = time.monotonic() - t0

    # Extract observables
    request_outputs = []

    for idx, (req_id, output) in enumerate(zip(request_ids, outputs)):
        result = output.outputs[0]
        tokens = list(result.token_ids)

        # Extract logprobs as flat list of floats (log probability of the chosen token)
        logits: list[float] = []
        if result.logprobs:
            for step_logprobs in result.logprobs:
                chosen_token = tokens[len(logits)] if len(logits) < len(tokens) else 0
                if chosen_token in step_logprobs:
                    logits.append(round(float(step_logprobs[chosen_token].logprob), 8))
                else:
                    logits.append(0.0)

        request_outputs.append({
            "id": req_id,
            "tokens": tokens,
            "logits": logits,
            "text": result.text,
            "finish_reason": result.finish_reason,
        })

    # Collect environment info from the running vLLM instance
    gpu_inventory = []
    driver_version = manifest["hardware_profile"]["gpu"]["driver_version"]
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        for line in smi.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if parts:
                gpu_inventory.append(parts[0])
                if len(parts) > 1:
                    driver_version = parts[1]
    except Exception:
        gpu_inventory = [manifest["hardware_profile"]["gpu"]["model"]]

    env_info = {
        "vllm_version": _get_vllm_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "unknown",
        "driver_version": driver_version,
        "gpu_inventory": gpu_inventory,
        "inference_time_s": round(inference_time, 3),
    }

    return {
        "request_outputs": request_outputs,
        "env_info": env_info,
        "resolved_env": resolved_env,
    }


def _get_vllm_version() -> str:
    try:
        import vllm
        return getattr(vllm, "__version__", "unknown")
    except Exception:
        return "unknown"
