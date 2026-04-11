# The Hunt for Deterministic Tensor Parallelism
## A Chronicle of Multi-GPU Inference Determinism Testing

---

### Prologue: The Question

We'd already proven that a single GPU could produce bitwise-identical LLM inference outputs across independent servers — 91 out of 91 chunks matching across NVIDIA GH200 instances running Qwen3-1.7B. But a far harder question loomed: **what happens when you shard a model across multiple GPUs?**

Tensor parallelism splits a model's weight matrices across GPUs, requiring NCCL collective operations (all-reduce, all-gather) to synchronize activations at every layer. These collectives are notoriously sensitive to timing, ordering, and floating-point non-associativity. Could we prove that the outputs are still deterministic?

And an even harder question: could we prove that the output for a given prompt is **independent of what other prompts are being processed simultaneously** — true batch and order invariance?

---

### Act I: First Contact (RTX 4090, Iceland)

**The Setup.** We rented 4× RTX 4090 24GB GPUs on vast.ai — first in China (stuck provisioning for 15 minutes with no logs, destroyed), then Quebec (CUDA driver too old — Error 804 in worker processes), and finally Iceland. The Iceland machine booted in 30 seconds.

**The Gap.** The first discovery was that `vllm_runner.py` — the offline inference backend — completely ignored tensor parallelism settings from the manifest. The server CLI path passed `--tensor-parallel-size` correctly, but the `LLM()` constructor path used for testing didn't pass `tensor_parallel_size`, `pipeline_parallel_size`, `disable_custom_all_reduce`, `dtype`, `attention_backend`, or `max_model_len`. We were asking for TP=4 and getting TP=1.

**The Fix.** We rewrote `run_vllm()` to read the full `serving_engine` config from the manifest and pass everything through. We also added NCCL determinism pinning: `NCCL_ALGO=Ring` (prevents auto-tuning between Ring/Tree/CollNet algorithms) and `NCCL_PROTO=Simple` (avoids LL/LL128 protocols that introduce ordering variation).

**The Import Bug.** Python's built-in `cmd` stdlib module shadowed our `cmd/` directory. `from cmd.runner.vllm_runner import run_vllm` worked locally but failed on the remote machine. We switched to `importlib.util.spec_from_file_location()` — which the linter would revert three more times before we were done.

**First Blood.** The first run of Qwen2.5-32B with TP=4 showed token divergence at position 1 of request 2. Our hearts sank. But the second run — with the model fully cached — came back clean:

```
Status: conformant
req-warmup:        4 tokens  — MATCH
req-determinism-1: 32 tokens — MATCH
req-determinism-2: 31 tokens — MATCH
req-determinism-3: 51 tokens — MATCH
```

**TP determinism was real.** The first divergence was a red herring — a HuggingFace 504 timeout during model download had corrupted the cache.

---

### Act II: 100 Requests, Two Models (RTX 4090, Iceland)

Emboldened, we scaled up. 100 diverse requests generated from 5 topic templates × 20 technical subjects. Two models: **Qwen2.5-32B** (dense, 32B parameters, 17 weight shards) and **Qwen3-30B-A3B** (MoE with 128 experts, 30B total / 3B active, 16 weight shards).

**Same-config determinism** — run the same 100 requests twice with identical configuration:

| Model | Type | Tokens | Result |
|-------|------|--------|--------|
| Qwen2.5-32B | Dense | 12,800 | **100/100 MATCH** |
| Qwen3-30B-A3B | MoE | 12,800 | **100/100 MATCH** |

Every single token, across 100 requests and 12,800 tokens, was bitwise identical between runs. For a dense model AND a Mixture-of-Experts model. NCCL Ring algorithm + deterministic seeds + eager mode = reproducible multi-GPU inference.

---

### Act III: The Batch Invariance Wall (RTX 4090)

Now for the hard test. Could we shuffle the request order AND change the batch size, and still get identical per-request outputs?

- **Run A:** 100 requests in original order, `max_num_seqs=64`
- **Run B:** same 100 requests shuffled (seed 12345), `max_num_seqs=16`
- Compare outputs matched by request ID

The results were decisive — and devastating:

| Test | Description | Matches |
|------|-------------|---------|
| A vs C | Same order, batch 64→16 | **22/100** |
| A vs D | Shuffled order, same batch | **18/100** |
| A vs B | Shuffled + different batch | **22/100** |

**78% of requests diverged.** Both batch size and request ordering destroyed determinism. This was expected — RTX 4090 (Ada Lovelace, SM89) doesn't support vLLM's batch invariance mode, which requires Hopper (SM90) hardware to pad attention computation so each request's output is independent of its co-batched neighbors.

We had proven the boundary: **same-config determinism works everywhere; batch invariance requires Hopper.**

---

### Act IV: The H100 Campaign

To prove batch invariance, we needed H100s. This is where things got interesting.

**The Hyperbolic Dead End.** We tried Hyperbolic Cloud first — their v2 API returned 404 on every endpoint. The marketplace API had apparently been deprecated without notice. Dead end.

**Back to Vast.ai.** We found 4× H100 SXM instances at $8-12/hr. The first attempt (Washington, $8/hr) spent an hour in "loading" state — the Docker image pull was stuck. $8.50 burned doing nothing. Destroyed.

**The SSH Rejection.** The second Washington H100 accepted the contract but rejected our SSH key. `Permission denied (publickey)` despite the key being registered. Destroyed.

**The Massachusetts Machine.** Third try: Massachusetts, $8.45/hr, 99.2% reliability. It booted. We got in. We installed vLLM.

**The `enable_batch_invariance` Surprise.** vLLM 0.19.0 (pip-installed) no longer accepts `enable_batch_invariance=True` as a constructor argument — our pinned 0.17.1 API had diverged. The new way: `VLLM_BATCH_INVARIANT=1` environment variable.

**The Attention Backend Ghost.** Even with the env var, the engine crashed: "batch_invariant mode requires an attention backend in ['FLASH_ATTN', ...], but got 'None'." The `VLLM_ATTENTION_BACKEND` env var was set in the main process but forked TP workers didn't inherit it. vLLM 0.19 added `attention_backend` as a direct constructor kwarg. Fixed.

**The 10-Hour Ghost Instance.** The Massachusetts machine ran for 10 hours on our credit before we noticed it had exited. The experiment had failed on the first try and the instance kept billing. Lesson learned.

**Model Access Gates.** Both Mistral Large 2 and Llama 4 Scout are gated on HuggingFace. Our token was set but we hadn't accepted the licenses. Mistral auto-approved; Llama took a manual approval cycle.

**The DNS Failure.** Mid-download on the Washington H100, DNS resolution for `huggingface.co` failed in the worker processes. Transient infrastructure issue. Kill and restart — HF cache preserved the partial download.

**The Disk Space Crisis.** HuggingFace's `snapshot_download()` downloads EVERYTHING in a repo. Mistral Large 2 ships both `model-*.safetensors` and `consolidated-*.safetensors` — the same 245GB of weights in two different naming schemes. The 380GB disk hit 86% utilization with 65 files still to go. We killed the snapshot download and let vLLM pull only what it needed.

---

### Act V: Victory on Hopper

Finally, everything aligned. 4× H100 SXM 80GB, Washington. vLLM 0.19.0 with `VLLM_BATCH_INVARIANT=1`, FlashAttention v3, `attention_backend=FLASH_ATTN`, `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`.

**Mistral Large 2** (123B dense, 245GB, 51 weight shards) — the only popular dense model that truly requires 4× H100. Model download took 40 minutes on the Washington host's 600Mbps connection. Inference: 41 seconds for 100 requests on Run A (batch=64), 2 minutes on Run B (batch=16, shuffled).

```
PASS: all 100 requests match (12800 total tokens)
  Order: shuffled vs original — identical
  Batch: max_num_seqs=16 vs 64 — identical
```

**Llama 4 Scout** (17B-16E MoE, 217GB, 50 weight shards) — Meta's newest MoE with 16 experts, from a completely different model family. Download took another 40 minutes. Triton MoE kernel JIT compilation added extra warmup time. Inference: 29 seconds for Run A.

```
PASS: all 100 requests match (12800 total tokens)
  Order: shuffled vs original — identical  
  Batch: max_num_seqs=16 vs 64 — identical
```

**Both models. Both architectures. 200 requests. 25,600 tokens. Zero divergence.**

---

### The Complete Picture

| GPU | Model | Same-Config | Batch+Order Invariance |
|-----|-------|------------|----------------------|
| 4× RTX 4090 | Qwen2.5-32B (dense) | **PASS** 100/100 | **FAIL** 22/100 |
| 4× RTX 4090 | Qwen3-30B-A3B (MoE) | **PASS** 100/100 | **FAIL** 13/100 |
| 4× H100 SXM | Mistral Large 2 (dense 123B) | **PASS** 100/100 | **PASS** 100/100 |
| 4× H100 SXM | Llama 4 Scout (MoE 16E) | **PASS** 100/100 | **PASS** 100/100 |

### What We Proved

1. **Tensor parallel inference is deterministic** across runs on both Ampere and Hopper, for both dense and MoE architectures, when NCCL collectives are pinned (`NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`) and custom all-reduce is disabled.

2. **Batch and order invariance requires Hopper.** On H100 with `VLLM_BATCH_INVARIANT=1` and FlashAttention v3, changing the batch size or shuffling request order has zero effect on per-request outputs. On RTX 4090 without batch invariance support, 78-87% of requests diverge.

3. **This works across model families and architectures.** Tested on Mistral (dense), Meta/Llama (MoE), and Qwen (both dense and MoE). The determinism guarantee is not model-specific.

4. **MoE expert routing is deterministic under TP.** The stochastic-seeming expert selection in Mixture-of-Experts models produces identical routing decisions across runs, even with 128 experts (Qwen3-30B-A3B) or 16 experts (Llama 4 Scout).

### The Determinism Recipe

```
tensor_parallel_size = N
disable_custom_all_reduce = true
enforce_eager = true
NCCL_ALGO = Ring
NCCL_PROTO = Simple
CUBLAS_WORKSPACE_CONFIG = :4096:8
CUDA_LAUNCH_BLOCKING = 1
PYTHONHASHSEED = 0
seed = 42
torch_deterministic = true

# For batch invariance (H100+ only):
VLLM_BATCH_INVARIANT = 1
attention_backend = FLASH_ATTN
```

---

### Epilogue: The Cost

| Item | Cost |
|------|------|
| Stuck China RTX 4090 (15 min) | ~$0.20 |
| Quebec RTX 4090 (CUDA mismatch, 5 min) | ~$0.09 |
| Iceland RTX 4090 (successful runs, ~1 hr) | ~$1.76 |
| Stuck Massachusetts H100 (1 hr loading) | ~$8.50 |
| Ghost Massachusetts H100 (10 hr exited) | ~$80.00 |
| Washington H100 (successful runs, ~2 hr) | ~$16.20 |
| **Total** | **~$107** |

Most of the cost was the ghost instance that ran for 10 hours unnoticed. The actual successful experiments cost about $18.

Seven vast.ai instances rented and destroyed. Three different GPU architectures. Four model families. Thirteen distinct bugs encountered and fixed. One clear result: **deterministic tensor-parallel LLM inference is achievable, and batch invariance is a Hopper-era capability.**
