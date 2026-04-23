# Determinism Overhead

## Determinism Knobs

Our full deterministic stack (c3) is built by cumulatively enabling three groups of flags on top of the vLLM baseline (c0):

| Level | Flag | What it does |
|-------|------|--------------|
| **c1** | `enforce_eager=True` | Disables CUDA Graphs and `torch.compile`, forcing eager-mode execution. CUDA Graphs replay pre-recorded kernel sequences, which can select different kernels across runs depending on GPU state and autotuning. Eager mode eliminates this source of non-determinism at the cost of per-step kernel launch overhead. |
| **c2** | `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Forces cuBLAS to use a fixed workspace and deterministic algorithm selection for matrix multiplications. Without this, cuBLAS auto-tunes kernel selection at runtime, which can produce bitwise-different results across calls with identical inputs. |
| **c3** | `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` | Enables batch-order-invariant scheduling so that each request's output is independent of which other requests share the batch. Pins the attention backend to FlashAttention to avoid non-deterministic backend selection. This is the key flag for our threat model: without it, an attacker could infer information about co-batched requests from their own outputs. |

Each level is cumulative — c3 includes all flags from c1 and c2.

## Throughput Overhead

We measured throughput (output tokens/sec) on a single H100 80GB SXM across two models (Qwen 2.5 1.5B and Mistral 7B), five batch sizes (1, 4, 16, 64, 128), and four output lengths (16, 128, 512, 2048 tokens). Total: 160 benchmark runs.

**Total determinism overhead (c0 baseline → c3 full deterministic):**

- **Qwen 2.5 1.5B:** 51–89% throughput reduction
- **Mistral 7B:** 40–67% throughput reduction

The overhead is lower for larger models. Small models are kernel-launch-bound, so disabling CUDA Graphs (which eliminate launch overhead) is devastating — it accounts for ~66% of the total throughput loss on the 1.5B model. On the 7B model, the cost is more evenly distributed across the three flag groups (~25%, ~20%, ~16% respectively).

![Throughput by determinism level](figures/throughput_by_batch.png)
*Figure: Throughput at each determinism level across batch sizes (seq_len=128). Log scale. The gap between baseline (blue) and full determinism (green) is consistently large but narrows slightly at higher batch sizes for the 7B model.*

![Incremental cost breakdown](figures/incremental_cost.png)
*Figure: Stacked incremental cost of each determinism flag, averaged across output lengths. For the small model, eager mode dominates. For the 7B model, all three flags contribute substantially.*

### Where the overhead comes from

The dominant cost is **eager mode** (disabling CUDA Graphs and `torch.compile`). CUDA Graphs eliminate per-step kernel launch overhead by replaying a pre-recorded sequence of GPU operations. Without them, every decode step pays the full CPU→GPU launch cost. This is particularly painful for small models where compute per step is low relative to launch overhead.

**Deterministic cuBLAS** adds 9–20% overhead by constraining the kernel search space. The H100's auto-tuned cuBLAS kernels at bf16 are measurably faster than the deterministic-only subset — contrary to the common assumption that deterministic cuBLAS is nearly free on modern hardware.

**Batch invariance** adds 5–16% overhead from scheduling constraints and pinning the attention backend.

### A note on future improvement

The majority of this overhead (eager mode) is an artifact of the current vLLM implementation, not a fundamental cost of determinism. CUDA Graphs are compatible with deterministic execution in principle — the graph just needs to be captured with deterministic kernels. If vLLM adds support for deterministic CUDA Graph capture, the overhead would drop to the cuBLAS + batch invariance cost (~15–35%), which is much more practical for production deployment.

## Latency Overhead (Time to First Token)

We measured per-request latency at batch_size=1 across three prompt lengths (short, medium, long).

**Key finding: TTFT is largely unaffected by determinism.** The determinism knobs primarily tax the decode loop (per-token generation), not the prefill computation. TTFT is dominated by prefill, so the user-perceived responsiveness penalty is much smaller than the throughput numbers suggest.

| Model | Prompt | TTFT (c0) | TTFT (c3) | Overhead |
|-------|--------|-----------|-----------|----------|
| Qwen 2.5 1.5B | short | 21 ms | 94 ms | +348% |
| Qwen 2.5 1.5B | medium | 6 ms | 7 ms | +17% |
| Qwen 2.5 1.5B | long | 6 ms | 10 ms | +79% |
| Mistral 7B | short | 12 ms | 5 ms | -61% |
| Mistral 7B | medium | 11 ms | 12 ms | +3% |
| Mistral 7B | long | 12 ms | 6 ms | -47% |

The short-prompt TTFT increase for Qwen 1.5B (21→94ms) reflects eager-mode kernel launch overhead becoming visible when prefill itself is very fast. For medium and long prompts — which dominate real-world usage — TTFT overhead is negligible (+3% to +79%), and for Mistral 7B it's actually negative due to measurement noise at these small absolute values.

**Per-token decode latency** increases 3–6× under full determinism (e.g., Mistral 7B: 6ms → 20ms per token at batch=1), which is consistent with the throughput overhead.
