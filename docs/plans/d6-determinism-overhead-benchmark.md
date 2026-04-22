# D6 Determinism Overhead Benchmark

**Goal:** measure the throughput and latency penalty of vLLM's determinism knobs relative to default (non-deterministic) vLLM inference.

---

## What we're measuring

Five configurations, each adding one knob on top of the previous, so we can attribute overhead per-knob:

| Config | Label | Settings | What it isolates |
|---|---|---|---|
| **C0** | `baseline` | vLLM defaults (CUDAGraphs on, compile on, batch_invariant off, no workspace config) | Production-mode baseline |
| **C1** | `+eager` | C0 + `enforce_eager=True` | Cost of disabling CUDAGraphs + torch.compile |
| **C2** | `+cublas` | C1 + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Cost of deterministic cublas kernel selection |
| **C3** | `+batch_inv` | C2 + `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` | Cost of batch-invariant scheduling + pinned attention |
| **C4** | `+all_reduce` | C3 + `disable_custom_all_reduce=True` | Cost of NCCL-only all-reduce (only measurable on TP>1) |

C3 is the full single-GPU deterministic config. C4 adds the multi-GPU overhead.

### Why incremental, not just C0 vs C3

If C3 is 40% slower than C0, you need to know whether it's 35% from `enforce_eager` and 5% from `batch_invariant`, or 10% from each. The incremental design lets you read the overhead curve and decide where your cost/determinism tradeoff lives.

---

## Workloads

Three axes: **model size**, **batch size**, and **sequence length**. We vary one at a time to see where each knob bites hardest.

### Models

| Model | Params | Type | GPUs needed | Why |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 1.5B | Dense | 1 | Fast iteration, isolates per-step overhead without memory pressure |
| Mistral-7B-Instruct-v0.3 | 7B | Dense | 1 | Realistic small-prod model, fits on one H100 with room for large batches |

Qwen2.5-1.5B is the fast inner loop — run all configs × all workloads here first. Mistral-7B confirms the pattern at production scale on 1 GPU. Both ungated, single H100, cheap sweep.

### Batch sizes

`1, 4, 16, 64, 128`

- Batch=1: latency-sensitive path, CUDAGraph overhead dominates
- Batch=64: the `max_num_seqs` default in our manifests
- Batch=128: scheduler stress test for batch_invariant padding

### Sequence lengths (output tokens)

`16, 128, 512, 2048`

- Short (16): init-dominated, CUDAGraph launch savings are proportionally largest
- Medium (128, 512): the sweet spot where generation throughput is the metric
- Long (2048): KV cache pressure, prefill/decode ratio shifts

### Fixed parameters across all runs

- `temperature=0` (greedy — same as our determinism tests)
- `seed=42`
- Prompts: reuse the Tier 2 corpus from `scripts/d6/prompts.py` (100 prompts), subsample to match batch size

---

## Metrics

Per run, collect:

1. **Throughput** (tokens/second, output only) — `total_output_tokens / wall_time_generation`
2. **Time to first token (TTFT)** — from `llm.generate()` call to first token callback (if available in offline mode; otherwise approximate from engine logs)
3. **Wall time** — total `llm.generate()` call duration
4. **Peak GPU memory** — `torch.cuda.max_memory_allocated()`

Report as:

- **Absolute** values per config
- **Relative overhead** = `(wall_time_Cn - wall_time_C0) / wall_time_C0` as a percentage
- **Per-knob delta** = `(wall_time_Cn - wall_time_C(n-1)) / wall_time_C0` as a percentage

---

## Execution

### Hardware

**Option A (cheap, 1-GPU):** Single vast.ai H100 SXM (~$1.40/hr). Run Qwen3-0.6B and Mistral-7B across all configs × batch sizes × seq lengths. Skip DBRX.

**Option B (full):** 4× H100 SXM on vast.ai (~$6.67/hr). Run all three models. DBRX only tested at the batch sizes that fit in 4×80GB.

Recommend starting with Option A. The per-knob overhead pattern should be visible on 1 GPU; DBRX adds the `+all_reduce` data point but at 5× the cost.

### Estimated run count

Option A:
- 5 configs × 5 batch sizes × 4 seq lengths × 2 models = **200 runs**
- Each run: ~5–60 seconds (Qwen) to ~10–180 seconds (Mistral-7B)
- Warmup run per config (discard first run to avoid one-time costs)
- Total wall time estimate: **~2–3 hours**
- Cost: ~$3–5

Option B adds:
- 5 configs × 3 batch sizes × 3 seq lengths × 1 model (DBRX) = **45 runs**
- Each run: ~60–600 seconds
- Total wall time: **+3–5 hours**
- Cost: ~$25–35

### Script design

A single script `scripts/d6/benchmark_determinism_overhead.py` that:

1. Takes `--model`, `--config` (c0/c1/c2/c3/c4), `--batch-size`, `--max-tokens` as args
2. Builds the `LLM(...)` kwargs per config
3. Generates `batch_size` prompts from the Tier 2 corpus (cycling if batch > 100)
4. Runs a warmup pass (discarded)
5. Runs the timed pass, records wall time + token counts
6. Prints a JSON line: `{"config": "c2", "model": "...", "batch": 16, "max_tokens": 128, "wall_s": 12.3, "tokens": 2048, "tok_per_s": 166.5, "peak_mem_gb": 14.2}`

A wrapper `scripts/d6/run_overhead_sweep.py` iterates all combinations, writes results to a JSONL file, and generates a summary table.

### Output format

```
results/overhead_benchmark_YYYYMMDD.jsonl   # one JSON object per run
results/overhead_benchmark_YYYYMMDD.md      # auto-generated markdown table
```

The markdown table looks like:

```
## Qwen3-0.6B, batch=16

| Config    | max_tok=16 | max_tok=128 | max_tok=512 | max_tok=2048 |
|-----------|-----------|-------------|-------------|--------------|
| baseline  | 1234 t/s  | 1180 t/s    | 1150 t/s    | 1120 t/s     |
| +eager    | 890 t/s (-28%) | ...    | ...         | ...          |
| +cublas   | 880 t/s (-1%) | ...     | ...         | ...          |
| +batch_inv| 850 t/s (-3%) | ...     | ...         | ...          |
```

---

## What we expect to find (hypotheses to test)

1. **`enforce_eager` is the biggest single cost** — CUDAGraphs eliminate kernel launch overhead, and torch.compile fuses ops. Disabling both should be a 20–40% throughput hit on short sequences, narrowing to 5–15% on long sequences (where compute dominates launch overhead).

2. **`CUBLAS_WORKSPACE_CONFIG` is cheap** (<5%) — it restricts cublas to deterministic algorithms but most modern GEMM kernels on H100 are already deterministic-by-default at bf16.

3. **`batch_invariant` cost scales with batch size** — at batch=1 it's zero (no batching to make invariant). At batch=64+ it adds padding and scheduling constraints. Expect 5–15% overhead at high batch sizes, near-zero at batch=1.

4. **`disable_custom_all_reduce` is small for TP=4** — NCCL's ring all-reduce is already close to the custom implementation's performance on H100 NVLink. Expect <5% difference.

5. **MoE (DBRX) pays more for `batch_invariant` than dense (Mistral)** — the expert routing sort/dispatch path has more batch-order-dependent operations to pin.

---

## Bail-outs

- If `enforce_eager` alone is >50% overhead at all sequence lengths, that's a finding worth reporting immediately — it means the determinism tax is dominated by a single knob and the others are noise.
- If any config produces different token IDs from C3 (the full deterministic config) at `temperature=0`, that's a determinism bug — stop benchmarking and investigate.
- If vast.ai capacity for H100 is zero, fall back to A100 SXM (80GB) — the overhead ratios should be similar even if absolute throughput differs.

---

## Deliverables

1. `scripts/d6/benchmark_determinism_overhead.py` — the benchmark script
2. `scripts/d6/run_overhead_sweep.py` — the sweep runner
3. `results/overhead_benchmark_YYYYMMDD.jsonl` — raw data
4. `results/overhead_benchmark_YYYYMMDD.md` — summary tables + per-knob breakdown
5. A section in the D6 report or a standalone doc: "Determinism overhead characterization"
