# Determinism Overhead Benchmark — Full Results

**Hardware:** 1× NVIDIA H100 80GB SXM (vast.ai)
**Software:** vLLM 0.17.1, PyTorch 2.10, CUDA, bf16
**Date:** 2026-04-17

## Configs (cumulative — each adds one flag on top of the previous)

| Config | Flags active | What this step adds |
|--------|-------------|-------------------|
| **c0** | CUDAGraphs, torch.compile, auto cublas, auto attention | *(baseline — all optimizations on)* |
| **c1** | c0 + `enforce_eager=True` | Disables CUDAGraphs and torch.compile |
| **c2** | c1 + `CUBLAS_WORKSPACE_CONFIG=:4096:8` | Forces deterministic cublas kernel selection |
| **c3** | c2 + `VLLM_BATCH_INVARIANT=1` + `attention_backend=FLASH_ATTN` | Batch-order-invariant scheduling + pinned attention backend |

c3 is the full deterministic config used in the D6 experiment.

---

# Qwen2.5-1.5B

## batch_size=1

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 363 | 437 | 389 | 438 |
| **c1** + enforce_eager | 129 | 133 | 137 | 136 |
| **c2** + cublas_workspace | 94 | 93 | 93 | 88 |
| **c3** + batch_invariant + flash_attn | 78 | 69 | 70 | 72 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -64.3% | -69.6% | -64.7% | -68.9% |
| **c2** + cublas_workspace | -74.0% | -78.8% | -76.0% | -80.0% |
| **c3** + batch_invariant + flash_attn | -78.5% | -84.3% | -82.0% | -83.7% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -64.3% | -69.6% | -64.7% | -68.9% |
| Forces deterministic cublas kernel selection | -9.7% | -9.2% | -11.3% | -11.1% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -4.5% | -5.5% | -6.0% | -3.7% |

## batch_size=4

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 1,213 | 1,710 | 1,228 | 626 |
| **c1** + enforce_eager | 408 | 508 | 356 | 190 |
| **c2** + cublas_workspace | 350 | 367 | 250 | 129 |
| **c3** + batch_invariant + flash_attn | 273 | 287 | 173 | 106 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -66.4% | -70.3% | -71.0% | -69.6% |
| **c2** + cublas_workspace | -71.2% | -78.6% | -79.6% | -79.4% |
| **c3** + batch_invariant + flash_attn | -77.5% | -83.2% | -85.9% | -83.1% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -66.4% | -70.3% | -71.0% | -69.6% |
| Forces deterministic cublas kernel selection | -4.8% | -8.3% | -8.6% | -9.8% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -6.3% | -4.6% | -6.3% | -3.7% |

## batch_size=16

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 4,036 | 6,299 | 4,056 | 2,336 |
| **c1** + enforce_eager | 1,400 | 1,969 | 1,175 | 502 |
| **c2** + cublas_workspace | 1,236 | 1,437 | 771 | 340 |
| **c3** + batch_invariant + flash_attn | 992 | 1,122 | 583 | 255 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -65.3% | -68.7% | -71.0% | -78.5% |
| **c2** + cublas_workspace | -69.4% | -77.2% | -81.0% | -85.5% |
| **c3** + batch_invariant + flash_attn | -75.4% | -82.2% | -85.6% | -89.1% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -65.3% | -68.7% | -71.0% | -78.5% |
| Forces deterministic cublas kernel selection | -4.1% | -8.4% | -10.0% | -6.9% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -6.1% | -5.0% | -4.6% | -3.6% |

## batch_size=64

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 11,374 | 20,514 | 14,226 | 8,253 |
| **c1** + enforce_eager | 4,786 | 7,209 | 4,799 | 3,011 |
| **c2** + cublas_workspace | 3,871 | 4,996 | 3,354 | 1,963 |
| **c3** + batch_invariant + flash_attn | 3,441 | 3,986 | 2,493 | 1,324 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -57.9% | -64.9% | -66.3% | -63.5% |
| **c2** + cublas_workspace | -66.0% | -75.6% | -76.4% | -76.2% |
| **c3** + batch_invariant + flash_attn | -69.7% | -80.6% | -82.5% | -84.0% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -57.9% | -64.9% | -66.3% | -63.5% |
| Forces deterministic cublas kernel selection | -8.1% | -10.8% | -10.2% | -12.7% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -3.8% | -4.9% | -6.1% | -7.8% |

## batch_size=128

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 12,329 | 32,412 | 25,227 | 15,026 |
| **c1** + enforce_eager | 7,191 | 11,002 | 7,984 | 5,008 |
| **c2** + cublas_workspace | 6,062 | 8,061 | 5,589 | 3,883 |
| **c3** + batch_invariant + flash_attn | 6,052 | 7,013 | 4,974 | 2,994 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -41.7% | -66.1% | -68.4% | -66.7% |
| **c2** + cublas_workspace | -50.8% | -75.1% | -77.8% | -74.2% |
| **c3** + batch_invariant + flash_attn | -50.9% | -78.4% | -80.3% | -80.1% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -41.7% | -66.1% | -68.4% | -66.7% |
| Forces deterministic cublas kernel selection | -9.2% | -9.1% | -9.5% | -7.5% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -0.1% | -3.2% | -2.4% | -5.9% |

---

# Mistral-7B

## batch_size=1

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 140 | 159 | 160 | 161 |
| **c1** + enforce_eager | 113 | 114 | 110 | 118 |
| **c2** + cublas_workspace | 79 | 86 | 72 | 84 |
| **c3** + batch_invariant + flash_attn | 50 | 51 | 50 | 51 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -19.1% | -28.7% | -31.4% | -26.8% |
| **c2** + cublas_workspace | -43.9% | -46.0% | -54.8% | -47.9% |
| **c3** + batch_invariant + flash_attn | -64.1% | -67.8% | -68.5% | -68.1% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -19.1% | -28.7% | -31.4% | -26.8% |
| Forces deterministic cublas kernel selection | -24.8% | -17.3% | -23.4% | -21.2% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -20.2% | -21.8% | -13.8% | -20.2% |

## batch_size=4

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 530 | 620 | 412 | 412 |
| **c1** + enforce_eager | 400 | 468 | 352 | 331 |
| **c2** + cublas_workspace | 278 | 338 | 244 | 234 |
| **c3** + batch_invariant + flash_attn | 184 | 200 | 152 | 154 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -24.5% | -24.5% | -14.7% | -19.8% |
| **c2** + cublas_workspace | -47.6% | -45.4% | -40.8% | -43.3% |
| **c3** + batch_invariant + flash_attn | -65.2% | -67.7% | -63.2% | -62.6% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -24.5% | -24.5% | -14.7% | -19.8% |
| Forces deterministic cublas kernel selection | -23.0% | -21.0% | -26.0% | -23.5% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -17.7% | -22.3% | -22.5% | -19.4% |

## batch_size=16

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 1,995 | 2,390 | 1,557 | 1,399 |
| **c1** + enforce_eager | 1,348 | 1,780 | 1,195 | 990 |
| **c2** + cublas_workspace | 1,105 | 1,245 | 828 | 688 |
| **c3** + batch_invariant + flash_attn | 733 | 805 | 549 | 474 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -32.4% | -25.5% | -23.3% | -29.3% |
| **c2** + cublas_workspace | -44.6% | -47.9% | -46.8% | -50.8% |
| **c3** + batch_invariant + flash_attn | -63.3% | -66.3% | -64.7% | -66.1% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -32.4% | -25.5% | -23.3% | -29.3% |
| Forces deterministic cublas kernel selection | -12.2% | -22.4% | -23.6% | -21.5% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -18.7% | -18.4% | -17.9% | -15.4% |

## batch_size=64

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 5,222 | 8,404 | 5,718 | 1,911 |
| **c1** + enforce_eager | 4,092 | 6,515 | 4,216 | 1,416 |
| **c2** + cublas_workspace | 3,420 | 4,606 | 3,046 | 941 |
| **c3** + batch_invariant + flash_attn | 2,572 | 3,799 | 2,301 | 630 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -21.6% | -22.5% | -26.3% | -25.9% |
| **c2** + cublas_workspace | -34.5% | -45.2% | -46.7% | -50.8% |
| **c3** + batch_invariant + flash_attn | -50.7% | -54.8% | -59.8% | -67.0% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -21.6% | -22.5% | -26.3% | -25.9% |
| Forces deterministic cublas kernel selection | -12.9% | -22.7% | -20.5% | -24.8% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -16.2% | -9.6% | -13.0% | -16.3% |

## batch_size=128

### Throughput (tokens/sec)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | 9,300 | 14,008 | 10,388 | 3,908 |
| **c1** + enforce_eager | 6,293 | 10,439 | 6,919 | 3,283 |
| **c2** + cublas_workspace | 5,978 | 7,925 | 5,709 | 2,132 |
| **c3** + batch_invariant + flash_attn | 5,620 | 7,538 | 4,472 | 1,447 |

### Overhead vs baseline (c0)

| Config | 16 tok | 128 tok | 512 tok | 2048 tok |
|--------|---------|---------|---------|---------|
| **c0** baseline (all optimizations) | — | — | — | — |
| **c1** + enforce_eager | -32.3% | -25.5% | -33.4% | -16.0% |
| **c2** + cublas_workspace | -35.7% | -43.4% | -45.0% | -45.4% |
| **c3** + batch_invariant + flash_attn | -39.6% | -46.2% | -56.9% | -63.0% |

### Incremental cost of each flag (% of baseline lost, vs previous config)

| Flag added | 16 tok | 128 tok | 512 tok | 2048 tok |
|------------|---------|---------|---------|---------|
| Disables CUDAGraphs and torch.compile | -32.3% | -25.5% | -33.4% | -16.0% |
| Forces deterministic cublas kernel selection | -3.4% | -17.9% | -11.6% | -29.4% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | -3.8% | -2.8% | -11.9% | -17.5% |

---

# Summary

## Total determinism tax (c0 baseline → c3 full deterministic)

| Model | Batch | 16 tok | 128 tok | 512 tok | 2048 tok |
|-------|-------|--------|---------|---------|----------|
| Qwen2.5-1.5B | 1 | -78.5% | -84.3% | -82.0% | -83.7% |
| Qwen2.5-1.5B | 4 | -77.5% | -83.2% | -85.9% | -83.1% |
| Qwen2.5-1.5B | 16 | -75.4% | -82.2% | -85.6% | -89.1% |
| Qwen2.5-1.5B | 64 | -69.7% | -80.6% | -82.5% | -84.0% |
| Qwen2.5-1.5B | 128 | -50.9% | -78.4% | -80.3% | -80.1% |
| Mistral-7B | 1 | -64.1% | -67.8% | -68.5% | -68.1% |
| Mistral-7B | 4 | -65.2% | -67.7% | -63.2% | -62.6% |
| Mistral-7B | 16 | -63.3% | -66.3% | -64.7% | -66.1% |
| Mistral-7B | 64 | -50.7% | -54.8% | -59.8% | -67.0% |
| Mistral-7B | 128 | -39.6% | -46.2% | -56.9% | -63.0% |

## Per-flag incremental cost (averaged across all batch sizes and sequence lengths)

| Flag | Qwen2.5-1.5B | Mistral-7B |
|------|-------------|-----------|
| Disables CUDAGraphs and torch.compile | 66.2% | 25.2% |
| Forces deterministic cublas kernel selection | 9.0% | 20.2% |
| Batch-order-invariant scheduling, pinned FLASH_ATTN backend | 4.7% | 16.0% |

## Key takeaways

1. **The full determinism stack (c3) costs 40–89% throughput** depending on model size, batch size, and sequence length.

2. **Small models pay more than large models.** Qwen 1.5B loses ~80% of throughput; Mistral 7B loses ~61%. Small models are kernel-launch-bound, so disabling CUDAGraphs (which eliminate launch overhead) is devastating.

3. **The overhead distributes differently by model size:**
   - **Qwen 1.5B (small):** `enforce_eager` is ~66% of the total hit. `cublas_workspace` and `batch_invariant` together add only ~14%. CUDAGraphs are everything for small models.
   - **Mistral 7B (medium):** the cost is more evenly spread — `enforce_eager` ~25%, `cublas_workspace` ~20%, `batch_invariant` ~16%. All three knobs matter.

4. **`cublas_workspace` is NOT cheap** — contrary to the common assumption that deterministic cublas is nearly free on modern hardware, it costs 9–20% of baseline throughput. The H100's auto-tuned cublas kernels at bf16 are measurably faster than the deterministic-only subset.

5. **`batch_invariant` cost scales with batch size for the 7B model** — at batch=1 it's ~20% (scheduling overhead even with no batching), but the overhead is consistent across batch sizes rather than growing. For the 1.5B model it's only 3–6% because kernel launch dominates everything else.

6. **If you want partial determinism at lower cost:** `enforce_eager` alone (c1) gives per-run reproducibility without batch-order invariance. On Mistral 7B it's a ~25% hit vs ~61% for the full stack. But it does NOT give you batch-order invariance — you need c3 for that, which is the actual D6 property.
