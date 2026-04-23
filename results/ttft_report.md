# TTFT Benchmark — Time To First Token

TTFT = wall_time(max_tokens=1) - one_decode_step

where decode_step = median(wall_time(max_tokens=2)) - median(wall_time(max_tokens=1))

Each measurement: batch_size=1, 10 reps after 3 warmup.

---

# Qwen2.5-1.5B-Instruct

| Config | short prompt | medium prompt | long prompt |
|--------|-------------|---------------|-------------|
| **c0** | 4.5ms | 5.4ms | 6.3ms |
| **c1** | 2.1ms (-53%) | 2.5ms (-53%) | 2.8ms (-55%) |
| **c2** | 1.4ms (-68%) | 3.0ms (-45%) | 1.0ms (-85%) |
| **c3** | 2.9ms (-37%) | 3.3ms (-38%) | 3.4ms (-47%) |

**Decode step (ms/tok) and t1 (prefill + 1 decode):**

| Config | short t1 | short decode | medium t1 | medium decode | long t1 | long decode |
|--------|----------|-------------|-----------|--------------|---------|------------|
| **c0** | 7.2ms | 2.7ms | 7.2ms | 1.8ms | 7.8ms | 1.5ms |
| **c1** | 11.6ms | 9.4ms | 11.6ms | 9.1ms | 12.0ms | 9.2ms |
| **c2** | 14.1ms | 12.6ms | 15.8ms | 12.8ms | 15.2ms | 14.2ms |
| **c3** | 17.0ms | 14.1ms | 17.1ms | 13.8ms | 17.6ms | 14.3ms |

---

# Mistral-7B-Instruct-v0.3

| Config | short prompt | medium prompt | long prompt |
|--------|-------------|---------------|-------------|
| **c0** | 3.9ms | 2.0ms | 3.9ms |
| **c1** | 2.4ms (-39%) | 2.6ms (+27%) | 3.1ms (-21%) |
| **c2** | 2.0ms (-47%) | 1.2ms (-42%) | 3.2ms (-18%) |
| **c3** | 3.4ms (-13%) | 1.8ms (-13%) | 3.4ms (-13%) |

**Decode step (ms/tok) and t1 (prefill + 1 decode):**

| Config | short t1 | short decode | medium t1 | medium decode | long t1 | long decode |
|--------|----------|-------------|-----------|--------------|---------|------------|
| **c0** | 9.5ms | 5.6ms | 8.3ms | 6.2ms | 9.8ms | 6.0ms |
| **c1** | 12.5ms | 10.2ms | 12.6ms | 10.0ms | 13.2ms | 10.1ms |
| **c2** | 16.1ms | 14.1ms | 15.7ms | 14.5ms | 16.9ms | 13.8ms |
| **c3** | 22.9ms | 19.5ms | 22.5ms | 20.8ms | 23.9ms | 20.5ms |

---

# Summary — TTFT overhead (c0 → c3)

| Model | short | medium | long |
|-------|-------|--------|------|
| Qwen2.5-1.5B-Instruct | 4.5→2.9ms (-37%) | 5.4→3.3ms (-38%) | 6.3→3.4ms (-47%) |
| Mistral-7B-Instruct-v0.3 | 3.9→3.4ms (-13%) | 2.0→1.8ms (-13%) | 3.9→3.4ms (-13%) |

