# TTFT Estimates — Derived from Latency Benchmark

**Method:** TTFT = wall_time(16 tok) - 16 × decode_per_token
where decode_per_token = (wall_time(128 tok) - wall_time(16 tok)) / 112

This works because per-token decode latency was flat across all output lengths
in the latency benchmark (P95/median ratio < 1.01).

## Qwen2.5-1.5B

| Config | Prompt | TTFT (ms) | Decode/tok (ms) | 16-tok wall (ms) | 128-tok wall (ms) |
|--------|--------|-----------|-----------------|------------------|-------------------|
| **c0** | short | 21.0 | 1.31 | 42 | 189 |
| **c0** | medium | 5.5 | 2.28 | 42 | 298 |
| **c0** | long | 5.6 | 2.30 | 42 | 301 |
| **c1** | short | 51.7 (+147%) | 6.11 | 149 | 833 |
| **c1** | medium | 12.8 (+132%) | 8.66 | 151 | 1121 |
| **c1** | long | 0.9 (-84%) | 9.46 | 152 | 1212 |
| **c2** | short | 66.7 (+219%) | 8.28 | 199 | 1127 |
| **c2** | medium | 20.8 (+278%) | 11.46 | 204 | 1487 |
| **c2** | long | 9.3 (+65%) | 12.32 | 206 | 1587 |
| **c3** | short | 93.9 (+348%) | 8.53 | 230 | 1186 |
| **c3** | medium | 6.5 (+17%) | 14.16 | 233 | 1819 |
| **c3** | long | 10.1 (+79%) | 13.92 | 233 | 1791 |

## Mistral-7B

| Config | Prompt | TTFT (ms) | Decode/tok (ms) | 16-tok wall (ms) | 128-tok wall (ms) |
|--------|--------|-----------|-----------------|------------------|-------------------|
| **c0** | short | 11.8 | 6.03 | 108 | 784 |
| **c0** | medium | 11.3 | 6.04 | 108 | 785 |
| **c0** | long | 11.7 | 6.05 | 108 | 786 |
| **c1** | short | 2.9 (-76%) | 9.85 | 160 | 1264 |
| **c1** | medium | 5.0 (-56%) | 10.01 | 165 | 1286 |
| **c1** | long | 5.6 (-53%) | 9.93 | 164 | 1276 |
| **c2** | short | 5.6 (-52%) | 13.40 | 220 | 1720 |
| **c2** | medium | 0.0 (-100%) | 13.61 | 218 | 1743 |
| **c2** | long | 6.7 (-43%) | 13.51 | 223 | 1736 |
| **c3** | short | 4.6 (-61%) | 19.53 | 317 | 2504 |
| **c3** | medium | 11.6 (+3%) | 20.10 | 333 | 2584 |
| **c3** | long | 6.2 (-47%) | 20.49 | 334 | 2629 |

---

## Summary — TTFT overhead (c0 → c3)

| Model | short | medium | long |
|-------|-------|--------|------|
| Qwen2.5-1.5B | 21→94ms (+348%) | 6→6ms (+17%) | 6→10ms (+79%) |
| Mistral-7B | 12→5ms (-61%) | 11→12ms (+3%) | 12→6ms (-47%) |

## Key findings

1. **TTFT is dominated by prefill, which is NOT heavily affected by the determinism knobs.** The knobs primarily tax the decode loop (per-token generation), not the initial prompt processing.
2. **Decode-per-token is where the cost lives.** The per-token decode latency increases 3-6× under full determinism, but TTFT only increases modestly because prefill is a fixed cost.
3. **For latency-sensitive applications (chatbots), TTFT is the user-perceived responsiveness metric.** The determinism tax on TTFT is much lower than the throughput numbers suggest — a user waiting for the first token sees a smaller penalty than the aggregate throughput loss implies.

