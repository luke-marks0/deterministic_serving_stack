# Memory Wipe Experiment

Proof of Secure Erasure (PoSE-DB) implementation for GPU compute nodes. Fills
all addressable memory (host DRAM, GPU HBM, disk) with cryptographically
verifiable noise, then proves the fill via challenge-response verification.

## What This Is For

When a GPU node is reassigned between tenants, previous data (model weights,
activations, KV cache, user prompts) may persist in DRAM, GPU HBM, or disk.
This experiment demonstrates a protocol that:

1. **Wipes** all accessible memory by filling it with noise
2. **Proves** the wipe happened via 1000 random challenge-response rounds
3. **Restores** the node to a working state (Docker + inference)
4. **Measures** the cost: wipe time, coverage, and downtime

The protocol is based on Bursuc et al., "Software-Based Memory Erasure with
relaxed isolation requirements" (PoSE-DB). See `plan/pose_paper.pdf`.

## What the Experiment Consists Of

A single script runs the full sanitization cycle on a GPU node:

```
baseline inference → teardown → wipe + prove → restore → post-wipe inference
```

### Phases

1. **Prep**: Strip the host to minimum (remove Lambda bloat packages)
2. **Baseline inference**: Pull NGC PyTorch image, load GPT-2, generate a token
3. **Teardown**: Remove container, image, Docker itself, unused CUDA libs
4. **Wipe + prove**: Fill DRAM, HBM, and 500 GiB of disk with noise. Run 1000
   challenge-response rounds where the verifier picks random block indices and
   checks the prover stored the correct noise.
5. **Restore**: Reinstall Docker, pull image
6. **Post-wipe inference**: Generate a token with the same prompt (proof of recovery)

### Two Noise Generation Methods

- **crypto**: AES-256-CTR inline generation. The verifier holds a secret seed
  and generates noise on the fly. Challenges are verified by regenerating the
  block from the seed — no stored copy needed.
- **devurandom**: Pre-generate noise from `/dev/urandom` (16 parallel cores)
  to a file, then stream from the file to each memory region. Challenges read
  from the stored file.

### Benchmark Matrix

The experiment runs across 2 instances × 2 methods = 4 runs, plus additional
optimization experiments (parallel fills, multi-core AES):

| Run | Instance | Method | Wipe Time | DRAM GiB/s | HBM GiB/s | Disk GiB/s |
|-----|----------|--------|-----------|-----------|-----------|-----------|
| 1 | GH200 (ARM) | crypto | 480s | 2.36 | 3.09 | 1.74 |
| 2 | GH200 (ARM) | devurandom | 481s | 2.73 | 4.43 | 1.57 |
| 3 | A10 (x86) | crypto | 1764s | 0.43 | 0.55 | 0.40 |
| 4 | A10 (x86) | devurandom | 1101s | 0.74 | 1.20 | 0.62 |

All runs: 1000/1000 challenges passed, baseline and post-wipe tokens match.

## How to Conduct the Experiment

### Prerequisites

- A Lambda Cloud account with API key in `$LAMBDALABS_API_KEY`
- SSH key named `macbook 2025` registered on Lambda
- `uv` installed locally (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- `jq` installed locally

### 1. Provision an Instance

```bash
cd experiments/memory_wipe

# Launch a GH200 (polls until capacity available)
./scripts/provision_gh200.sh

# Or launch any GPU instance manually via Lambda dashboard
# Supported: gpu_1x_gh200, gpu_1x_a10, gpu_1x_h100_pcie, etc.
```

### 2. Run a Single Benchmark

SSH into the instance and run directly:

```bash
IP=<instance-ip>

# Deploy code
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' \
    --exclude 'reports' --exclude 'memory-sanitization' \
    . ubuntu@$IP:~/memory_wipe/

# Setup
ssh ubuntu@$IP 'cd ~/memory_wipe && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="$HOME/.local/bin:$PATH" && \
    uv sync --extra dev'

# Run with crypto method, 500 GiB disk cap
ssh ubuntu@$IP "sudo bash -c 'cd /home/ubuntu/memory_wipe && \
    nohup .venv/bin/python3 -u scripts/benchmark.py \
    --method crypto --disk-gb 500 \
    > /tmp/benchmark.log 2>&1 &'"

# Monitor
ssh ubuntu@$IP 'sudo tail -f /tmp/benchmark.log'

# Pull report when done
scp ubuntu@$IP:~/memory_wipe/reports/*.json reports/
```

### 3. Run the Full Sweep (All 4 Combinations)

From your local machine:

```bash
# Edit scripts/sweep.py to set instance IPs
python3 scripts/sweep.py
```

This deploys to both instances, runs crypto + devurandom on each, pulls all
reports, and prints a comparison table.

### 4. Available Flags

```
scripts/benchmark.py --method crypto|devurandom  # Noise generation method
                     --disk-gb 500               # Disk wipe cap (GiB)
                     --parallel                  # Fill all regions concurrently
                     --multicore 8               # AES workers per region
```

### 5. View Results

Open the interactive visualization:

```bash
open reports/visualization.html
```

Or inspect individual JSON reports in `reports/`.

## Key Files

| File | Purpose |
|------|---------|
| `scripts/benchmark.py` | Main experiment script (runs on GPU instance) |
| `scripts/sweep.py` | Orchestrate benchmarks across multiple instances |
| `scripts/provision_gh200.sh` | Poll Lambda Cloud and auto-launch GH200 |
| `scripts/probe_limits.py` | Binary search for max allocatable memory |
| `src/pose/noise.py` | AES-256-CTR noise generation (single + multi-core) |
| `src/pose/devurandom.py` | /dev/urandom noise generation + O_DIRECT streaming |
| `src/pose/detect.py` | Auto-detect memory ceilings (NUMA, cudaMemGetInfo) |
| `src/pose/memory/dram.py` | Host DRAM region (mmap + NUMA pinning) |
| `src/pose/memory/hbm.py` | GPU HBM region (raw ctypes to libcudart.so) |
| `src/pose/memory/nvme.py` | Disk region (O_DIRECT) |
| `src/pose/protocol.py` | PoSE-DB protocol orchestrator |
| `src/pose/verifier.py` | Verifier (holds seed, streams noise, checks challenges) |
| `src/pose/prover.py` | Prover (stores blocks, responds to challenges) |
| `src/pose/tracer.py` | Step-level timing trace collector |
| `reports/visualization.html` | Interactive HTML comparison of all runs |
| `docs/findings.md` | Detailed findings and analysis |
| `docs/plans/benchmark.md` | Benchmark sweep implementation plan |
