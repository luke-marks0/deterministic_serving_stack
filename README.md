# Deterministic Serving Stack

Bitwise identical LLM inference across independent servers. Given the same manifest and container, every run produces the same tokens — verified across 3 models, 2 servers, and 8.88 million tokens.

## Results

**157/157 cross-server comparisons match (100%)** across two independent NVIDIA GH200 480GB instances on Lambda Cloud:

| Model | Type | Repeated | Diverse | Tokens |
|-------|------|----------|---------|--------|
| Qwen3-1.7B | Dense transformer | 20/20 match | 34/34 match | 1.6M |
| Qwen3-30B-A3B | Mixture of Experts | 20/20 match | 34/34 match | 2.0M |
| Mistral-7B-Instruct-v0.3 | Dense transformer | 20/20 match | 34/34 match | 2.0M |

Each chunk is 30,000 tokens of greedy decoding (temperature=0). Same container image on both servers, same seed, same config.

## Architecture

```
                              Deterministic Serving Stack
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  ┌──────────┐    ┌──────────┐    ┌──────────────────────────────┐   │
 │  │ Manifest │───>│ Resolver │───>│ Resolved manifest + Lockfile │   │
 │  │ (author) │    │          │    │ (pinned revisions, digests)  │   │
 │  └──────────┘    └──────────┘    └───────────────┬──────────────┘   │
 │                                                  │                   │
 │                                                  v                   │
 │  ┌────────────────────────────────────────────────────────────────┐  │
 │  │                    Nix Container Image                         │  │
 │  │  ┌──────────────────────────────────────────────────────────┐  │  │
 │  │  │ Proxy Server (cmd/server/main.py)                        │  │  │
 │  │  │  POST /manifest ── validate schema                       │  │  │
 │  │  │                  ── verify GPU model, count, driver       │  │  │
 │  │  │                  ── verify model file digests             │  │  │
 │  │  │                  ── start vLLM with manifest settings     │  │  │
 │  │  │  GET  /manifest ── return active config + health          │  │  │
 │  │  │  POST /v1/...   ── proxy to vLLM + capture log           │  │  │
 │  │  └──────────────────────────┬───────────────────────────────┘  │  │
 │  │                             │                                  │  │
 │  │                             v                                  │  │
 │  │  ┌──────────────────────────────────────────────────────────┐  │  │
 │  │  │ vLLM 0.17.1 (VLLM_BATCH_INVARIANT=1, --enforce-eager)  │  │  │
 │  │  │  --model, --revision, --seed, --dtype,                   │  │  │
 │  │  │  --attention-backend, --max-model-len, ...               │  │  │
 │  │  │  (every manifest field passed as CLI flag or env var)     │  │  │
 │  │  └──────────────────────────────────────────────────────────┘  │  │
 │  └────────────────────────────────────────────────────────────────┘  │
 │                                                                      │
 │  ┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────────┐  │
 │  │  Runner  │───>│ Capture  │───>│ Run Bundle│───>│   Verifier   │  │
 │  │(tokens,  │    │(request/ │    │(observ-   │    │(compare two  │  │
 │  │ logits,  │    │ response │    │ ables,    │    │ bundles via  │  │
 │  │ frames)  │    │ logging) │    │ frames,   │    │ comparison   │  │
 │  │          │    │          │    │ provenance│    │ config)      │  │
 │  └──────────┘    └──────────┘    └───────────┘    └──────────────┘  │
 └──────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Pull and run the container

```bash
# One-time Docker setup (see "Running the Container" below for details)
sudo nvidia-ctk runtime configure --runtime=docker
sudo tee /etc/docker/daemon.json <<'EOF'
{"runtimes":{"nvidia":{"args":[],"path":"nvidia-container-runtime"}},"default-runtime":"nvidia"}
EOF
sudo systemctl restart docker

# Pull the image
docker pull ghcr.io/derpyplops/deterministic-serving-runtime:latest

# Start the server
docker run -d --name vllm-server \
  --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VLLM_BATCH_INVARIANT=1 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e HOME=/tmp \
  -p 8000:8000 \
  ghcr.io/derpyplops/deterministic-serving-runtime:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --host 0.0.0.0 --port 8000 --seed 42 \
    --enforce-eager --attention-backend TRITON_ATTN \
    --max-model-len 4096

# Wait for it
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 3; done
```

### 2. Send a request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "What is deterministic computation?"}],
    "max_tokens": 100,
    "temperature": 0,
    "seed": 42
  }'
```

### 3. Verify determinism (run the same request twice)

```bash
for i in 1 2; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"temperature":0,"seed":42}' \
    | python3 -c "import sys,json,hashlib; c=json.load(sys.stdin)['choices'][0]['message']['content']; print(f'Run {'"$i"'}: {hashlib.sha256(c.encode()).hexdigest()[:16]}')"
done
# Both hashes will be identical
```

### 4. Run the synthetic pipeline (no GPU needed)

```bash
tmp=$(mktemp -d)
python3 cmd/resolver/main.py --manifest manifests/qwen3-1.7b.manifest.json \
  --lockfile-out $tmp/lock.json --resolve-hf
python3 cmd/builder/main.py --lockfile $tmp/lock.json --lockfile-out $tmp/built.json
python3 cmd/runner/main.py --manifest manifests/qwen3-1.7b.manifest.json \
  --lockfile $tmp/built.json --out-dir $tmp/run
# Produces a run bundle with tokens, logits, and deterministic network frames
```

## How It Works

**Manifest** declares the full workload: model (pinned to HF commit SHA), runtime config (seed, dtype, attention backend, batch invariance), hardware requirements, requests, and comparison criteria.

**Resolver** pins everything to immutable references: resolves HF revisions, enumerates model files with per-file SHA256 digests, produces a lockfile.

**Nix container** pins the entire software stack: vLLM, PyTorch, CUDA toolkit, Triton, all Python deps. Same flake = same container = same behavior on any machine.

**Server** validates the manifest against the runtime (GPU model/count, driver version, CUDA version, model file digests), then starts vLLM with every manifest field passed as a CLI flag or env var.

**Runner** generates a run bundle containing tokens, logits, and deterministic L2 network frames (constructed by a simulated TCP/IP stack from the inference output).

**Verifier** compares two run bundles using the manifest's comparison config (exact match for tokens, tolerance for logits, SHA256 for network egress).

## What Makes It Deterministic

| Layer | How |
|-------|-----|
| **Software** | Hermetic Nix container — identical binary on every machine |
| **Model weights** | HF commit SHA pinned, per-file SHA256 verified before serving |
| **CUDA/cuBLAS** | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `VLLM_BATCH_INVARIANT=1` |
| **Attention** | `--enforce-eager` (no CUDA graphs), fixed attention backend |
| **Scheduling** | Greedy decoding (temperature=0), fixed seed |
| **Network frames** | Simulated TCP/IP stack with fixed MSS segmentation, software checksums, no offloads |

## Repository Structure

```
cmd/
  server/         Proxy server with POST/GET /manifest endpoint
  resolver/       Manifest + HF resolution -> lockfile
  builder/        Lockfile -> lockfile with runtime_closure_digest
  runner/         Manifest + lockfile -> run bundle (synthetic or vLLM)
  capture/        Server capture log -> run bundle
  verifier/       Compare two run bundles -> conformance report
pkg/
  manifest/       Pydantic manifest model (typed validation)
  common/         Canonical JSON, SHA256, schema validation, HF resolution
  networkdet/     Deterministic L2 frame construction (sim TCP/IP stack)
  hardware/       GPU probing and conformance
schemas/          JSON Schema contracts (manifest, lockfile, run_bundle, verify_report)
manifests/        Model-specific manifests (Qwen3-1.7B)
experiments/      Determinism experiment scripts (tiered: smoke/medium/full)
docs/             Architecture diagrams, field reports, memos
```

## Running the Container

### Prerequisites
- NVIDIA GPU with compute capability >= 9.0 (H100, GH200, etc.)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- The NVIDIA runtime must be the **default Docker runtime**

### Docker + NVIDIA setup

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo tee /etc/docker/daemon.json <<'EOF'
{"runtimes":{"nvidia":{"args":[],"path":"nvidia-container-runtime"}},"default-runtime":"nvidia"}
EOF
sudo systemctl restart docker
```

### Start the server

```bash
docker run -d --name vllm-server \
  --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e VLLM_BATCH_INVARIANT=1 \
  -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  -e HOME=/tmp \
  -p 8000:8000 \
  ghcr.io/derpyplops/deterministic-serving-runtime:latest \
  python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-1.7B \
    --host 0.0.0.0 --port 8000 --seed 42 \
    --enforce-eager --attention-backend TRITON_ATTN \
    --max-model-len 4096
```

### Notes
- Replace `Qwen/Qwen3-1.7B` with any supported model
- Set `-e HF_TOKEN=<token>` for gated models
- `--privileged` is required on Lambda Cloud GH200 instances
- `NVIDIA_DRIVER_CAPABILITIES=all` triggers driver lib injection
- Podman 3.x doesn't work — use Docker

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `Failed to infer device type` | Add `--privileged -e NVIDIA_DRIVER_CAPABILITIES=all` |
| `No CUDA GPUs are available` | Add `--privileged` |
| `Can't initialize NVML` | Set `"default-runtime": "nvidia"` in daemon.json |
| `Failed to find C compiler` | Container must include gcc (current image does) |
| `GLIBC_2.38 not found` | Don't set `LD_LIBRARY_PATH` to host system paths |

## Building from Source

```bash
# Build the hermetic runtime closure
nix build .#closure

# Build the OCI image
nix build .#oci

# Load into Docker
docker load < result
```

## CI Gates

| Gate | What it runs | Command |
|------|-------------|---------|
| PR | lint + schema + unit/integration | `make ci-pr` |
| Main | + e2e + determinism + nix closure | `make ci-main` |
| Nightly | + chaos + long-run | `make ci-nightly` |
| Release | + release contracts | `make ci-release` |
