# Deterministic Serving Stack

Bitwise identical LLM inference across independent servers. Given the same manifest and container, every run produces the same tokens — verified across 3 models, 2 servers, and 8.88 million tokens.

## Results

**157/157 cross-server comparisons match (100%)** across two independent NVIDIA GH200 480GB instances on Lambda Cloud:

| Model | Type | Repeated | Diverse | Tokens | Manifest |
|-------|------|----------|---------|--------|----------|
| Qwen3-1.7B | Dense transformer | 20/20 match | 34/34 match | 1.6M | [`qwen3-1.7b.manifest.json`](manifests/qwen3-1.7b.manifest.json) |
| Qwen3-30B-A3B | Mixture of Experts | 20/20 match | 34/34 match | 2.0M | [`qwen3-30b-moe-tp4.manifest.json`](manifests/qwen3-30b-moe-tp4.manifest.json) |
| Mistral-7B-Instruct-v0.3 | Dense transformer | 20/20 match | 34/34 match | 2.0M | [`mistral-large2-tp4.manifest.json`](manifests/mistral-large2-tp4.manifest.json) |

Each chunk is 30,000 tokens of greedy decoding (temperature=0). Same container image on both servers, same seed, same config.

## For Reviewers

Three tiers, each self-contained. Pick the level of hardware you have access to.

### Tier 0 — No GPU (2 min, laptop)

Reproduces the deterministic network-frame construction claim on any machine. Proves the L2 frame digests are reproducible byte-for-byte without any GPU or model.

```bash
git clone https://github.com/derpyplops/deterministic-serving.git
cd deterministic-serving
python3 demo/run_demo.py --part 1
```

**Expected output:** a capture digest like `sha256:...` printed twice. Running on any other machine with Python 3.10+ produces the same digest. This is the portion of the determinism argument that does not depend on GPU semantics.

### Tier 1 — One GPU (~15 min, one H100/GH200 host)

Reproduces a deterministic vLLM server and verifies identical tokens across two independent requests to the same server.

```bash
# Starts the server from a manifest: pulls the Nix closure (or pip-falls-back),
# downloads the model, resolves the lockfile, launches vLLM.
scripts/reproduce.sh manifests/qwen3-1.7b.manifest.json

# In another shell, run the same prompt twice and hash the output.
for i in 1 2; do
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"Hello"}],"max_tokens":50,"temperature":0,"seed":42}' \
    | python3 -c "import sys,json,hashlib; c=json.load(sys.stdin)['choices'][0]['message']['content']; print(f'Run '$i': {hashlib.sha256(c.encode()).hexdigest()[:16]}')"
done
```

**Expected output:** two identical 16-char hex digests. `scripts/reproduce.sh` verifies the Nix closure digest matches the one pinned in the manifest before starting the server; with Nix available this is the hermetic path. Without Nix the script falls back to unpinned pip (best-effort, non-hermetic) — the reviewer-grade path is the Nix one.

### Tier 2 — Two GPUs, two hosts (~30 min)

Reproduces the cross-server determinism claim from the Results table. Provision two independent H100/GH200 hosts, run the same manifest on each, compare run bundles byte-for-byte.

```bash
# On host A and host B (same manifest, same commit, same container):
scripts/reproduce.sh manifests/qwen3-1.7b.manifest.json &
until curl -sf http://localhost:8000/health; do sleep 3; done

python3 cmd/runner/main.py \
  --manifest manifests/qwen3-1.7b.manifest.json \
  --lockfile .reproduce-run/lockfile.built.v1.json \
  --out-dir /tmp/bundle \
  --mode vllm

# On either host, after copying the other bundle over:
python3 cmd/verifier/main.py \
  --baseline /tmp/bundle-A/run_bundle.v1.json \
  --candidate /tmp/bundle-B/run_bundle.v1.json \
  --report-out /tmp/verify.json \
  --summary-out /tmp/verify.txt
```

**Expected output:** `verify.txt` reports per-request token-equality; `verify.json` contains the structured `verify_report.v1` with first-mismatch path on any divergence. The full Results-table experiment (20 repeated × 34 diverse chunks × 30K tokens per model) is driven by `experiments/multinode-determinism/scripts/run_tiered.py`.

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

## Running the Server

### Prerequisites
- NVIDIA GPU with compute capability >= 9.0 (H100, GH200, etc.)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- The NVIDIA runtime must be the **default Docker runtime**

### One-time Docker + NVIDIA setup

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo tee /etc/docker/daemon.json <<'EOF'
{"runtimes":{"nvidia":{"args":[],"path":"nvidia-container-runtime"}},"default-runtime":"nvidia"}
EOF
sudo systemctl restart docker
```

### Start the server

```bash
docker pull ghcr.io/derpyplops/deterministic-serving-runtime:latest

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

until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 3; done
```

### Send a request

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

## Repository Structure

```
cmd/              CLI entry points (server, resolver, builder, runner, capture, verifier)
pkg/              Shared library code (manifest model, networkdet, common utilities)
schemas/          JSON Schema contracts (manifest, lockfile, run_bundle, verify_report)
manifests/        Model-specific manifest files
demo/             Laptop-runnable determinism demo (no GPU)
experiments/      Determinism experiments, organized by topic
tests/            unit/ integration/ e2e/ determinism/ chaos/
scripts/          reproduce.sh, CI scripts
docs/             ADRs, conformance docs, release policy, diagrams
```

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

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## Citation

A manuscript describing this work is in preparation. Until it is published, please cite the software artifact directly — see [`CITATION.cff`](CITATION.cff), or use:

```bibtex
@misc{deterministic-serving-stack,
  title        = {Deterministic Serving Stack: Bitwise-Reproducible LLM Inference Across Independent Servers},
  author       = {Ng, Jonathan},
  year         = {2026},
  note         = {Manuscript in preparation.},
  url          = {https://github.com/derpyplops/deterministic-serving}
}
```

When citing a specific result, include the commit SHA. This file will be updated with the paper citation (and an arXiv `eprint`) once the preprint is available.
