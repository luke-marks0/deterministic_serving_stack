# Deterministic Serving Stack

Batch-invariant vLLM inference with full provenance. Given the same manifest, every run produces bitwise identical outputs — verified across independent servers, concurrency levels, and GPU memory conditions.

**Proven on hardware**: two GH200 480GB instances serving Qwen3-1.7B with `VLLM_BATCH_INVARIANT=1` produce identical outputs for 16/16 requests across nodes, 32-request concurrent batches, and 8K-token context windows.

## Quick Start (Lambda Cloud)

```bash
# 1. Provision a GH200 or H100 instance
deploy/lambda/grab_instance.sh

# 2. Setup the node (installs vLLM, caches model)
deploy/lambda/setup_node.sh <ip>

# 3. Start the deterministic server
deploy/lambda/start_server.sh <ip>

# 4. Query it (OpenAI-compatible)
curl http://<ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"Hello"}],"temperature":0,"seed":42}'

# 5. Verify determinism (sends same requests twice, compares bundles)
ssh ubuntu@<ip> 'cd deterministic_serving_stack && bash deploy/lambda/verify.sh'
```

## How It Works

```
manifest.json (you declare: model, seed, hardware, batch invariance)
  → resolver (pins HF model to immutable commit SHA + per-file digests)
    → lockfile.json (every artifact content-addressed)
      → builder (computes runtime_closure_digest from Nix closure)
        → lockfile.built.json
          → server (validates everything at boot, serves via vLLM)
            → capture.jsonl (every request/response logged)
              → capture tool (converts to run bundle)
                → verifier (compares two bundles → conformant/mismatch)
```

The server refuses to start if the lockfile digest doesn't match the manifest. The Nix closure produces a stable `runtime_closure_digest` — rebuild the same flake on any machine and get the same hash.

## Repository Structure

```
cmd/
  resolver/       Manifest → lockfile (HF resolution, artifact pinning)
  builder/        Lockfile → lockfile with runtime_closure_digest
  runner/         Manifest + lockfile → run bundle (synthetic or vLLM mode)
  server/         Deterministic vLLM server with capture proxy
  capture/        Server capture log → verifiable run bundle
  coordinator/    Multi-node deterministic request dispatcher
  verifier/       Compare two run bundles → conformance report
pkg/
  common/         Canonical JSON, SHA256, schema validation, HF resolution
  hardware/       Topology, rack policy, failure domains, SLO tracking
schemas/          JSON Schema contracts (manifest, lockfile, run_bundle, verify_report)
manifests/        Model-specific manifests (Qwen3-1.7B)
nix/              Nix closure and OCI image derivations
deploy/
  lambda/         Lambda Cloud provisioning and deployment scripts
  k8s/            Kubernetes manifests (single-node, multi-node replicated)
  helm/           Helm chart
tests/
  unit/           Schema validation, topology, bug regression tests
  integration/    Component interaction tests
  determinism/    D0-D5 determinism matrix
  e2e/            End-to-end pipeline tests
  chaos/          Fault injection scaffolding
  fixtures/       Positive and negative test fixtures
```

## What's Deterministic

| Attack | Result |
|--------|--------|
| bs=1 vs bs=32 concurrent | Identical |
| Max context (8192 tokens) | Identical |
| Two independent servers | Identical |
| GPU memory pressure (1GB extra) | Identical |
| Env var mutation after boot | No effect |
| Long output (2048 tokens) | Identical |

## Known Vulnerabilities

1. **Proxy bypass**: vLLM on port 8001 can be hit directly, bypassing the capture log.
2. **No per-request lockfile re-validation**: lockfile is checked at boot only.
3. **bundle_digest not re-verified**: verifier checks observable digests but doesn't recompute bundle_digest.

See `docs/conformance/SPEC_REVIEW.md` for the full audit.

## Conformance Status

38/44 spec requirements implemented, 4 scaffolding (networking), 2 planned.

The userspace networking stack (DPDK/L2 determinism) and on-disk artifact verification are scaffolding — the schemas and provenance fields exist but there's no real networking stack. Activations are a deterministic proxy (token hash), not actual intermediate values from the model.

## Running the Container

### Prerequisites
- NVIDIA GPU with compute capability >= 9.0 (H100, GH200, etc.)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- The NVIDIA runtime must be the **default Docker runtime** (see setup below)

### One-time Docker + NVIDIA setup

The container is a hermetic Nix image — it doesn't use the host's glibc or CUDA toolkit. It only needs the host's NVIDIA kernel driver injected at runtime. This requires the NVIDIA Container Toolkit to be configured correctly.

```bash
# 1. Install NVIDIA Container Toolkit (if not already installed)
# See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 2. Configure Docker to use nvidia as the DEFAULT runtime
#    This is critical — just having --gpus all is not enough for Nix containers
sudo nvidia-ctk runtime configure --runtime=docker

# 3. Set nvidia as the default runtime (edit /etc/docker/daemon.json):
sudo tee /etc/docker/daemon.json <<'EOF'
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia"
}
EOF

# 4. Restart Docker
sudo systemctl restart docker

# 5. Verify GPU access works
docker run --rm --gpus all --privileged \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  ghcr.io/derpyplops/deterministic-serving-runtime:latest \
  python3 -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: NVIDIA GH200 480GB (or your GPU name)
```

**Important gotchas we hit during setup:**

1. **`--privileged` is required** on some hosts (notably Lambda Cloud GH200 instances). Without it, `/dev/nvidia0` is visible but `cuInit()` returns `CUDA_ERROR_NO_DEVICE` due to cgroup restrictions.

2. **`NVIDIA_DRIVER_CAPABILITIES=all`** must be set. The nvidia-container-runtime injects driver libs into `/usr/lib64/` but the Nix Python environment doesn't search that path by default. The container's `LD_LIBRARY_PATH` includes `/usr/lib64` but `NVIDIA_DRIVER_CAPABILITIES=all` is needed to trigger the lib injection.

3. **Podman doesn't work** (v3.4.x on Lambda). It lacks CDI support and the `--gpus` flag doesn't inject driver libs. Use Docker.

4. **Don't set `LD_LIBRARY_PATH` to host paths** (e.g. `/usr/lib/aarch64-linux-gnu`). The Nix image uses a newer glibc than the host — mixing them causes symbol version errors like `GLIBC_2.38 not found`.

### Pull the image

```bash
docker pull ghcr.io/derpyplops/deterministic-serving-runtime:latest

# Or load from a nix-built tarball:
# nix build .#oci && docker load < result
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
    --host 0.0.0.0 \
    --port 8000 \
    --seed 42 \
    --enforce-eager \
    --attention-backend TRITON_ATTN \
    --max-model-len 4096
```

### Wait for startup (~30-90s depending on model download)

```bash
until curl -s http://localhost:8000/health > /dev/null 2>&1; do sleep 3; done
echo "ready"
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

### Stop

```bash
docker rm -f vllm-server
```

### Notes
- Replace `Qwen/Qwen3-1.7B` with any supported model (e.g. `Qwen/Qwen3-8B`)
- Set `-e HF_TOKEN=<token>` for gated models
- `VLLM_BATCH_INVARIANT=1` enables deterministic batch-invariant inference
- `--enforce-eager` disables CUDA graphs (required for batch invariance)
- `--attention-backend TRITON_ATTN` or `FLASH_ATTN` (both supported)
- The container includes `gcc` for Triton JIT compilation at runtime

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Failed to infer device type` | GPU not visible to container | Add `--privileged -e NVIDIA_DRIVER_CAPABILITIES=all` |
| `No CUDA GPUs are available` | `cuInit()` blocked by cgroups | Add `--privileged` |
| `Can't initialize NVML` | Driver libs not injected | Ensure `"default-runtime": "nvidia"` in daemon.json |
| `Failed to find C compiler` | Triton can't JIT kernels | Container must include gcc (current image does) |
| `GLIBC_2.38 not found` | Host libs leaking into Nix env | Don't set `LD_LIBRARY_PATH` to host system paths |

## Building the Nix Closure

```bash
# Build the hermetic runtime closure
nix build .#closure

# Compute the runtime_closure_digest
nix path-info --json --recursive $(nix build .#closure --print-out-paths) | \
  python3 -c "import json,sys,hashlib; d=json.load(sys.stdin); ..."

# Build the OCI image
nix build .#oci

# Push to registry
scripts/ci/build_oci.sh --push --registry ghcr.io/yourorg/deterministic-serving
```

## CI Gates

| Gate | Runs | Command |
|------|------|---------|
| PR | lint + schema + unit/integration | `make ci-pr` |
| Main | + e2e + determinism + nix closure | `make ci-main` |
| Nightly | + chaos + D0-D5 + long-run | `make ci-nightly` |
| Release | + release contracts | `make ci-release` |

## Local Development

```bash
# Run fast tests
make ci-pr

# Run the synthetic pipeline (no GPU needed)
tmp=$(mktemp -d)
python3 cmd/resolver/main.py --manifest manifests/qwen3-1.7b.manifest.json \
  --lockfile-out $tmp/lock.json --resolve-hf
python3 cmd/builder/main.py --lockfile $tmp/lock.json --lockfile-out $tmp/built.json
python3 cmd/runner/main.py --manifest manifests/qwen3-1.7b.manifest.json \
  --lockfile $tmp/built.json --out-dir $tmp/run

# Or with a real GPU:
python3 cmd/runner/main.py --manifest manifests/qwen3-1.7b.manifest.json \
  --lockfile $tmp/built.json --out-dir $tmp/run --mode vllm
```
