# Plan: End-to-End Determinism Experiment

## Goal

Given `(nix flake, manifest)` as inputs, produce `(inference output, packet output)` where:
- **inference output** = tokens + logprobs for every request in the manifest
- **packet output** = all network packets after warden normalization, in canonical form

Running this twice on two independent machines with the same flake and manifest must produce **bitwise identical** inference output and packet output.

## Architecture

```
manifest.json ──┐
                 ├──> proxy server ──> vLLM ──> tokens + logprobs
flake.nix ──────┘        │
                         │ responses
                         ▼
                   net stack (sim) ──> raw L2 frames
                         │
                         ▼
                   ActiveWarden.normalize() ──> scrubbed L2 frames
                         │
                         ▼
                   CaptureRing ──> canonical packet output
                         │
                         ▼
                   output bundle: {
                     inference: [{id, tokens, logprobs}],
                     packets: {digest, frames: [{index, hex}]},
                     manifest_digest,
                     closure_hash,
                   }
```

The key insight: we already have all the pieces. The `DeterministicNetStack` builds L2 frames from response bytes. The `ActiveWarden` normalizes them. The `CaptureRing` records them. We just need to wire them together in the experiment flow and save the output in a comparable format.

## What exists today

| Component | File | Status |
|-----------|------|--------|
| Proxy server with /manifest | `cmd/server/main.py` | Working, deployed |
| vLLM inference | via proxy → vLLM subprocess | Working, deployed |
| Deterministic frame builder | `pkg/networkdet/frame.py` | Working, tested |
| Active warden (MRF normalizer) | `pkg/networkdet/warden.py` | Working, 32 tests |
| Capture ring | `pkg/networkdet/capture.py` | Working, tested |
| Simulated network backend | `pkg/networkdet/backend_sim.py` | Working |
| Net stack facade | `pkg/networkdet/__init__.py` | Working, but disconnected from runner after v2 cleanup |
| Tiered experiment runner | `experiments/run_tiered.py` | Working, but no packet output |

## What needs to be built

### 1. Experiment orchestrator

**New file:** `experiments/run_e2e.py`

This replaces `run_tiered.py` for the e2e case. It:

1. Reads a manifest from disk
2. POSTs it to the proxy server's `/manifest` endpoint
3. Waits for vLLM to be ready
4. For each request in the manifest:
   a. Sends the request to `/v1/chat/completions`
   b. Records the response (tokens, logprobs)
   c. Serializes the response to bytes
   d. Feeds the bytes through the deterministic net stack → warden → capture ring
5. Writes the output bundle to disk

```python
def run_e2e(manifest_path: str, server_url: str, out_dir: str):
    manifest = json.load(open(manifest_path))

    # POST manifest to server
    post_manifest(server_url, manifest)

    # Create net stack + warden
    net = DeterministicNetStack(config, run_id=manifest["run_id"])
    warden = ActiveWarden(secret=b"experiment-key")

    results = []
    for i, req in enumerate(manifest["requests"]):
        # Send inference request
        response = send_completion(server_url, manifest["model"]["source"], req)

        # Record inference output
        results.append({
            "id": req["id"],
            "tokens": response["token_ids"],
            "logprobs": response["logprobs"],
            "content_hash": sha256(response["content"]),
        })

        # Build deterministic frames from the response
        response_bytes = canonical_json_bytes(response)
        frames = net.process_response(conn_index=i, response_bytes=response_bytes)

        # Pass each frame through the warden
        for frame in frames:
            normalized = warden.normalize(frame)
            # normalized frame is already captured by the net stack's capture ring

    # Write output bundle
    bundle = {
        "manifest_digest": sha256(canonical_json_bytes(manifest)),
        "closure_hash": os.environ.get("CLOSURE_HASH", "unknown"),
        "inference": results,
        "packets": {
            "digest": net.capture_digest(),
            "frame_count": net.frame_count(),
            "frames": net.capture_frames_hex(),
        },
    }
    write_json(out_dir / "e2e_bundle.json", bundle)
```

### 2. Warden integration into the packet pipeline

Currently, the net stack builds frames but doesn't pass them through the warden. We need a simple wrapper:

**Option A:** Add a `warden` parameter to `DeterministicNetStack` that normalizes each frame before capture.

**Option B (simpler):** Do it in the orchestrator — after `net.process_response()` returns frames, pass each through `warden.normalize()`, then record the normalized frames in a separate capture ring.

Option B is better because it keeps the net stack and warden as separate, testable components. The orchestrator composes them.

```python
warden = ActiveWarden(secret=b"experiment-key")
warden_capture = CaptureRing()

for frame in raw_frames:
    normalized = warden.normalize(frame)
    if normalized is not None:
        warden_capture.record(normalized)

# The packet output is the warden_capture digest
packet_digest = warden_capture.digest()
```

### 3. Canonical packet output format

The output must be comparable across machines. The frames themselves are deterministic (same inputs → same frames → same warden output). The canonical form is:

```json
{
  "packet_digest": "sha256:abc123...",
  "frame_count": 42,
  "warden_secret": "experiment-key",
  "frames": [
    {"index": 0, "hex": "0200000000010200000000020800450000..."},
    {"index": 1, "hex": "0200000000010200000000020800450000..."},
  ]
}
```

The `packet_digest` is the SHA256 of all normalized frame bytes concatenated in order. Two machines producing the same `packet_digest` proves packet-level determinism.

### 4. Comparison tool

**New file:** `experiments/compare_e2e.py`

Takes two output bundles and compares:

```bash
python3 experiments/compare_e2e.py --bundle-a s1/e2e_bundle.json --bundle-b s2/e2e_bundle.json
```

Output:
```
Manifest digest:  MATCH (sha256:...)
Closure hash:     MATCH (sha256:...)
Inference:
  req-001: tokens MATCH, logprobs MATCH
  req-002: tokens MATCH, logprobs MATCH
Packets:
  digest: MATCH (sha256:...)
  frame_count: 42 / 42
VERDICT: IDENTICAL
```

### 5. Deployment script

**New file:** `experiments/run_e2e_both_servers.sh`

Orchestrates the experiment on two Lambda servers:

```bash
#!/bin/bash
S1=192.222.50.183
S2=192.222.56.186
MANIFEST=manifests/qwen3-1.7b.manifest.json

# Start containers on both (using the same nix image)
for IP in $S1 $S2; do
  ssh ubuntu@$IP "sudo docker run -d --name e2e --gpus all --privileged ..."
done

# Wait for both servers
for IP in $S1 $S2; do
  wait_for_health $IP
done

# Run experiment on both
ssh ubuntu@$S1 "python3 experiments/run_e2e.py --manifest $MANIFEST --server http://localhost:8000 --out /tmp/e2e-s1"
ssh ubuntu@$S2 "python3 experiments/run_e2e.py --manifest $MANIFEST --server http://localhost:8000 --out /tmp/e2e-s2"

# Collect results
scp ubuntu@$S1:/tmp/e2e-s1/e2e_bundle.json /tmp/s1.json
scp ubuntu@$S2:/tmp/e2e-s2/e2e_bundle.json /tmp/s2.json

# Compare
python3 experiments/compare_e2e.py --bundle-a /tmp/s1.json --bundle-b /tmp/s2.json
```

## What makes the packet output deterministic

Each component in the pipeline is deterministic given the same inputs:

1. **vLLM inference**: Same model + same seed + same prompt + batch invariance → same tokens (verified by our experiments)
2. **Response serialization**: `canonical_json_bytes(response)` → same bytes (deterministic JSON serialization)
3. **Frame builder**: Same bytes + same config (MTU, MSS, addressing) + same conn_index → same L2 frames (tested)
4. **Warden normalize**: Same frame + same secret → same normalized frame (keyed hash for ISN/IP-ID, deterministic field rewriting, tested with 32 tests)
5. **Capture ring**: Same frames in same order → same digest (SHA256 of concatenated bytes)

The only external dependency is the nix closure (pinned by flake) and the warden secret (declared in the experiment config).

## Implementation order

```
1. experiments/run_e2e.py          — orchestrator
2. experiments/compare_e2e.py      — comparison tool
3. tests/e2e/test_e2e_pipeline.py  — synthetic test (no GPU)
4. experiments/run_e2e_both.sh     — deployment script
5. Run on Lambda servers           — real test
```

### Task 1: Orchestrator (`experiments/run_e2e.py`)

Wire together: HTTP client → net stack → warden → capture → output bundle.

Test locally with the synthetic runner (no GPU):
```python
# Build frames from synthetic response
response_bytes = canonical_json_bytes({"tokens": [1, 2, 3], "logprobs": [-0.1]})
frames = net.process_response(conn_index=0, response_bytes=response_bytes)
for frame in frames:
    normalized = warden.normalize(frame)
    if normalized:
        capture.record(normalized)
print(capture.digest())  # deterministic
```

### Task 2: Comparison tool (`experiments/compare_e2e.py`)

Load two JSON bundles, compare field by field. Exit 0 if identical, exit 1 if divergent.

### Task 3: Synthetic e2e test (`tests/e2e/test_e2e_pipeline.py`)

No GPU needed. Tests that:
1. Given the same manifest and synthetic responses, two runs produce the same bundle
2. The packet digest matches between runs
3. Changing the warden secret changes the packet digest
4. Changing a response changes both inference and packet digests

This test proves the pipeline is deterministic end-to-end without needing actual inference.

### Task 4: Deployment script

Shell script that SSHes into both Lambda servers, starts containers, runs the experiment, collects results, and compares. Uses the same pattern as our existing experiment scripts.

### Task 5: Real test on Lambda

Run the experiment on both GH200 servers with the nix container and Qwen3-1.7B. Compare the bundles. This is the proof that `(nix, manifest) → (inference, packets)` is reproducible across machines.

## Net stack config without the network manifest section

The v2 manifest removed the `network` section. The net stack's `parse_net_config()` already handles this — it returns sensible defaults (MTU 1500, MSS 1460, no offloads, plaintext). The experiment should use these defaults. If we need custom addressing (src/dst IP/MAC), pass them as CLI args to `run_e2e.py`, not in the manifest.

## Testing strategy

| Test | What it proves | GPU needed |
|------|---------------|------------|
| `test_e2e_pipeline.py` (synthetic) | Pipeline wiring is correct, output is deterministic | No |
| `run_e2e.py` on one server | Real inference + packet output works | Yes |
| `run_e2e.py` on two servers + compare | Cross-machine determinism | Yes × 2 |

The synthetic test is the gate for CI. The real tests are for experiments.
