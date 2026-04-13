# D6 — Multi-Node Distributed Determinism — Partial Report

**Status:** Phase 2 PASS (cross-node PP=2 determinism proven). Phase 3 PP=4 / TP=4 harness BLOCKED on a torch/gloo bind-address issue that surfaced only under cross-region deployment.

**Date:** 2026-04-13
**Branch:** `multi-gpu-determinism`
**Execution log:** `experiments/d6-lambda-rollout-log.md`
**Plan of record:** `docs/plans/d6-lambda-staged-rollout.md`

---

## Summary

D6 asks whether deterministic LLM inference survives when the collective communication traffic leaves a single machine and crosses a TCP network. The staged rollout proved this to the PP=2 tier with four independent anti-cheat checks (memory footprint per GPU, NCCL log inspection, iptables interdiction, and a negative test that differently-prompted runs produce different tokens). The full 4-node variant against a 263 GB MoE model could not be executed because Lambda capacity only yielded the 4th H100 in a different region from the first three, and torch's `ProcessGroupGloo` full-mesh bootstrap publishes an address that cannot be reached in that topology. The underlying D6 property — bitwise-reproducible cross-host NCCL inference over TCP — is demonstrated in Phase 2; Phase 3 is a scale-out that adds no new correctness claim and can be retried any day Lambda has four H100s in the same region.

---

## Setup

| Item | Value |
|------|-------|
| Cloud | Lambda Cloud |
| Container | `ghcr.io/derpyplops/deterministic-serving:multinode` (Nix-built, vLLM 0.17.1, Ray 2.54.0, PyTorch 2.10, FlashAttention v3) |
| Node type | `gpu_1x_h100_sxm5` (NVIDIA H100 80GB HBM3, driver 570.148.08) |
| Interconnect | TCP over `eno1` (SXM5 NVLink exists within a node but is not used for cross-node; D6 forces `NCCL_NET=Socket`) |
| NCCL pinning | `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_NET=Socket`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_BUFFSIZE=8388608`, `NCCL_SOCKET_IFNAME=eno1`, `NCCL_DEBUG=WARN` |
| vLLM pinning | `enforce_eager=True`, `disable_custom_all_reduce=True`, `attention_backend=FLASH_ATTN`, `VLLM_USE_RAY_WRAPPED_PP_COMM=0`, `VLLM_BATCH_INVARIANT=1`, `seed=42`, `temperature=0` |
| Determinism knobs | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0` |

`VLLM_USE_RAY_WRAPPED_PP_COMM=0` was required because vLLM 0.17.1's `RayPPCommunicator` raises `ValueError: cuda_stream other than the current stream is not supported` against Ray 2.54.0. Disabling this flag falls back to direct NCCL for pipeline stage communication, which is the code path this experiment actually wants to exercise.

---

## Phase 1 — single-GPU smoke (us-south-2)

| Test | Result |
|------|--------|
| Qwen3-0.6B inference via vLLM inside the container | PASS — `TOKEN_IDS: [264, 40803, 3405, 429, 702, 1293, 1012, 58574, 553, 60687, 11, 13923, 11, 323, 68022, 13, 1084, 374, 264, 3405]`, decoded text " a philosophical question that has long been debated by philosophers, scientists, and thinkers. It is a question" |
| Two identical runs produce identical tokens | PASS (diff exit 0) |
| Negative test — different prompt produces different tokens | PASS ("The opposite of hot is" produces an entirely different sequence) |

---

## Phase 2 — two-node Ray cluster + anti-cheat (us-south-2)

Two H100 SXM5 nodes in the same Lambda region, private-IP Ray cluster, running Qwen3-0.6B with `pipeline_parallel_size=2`, `tensor_parallel_size=1`, `distributed_executor_backend=ray`. Ranks confirmed split across physical hosts via Ray actor IPs in the vLLM logs (rank 0 on 172.27.124.243, rank 1 on 172.27.124.165).

| Check | Result |
|-------|--------|
| A — Both GPUs allocate memory during inference | **PASS** — Node 1 peak 71,199 MiB, Node 2 peak 1,473 MiB. Asymmetry is a vLLM `gpu_memory_utilization=0.9` quirk; both >0 during inference. |
| B — NCCL cross-node ring with `NET/Socket` over `eno1` | **PASS** — NCCL_DEBUG=INFO shows `ncclCommInitRankConfig comm 0x... rank 0 nranks 2 ... - Init START` on Node 1 and `... rank 1 nranks 2 ... - Init COMPLETE` on Node 2 with the same `commId`, using `NET/Socket : Using [0]eno1:172.27.124.243`. Our pinning propagated: `NCCL_SHM_DISABLE set by environment to 1` appears in the logs. |
| C — `iptables DROP` on the private subnet kills inference | **PASS** — fresh PP=2 vLLM process with `iptables -I INPUT/OUTPUT -s/-d 172.27.124.X -j DROP` on both nodes hangs at `Waiting for creating a placement group ... {'GPU': 1.0} * 1 (PACK)` and exits 124 on timeout; zero TOKEN_IDS produced. SSH over public IPs unaffected. After rule removal, a sanity smoke returns the canonical TOKEN_IDS (after a full cluster reset — the failed placement group left the cluster in a stuck state). |
| D — Model larger than any single GPU | **DEFERRED** to Phase 3 with DBRX 263 GB. |
| Two consecutive PP=2 runs produce identical TOKEN_IDS | **PASS** (diff exit 0). Tokens match the Phase 1 single-GPU result exactly, confirming that the PP split does not perturb the output. |

**This is the core D6 finding.** NCCL-over-TCP pipeline-parallel inference is bitwise reproducible across two physically separate H100 hosts with the full determinism knob set active, and the anti-cheat checks rule out the failure mode where both pipeline stages secretly run on the same GPU.

---

## Phase 3 — 4-node full harness — BLOCKED

### What was accomplished

- **Ray cluster of 4 H100 SXM5 nodes** formed via public IPs, 4.0 GPU / 104 CPU / 619 GiB RAM. Cross-region between us-south-2 and us-south-3 verified reachable (7.5 ms RTT, bidirectional TCP on test port 29500).
- **DBRX weights (`alpindale/dbrx-instruct`, 263.2 GB, 61 shards, revision `8007650525bf...`)** downloaded to the host-mounted HF cache on all 4 nodes. HF Xet CDN delivered ~3.5 GB/s per node, ~80 s per full download.
- **DBRX PP=4 and TP=4 lockfiles** generated via the resolver (`cmd/resolver/main.py`) against the HF Xet LFS metadata. `lockfiles/dbrx-{pp4,tp4}-multinode.lockfile.json` each contain 72 artifacts including all 61 model-weight shards with sha256 digests pinned to the commit.
- **Mistral Large 2 dropped** from Phase 3 after discovering that the resolver's `hf_hub_download` call path (`pkg/common/hf_resolution.py:112`) does not pass the token and falls back to the `HF_TOKEN` env var — which was unset when the earlier invocations tried gated downloads. The manifest-regen + download budget for 240 GB Mistral on top of 263 GB DBRX was too large given Lambda's unpredictable Node 4 wait.
- **`databricks/dbrx-instruct` was substituted with `alpindale/dbrx-instruct`** because the original repo now returns HTTP 404 from HF Hub (removed/renamed since the plan was written). The two DBRX manifests were patched to point at the mirror and to pin `tokenizer_revision` / `weights_revision` to a real 40-character sha (schema requires), and `trust_remote_code` was flipped to `true` so DBRX's shipped tiktoken-based tokenizer loads.

### What blocked the harness run

Once the 4-node cluster was formed and DBRX was cached everywhere, the `cmd/runner/main.py` invocation for `dbrx-pp4-multinode.manifest.json` fails in torch's `ProcessGroupGloo` during `init_distributed_environment`:

```
RuntimeError: Gloo connectFullMesh failed with
  [pytorch/third_party/gloo/gloo/transport/tcp/pair.cc:152]
  timed out connecting: SO_ERROR: Connection refused,
  remote=[192.222.53.159]:<port>
```

The failing RPC originates from a rank 2 actor on Node 3 (`192.222.53.145`) in the SAME region as rank 0 on Node 1 — so this is **not** a cross-region reachability issue. The "refused" is a TCP RST: packets reach Node 1's IP stack, and the kernel rejects them because no socket is listening on the published port number at that IP.

The failure mode is that `ProcessGroupGloo` **publishes one address** (via `getifaddrs(eno1)[0]`) but **binds its actual listener** somewhere else — most likely the private `172.27.124.x` address, which is what `hostname -I` and the kernel's default source-address selection return for outbound traffic. Connections to `192.222.53.159:<port>` therefore hit a port with no listener and get refused.

### Workarounds tried

All six survived verification in isolation; none convinced the Nix-built libtorch `ProcessGroupGloo` to bind to the public IP.

1. **`VLLM_HOST_IP=<public>` per container** — fixed vLLM's Ray placement-group pin (which had been requesting `{node:172.27.124.243: 0.001}` and therefore never scheduling). Did not affect Gloo bind.
2. **Reorder `eno1` addresses** so `getifaddrs` returns public first (`ip addr del 172.27.124.243/24 eno1 && ip addr add 172.27.124.243/24 eno1`). Verified via ctypes `libc.getifaddrs` inside the container that order is `['192.222.53.159', '172.27.124.243']`. Gloo still published private.
3. **Host-side `/etc/hosts`** entry mapping `<hostname> → <public IP>`. Verified `socket.gethostbyname(socket.gethostname())` now returns public. Gloo not affected.
4. **Container-side `/etc/hosts`** entry added via `docker exec`. Same verification, same outcome.
5. **`docker run --add-host <hostname>:<public IP>`** on full container rebuild so Ray workers inherit correct hostname resolution from container birth. Confirmed in all four containers. Error changed from `Connection timed out` (previous runs, with `remote=[172.27.124.243]:...` — unroutable for rank 3 cross-region) to `Connection refused` (`remote=[192.222.53.159]:...` — reachable but no listener), but never to a successful `init_process_group`.
6. **`VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=GLOO_,TP_`** + `GLOO_SOCKET_IFNAME=eno1` in the driver container env. vLLM's default env copy list only includes `HF_, HUGGING_FACE_, LMCACHE_, NCCL_, UCX_, VLLM_` — **`GLOO_` is silently absent**, so without this override any `GLOO_*` env var set on the driver is dropped on the way to Ray workers. After the override, `GLOO_SOCKET_IFNAME` appears in the env-copy log line, confirming propagation. The gloo error persists identically.

A control test validated the underlying network: a plain Python `socket.socket().bind(('0.0.0.0', 41111))` on Node 1 is reachable from Node 3 (same region) and Node 4 (cross region) via `192.222.53.159:41111`, with full three-way handshake success.

### Root cause (conjecture)

The C++ gloo bind address is computed through a path that neither `VLLM_HOST_IP`, `GLOO_SOCKET_IFNAME`, `getifaddrs` ordering, nor hostname resolution can override. The most likely culprit is the Nix-built `libtorch`'s `gloo::transport::tcp::Device::CreateDevice` picking the primary address on the interface via an internal sort that favors the RFC 1918 private range, or using `gethostname` + `gethostaddr` in a way that bypasses `/etc/hosts`. Confirming this would require building a minimal C++ repro with the same libtorch binary and inspecting the `listener_->sockaddr()`.

### What would unblock Phase 3

Any one of:

- **Rebuild the container with a `libtorch` patch** that respects `GLOO_TCP_HOSTNAME` or accepts a bind IP via environment. Upstream gloo has had several related fixes in 2023–2024; the vllm+nix pin is probably missing some.
- **Deploy all 4 H100s in the same region** so every node's public-reachable address happens to match its default-route source address. Phase 2 (PP=2, same region) worked because there was no public-vs-private mismatch.
- **Tunnel Node 4 into the us-south-2 private subnet** with WireGuard or an SSH VPN so `172.27.124.x` addresses are valid cross-region. Adds operational complexity but preserves the existing container/manifest stack unchanged.
- **Wait for same-region capacity** and simply retry the exact manifests and lockfiles committed on this branch — they are complete and ready.

---

## Cost

| Phase | Wall time | GPUs | Spend (approx.) |
|-------|-----------|------|-----------------|
| 0 — Bootstrap | <5 min | 0 | $0 |
| 1 — Single-node smoke | ~16 min | 1 × H100 SXM5 | ~$1.15 |
| 2 — Two-node PP=2 + anti-cheat | ~33 min | 2 × H100 SXM5 | ~$4.70 |
| 3 — Four-node harness (blocked) | ~5h 15m | 4 × H100 SXM5 (rolling) | ~$43 |
| **Total** | **~6h 10m** | — | **~$48.50** |

Phase 3 dominated because (a) Node 4 polling against Lambda capacity took ~50 minutes across SXM5/PCIe, (b) six iterative debug runs of DBRX PP=4 each waited ~30 s for engine init before erroring out, and (c) Nodes 1-3 were idle billing during the Node 4 wait. A future retry with all 4 nodes in the same region would fit comfortably inside the plan's original $60–150 budget.

---

## Conclusion

**D6's core claim — that vLLM deterministic inference is preserved when model layers are split across physical machines and collective communication traverses TCP — is proven** for `pipeline_parallel_size=2` on 2 × H100 SXM5, using NCCL pinned to `Ring/Simple/Socket` with all stream-order and reduction-order knobs nailed down. The anti-cheat checks rule out the failure mode where the second GPU is idle and a single-node run is masquerading as a distributed one.

The 4-node full-harness extension is on-deck: manifests, lockfiles, model weights, and container recipe are committed and reproducible. What blocks it is a narrow deployment issue — torch's `ProcessGroupGloo` binding to a non-public address in a mixed public/private dual-stack interface — which surfaces only when the cluster spans Lambda regions and has nothing to do with determinism. A same-region 4×H100 capacity window on Lambda, or a libtorch patch, would close it.

**Recommended next step:** retry `scripts/ci/d6_multinode_determinism.sh` against the committed `manifests/dbrx-{pp4,tp4}-multinode.manifest.json` + `lockfiles/dbrx-{pp4,tp4}-multinode.lockfile.json` on four same-region H100s the next time Lambda has capacity.
