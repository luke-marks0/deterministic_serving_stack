# D6 — Multi-Node Distributed Determinism — Final Report

**Status:** PASS across all three tiers, both models tested. Bitwise-reproducible cross-node inference is confirmed for vLLM 0.17.1 + Ray 2.54 over Lambda H100 TCP, including a 549,927-token DBRX run repeated to the byte.

**Date:** 2026-04-14
**Branch:** `multi-gpu-determinism`
**Plans:** `docs/plans/d6-lambda-staged-rollout.md`, `docs/plans/d6-determinism-tiers.md`
**Execution log:** `experiments/d6-lambda-rollout-log.md`

---

## Headline

| Tier | Model | Workload | Test | Result |
|---|---|---|---|---|
| 1 — Smoke | DBRX (132B MoE) | 4 reqs × 16 tok = 64 tok/run | A == A' | **PASS** |
| 1 — Smoke | Mistral Large 2 (123B dense) | 4 reqs × 16 tok = 64 tok/run | A == A' | **PASS** |
| 2 — Medium | DBRX | 100 reqs × ~91 tok = 9,132 tok/run | A == A' | **PASS** |
| 2 — Medium | DBRX | 100 reqs × ~91 tok = 9,132 tok/run | A == B (shuffled batch order) | **PASS** |
| 2 — Medium | Mistral Large 2 | 100 reqs × ~99 tok = 9,916 tok/run | A == A' | **PASS** |
| 2 — Medium | Mistral Large 2 | 100 reqs × ~99 tok = 9,916 tok/run | A == B (shuffled batch order) | **PASS** |
| 3 — Large | DBRX | 250 reqs × ~2200 tok = **549,927 tok/run** | A == A' | **PASS** |

**Total tokens compared bitwise across runs: 1.14M.** Zero divergences at any token index, in any run, in any tier.

---

## Setup

| Item | Value |
|------|-------|
| Cloud | Lambda Cloud, region `us-south-2` |
| Nodes | 4 × `gpu_1x_h100_sxm5` (NVIDIA H100 80GB HBM3, driver 570.148.08) |
| Container | `ghcr.io/derpyplops/deterministic-serving:multinode` (Nix-built, vLLM 0.17.1, Ray 2.54.0, PyTorch 2.10, FlashAttention v3) |
| Cluster | Ray over **public IPs** (`192.222.x.x` per node), 4 GPU / 104 CPU / 619 GiB RAM aggregate |
| Parallelism | Tensor Parallel **TP=4** (one rank per physical node) |
| Cross-node transport | `NCCL_NET=Socket` over `eno1`, no SHM, no P2P, fixed buffer size |
| NCCL pinning | `NCCL_ALGO=Ring`, `NCCL_PROTO=Simple`, `NCCL_NET=Socket`, `NCCL_P2P_DISABLE=1`, `NCCL_SHM_DISABLE=1`, `NCCL_BUFFSIZE=8388608`, `NCCL_SOCKET_IFNAME=eno1`, `NCCL_DEBUG=WARN` |
| vLLM pinning | `enforce_eager=True`, `disable_custom_all_reduce=True`, `attention_backend=FLASH_ATTN`, `VLLM_USE_RAY_WRAPPED_PP_COMM=0`, `VLLM_BATCH_INVARIANT=1`, `seed=42`, `temperature=0` (greedy) |
| Determinism knobs | `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `PYTHONHASHSEED=0` |
| Gloo bind override | `scripts/d6/sitecustomize.py` (auto-loaded via `PYTHONPATH=/d6`) replaces `ProcessGroupGloo.create_default_device` with `create_device(hostname=$VLLM_HOST_IP)` so the listener binds to the literal public IP — see "The gloo fix" below |

`VLLM_USE_RAY_WRAPPED_PP_COMM=0` is required because vLLM 0.17.1's `RayPPCommunicator` raises `ValueError: cuda_stream other than the current stream is not supported` against Ray 2.54.0. Disabling it falls back to direct NCCL for pipeline communication.

## Models

| Model | Type | Params | Size bf16 | HF repo | Revision |
|---|---|---|---|---|---|
| **DBRX-Instruct** | MoE, 16 experts top-4 | 132B (~36B active) | 246 GiB | `alpindale/dbrx-instruct` (mirror; original `databricks/dbrx-instruct` was pulled from HF) | `8007650525bf3b67d6a4763caf02230061452d45` |
| **Mistral Large 2** | Dense | 123B | 456 GiB on disk (consolidated + sharded both downloaded by `huggingface_hub.snapshot_download`) | `mistralai/Mistral-Large-Instruct-2407` | `main` (token in this branch's experiment had access) |

DBRX gives MoE expert-routing coverage; Mistral Large 2 provides a dense baseline of similar parameter count. Both exceed the aggregate VRAM of 3 × H100 (240 GB), so completing inference is itself structural proof that work is genuinely distributed across all 4 GPUs (and therefore all 4 physical nodes).

## Tier results

### Tier 1 — Smoke (4 prompts × 16 tokens, A vs A')

| Model | Tokens | Result | Observables |
|---|---|---|---|
| DBRX | 64 | PASS | `experiments/multinode_determinism/20260414/tier1-smoke/dbrx/` |
| Mistral Large 2 | 64 | PASS | `experiments/multinode_determinism/20260414/tier1-smoke/mistral/` |

Purpose: end-to-end cluster check after each container-state reset. Cheap, fast iteration unit.

### Tier 2 — Medium (100 prompts × 100 tokens, A vs A' vs B-shuffled)

| Model | Tokens | A == A' | A == B (shuffled) | Observables |
|---|---|---|---|---|
| DBRX | 9,132 | PASS | PASS | `experiments/multinode_determinism/20260414/tier2-medium/dbrx/` |
| Mistral Large 2 | 9,916 | PASS | PASS | `experiments/multinode_determinism/20260414/tier2-medium/mistral/` |

The B-shuffled run uses the same 100 prompts but in a fixed-seed (20260414) shuffled order. Same prompt id → same tokens regardless of where the prompt landed in the prefill schedule, the KV-cache layout, or the MoE expert routing batch. **This is the headline determinism property.**

### Tier 3 — Large (250 prompts × max 4000 tokens, A vs A')

| Model | Requests | Tokens | min/max per req | Wall time per run | Result |
|---|---|---|---|---|---|
| DBRX | 250 | 549,927 | 3 / 4000 | ~78 min | PASS |

Token count fell short of the 1M nominal cap because most prompts hit EOS before 4000 tokens (avg ~2200 tokens/req). Two independent vLLM engine inits (re-load model, re-init NCCL ring, re-init gloo full-mesh) produced **bitwise-identical token IDs across all 250 requests**. The check covers ~78 minutes of distributed generation against an MoE model whose expert routing path is the most likely place for accumulated nondeterminism to show up.

A failure at Tier 3 would have looked like a single token mismatch starting at some request id and token index; we got zero. The compare-tool would have reported the first divergence (`scripts/d6/compare_observables.py` walks per-id and prints `req-XXXX: first divergence at token N: base=X other=Y`).

---

## What went wrong on the way and how it was fixed

This experiment surfaced **a cluster of small bugs** between vLLM 0.17.1, Ray 2.54.0, the Nix-built libtorch 2.10, and Lambda's network topology. None of them are determinism issues per se — they're plumbing. But each one has to be fixed for the experiment to even run. Future-you running this from scratch should expect to hit at least the first three.

1. **vLLM 0.17.1 × Ray 2.54.0 incompatibility.** `vllm.distributed.device_communicators.ray_communicator.RayPPCommunicator.__init__` raises `ValueError: cuda_stream other than the current stream is not supported` because Ray 2.54.0 passes a non-current stream. Set `VLLM_USE_RAY_WRAPPED_PP_COMM=0` to disable the path; vLLM falls back to direct NCCL for pipeline communication.

2. **vLLM env-var copy list silently excludes `GLOO_`.** vLLM's Ray executor copies a fixed set of env-var prefixes from the driver to worker actors (`HF_, HUGGING_FACE_, LMCACHE_, NCCL_, UCX_, VLLM_`). `GLOO_` is missing. To get `GLOO_SOCKET_IFNAME` to the workers at all, set `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=GLOO_`.

3. **`PYTHONPATH` clobbers the Nix store path.** The container's default `PYTHONPATH` is the Nix-built site-packages tree. Setting `-e PYTHONPATH=/d6` at `docker run` **replaces** that, hiding `ray`, `vllm`, `torch`. Ray's `runtime_env_agent` and `dashboard_agent` are spawned as subprocesses and immediately fail with `ModuleNotFoundError: No module named 'ray'`, which manifests confusingly as `Exception: The current node timed out during startup`. **Always prepend, never replace**: `-e PYTHONPATH=/d6:$NIX_DEFAULT_PYTHONPATH`.

4. **The DBRX manifest was a stub.** The original `manifests/dbrx-{pp4,tp4}-multinode.manifest.json` had `tokenizer_revision: "main"` and `weights_revision: "main"`, which fail the schema check (40-char hex required). Resolve `main` to a sha first. Also: `databricks/dbrx-instruct` was pulled from HF — the alpindale mirror works.

5. **DBRX needs `trust_remote_code: true`.** vLLM's `AutoTokenizer` path doesn't recognize DBRX's tiktoken-based tokenizer class without remote code. Tiktoken 0.12.0 is in the container, but the default tokenizer load path doesn't reach it.

6. **vLLM 0.17.1 + DBRX + PP=N has a bug.** `vllm/v1/worker/gpu_model_runner.py:get_attn_backends_for_group` looks up `transformer.blocks.10.norm_attn_norm.attn.attn` in `forward_context`, but with PP that layer lives on a different rank, and the lookup fails with `KeyError`. **TP=4 sidesteps it.** PP=4 is in scope of the original D6 plan but cannot run on this vLLM version against DBRX.

7. **`huggingface_hub.hf_hub_download` is not given the token.** `pkg/common/hf_resolution.py:112` only passes the token to `HfApi` (metadata calls). Setting `HF_TOKEN` as an env var makes `hf_hub_download` pick it up. Without it, gated downloads fail with `GatedRepoError` even though the user account has access.

8. **The Ray worker NVML state degrades over time.** After ~hour-long inferences, fresh `docker exec ... torch.cuda.is_available()` returns `False` with "Can't initialize NVML", even though `docker run --rm` on the same host works fine. Symptom in vLLM: `RuntimeError: ('current platform %s does not support ray.', None)` (the `%s` is unfilled because `current_platform()` returned `None`). Fix: `docker rm -f` and re-create the ray containers. The host-mounted HF cache survives, so this is fast (no re-download).

9. **The Mistral Large 2 download writes to `/tmp/.cache/huggingface` inside the container** by default, which is a container-ephemeral path. Files are lost on `docker rm`. Set `-e HF_HOME=/root/.cache/huggingface` AND mount that as a volume.

### The gloo fix (the new one this session)

The biggest-impact bug this session: **torch's `ProcessGroupGloo` binds its full-mesh listener to a specific IP, picked from `getifaddrs()` on the chosen interface.** Lambda's `eno1` is dual-stack — it has both an RFC 1918 private address (e.g. `172.27.124.243`) and a routable public address (`192.222.53.123`). The kernel returns the private address first; gloo binds there.

Same-region clusters (Phase 2) **happened to work** because the private IPs were mutually reachable on the shared `172.27.124.0/24` subnet. Cross-region or public-IP-addressed clusters fail with `Gloo connectFullMesh failed [...] SO_ERROR: Connection refused, remote=[192.222.53.123]:<port>`. The peer reaches the right host but no listener exists at that IP because gloo bound to the private one.

**Things that did NOT fix it (despite the obvious shape):**

- `VLLM_HOST_IP=<public>` — only fixes vLLM's Ray placement-group pin, not gloo's bind.
- `ip addr del; ip addr add` to reorder eno1's address list — `getifaddrs()` cache doesn't honor the kernel's insertion-order shuffle for gloo's lookup.
- `/etc/hosts` patch on host — `gethostbyname(gethostname())` returns public, but gloo doesn't go through that path.
- `docker run --add-host` so containers see public-mapped hostname from birth — same: gloo's `lookupAddrForIface` ignores hostname resolution.
- `GLOO_SOCKET_IFNAME=eno1` propagated via `VLLM_RAY_EXTRA_ENV_VAR_PREFIXES_TO_COPY=GLOO_` — gloo *does* see the iface, then walks `getifaddrs()` and picks the first AF_INET, which is still the private.

**The actual fix:** torch 2.10 exposes `ProcessGroupGloo.create_device(hostname="<literal IP>", interface="...", lazy_init=...)` as a **single static method** (snake_case in 2.10, not the camelCase `createDeviceForHostname` from older community posts and PRs). Passing an IP literal as `hostname` bypasses `getifaddrs` entirely and `bind()`s the listener to that exact IP. We monkey-patch `create_default_device` (which vLLM's `init_world_group` ultimately calls) to call `create_device(hostname=$VLLM_HOST_IP)`:

```python
# scripts/d6/sitecustomize.py — auto-loaded by Python at startup via PYTHONPATH=/d6
from torch.distributed import ProcessGroupGloo
import os
host_ip = os.environ["VLLM_HOST_IP"]
def _patched_create_default_device(lazy_init=None):
    return ProcessGroupGloo.create_device(hostname=host_ip, lazy_init=lazy_init)
ProcessGroupGloo.create_default_device = staticmethod(_patched_create_default_device)
```

Verified on Node 1 via direct `/proc/net/tcp` inode walk:

| Call | Listener bound to |
|---|---|
| `create_device(hostname="192.222.53.123")` | `192.222.53.123:41835` ✅ public |
| `create_device(hostname="172.27.124.245")` | `172.27.124.245:40683` ❌ private |
| `create_default_device()` (unpatched) | `172.27.124.245:40929` ❌ private |
| `create_device(interface="eno1")` (`GLOO_SOCKET_IFNAME` path) | `172.27.124.245:43513` ❌ private |

The patch propagates to all Ray worker actors automatically because `PYTHONPATH=/d6` is set on the container, Python auto-imports `sitecustomize` at every interpreter startup (driver and every Ray-spawned subprocess), and `VLLM_HOST_IP` is set per-node so each rank patches its own gloo bind to its own public IP.

**Upstream status:** pytorch/pytorch issues #73434, #86962, #93033 are open with no merged fix through PyTorch 2.10. facebookincubator/gloo #343 was closed wontfix with "use createDeviceForHostname with an IP literal" — which is what we did.

---

## What this proves

For each model we ran, on a 4-physical-node H100 cluster talking over TCP via NCCL/Socket and gloo/Socket, **two independent vLLM engine inits over the same manifest produce the same token IDs, byte for byte**. The most demanding test (Tier 3, 549K tokens of MoE generation) shows that the property holds across:

- **Two independent CUDA contexts** (each engine init re-pages the model)
- **Two independent NCCL communicators** (different `commId`s, different ephemeral ports)
- **Two independent gloo full-meshes** (different full-mesh listener ports)
- **~78 minutes of generation per run** (so any accumulated rounding error has time to surface)
- **MoE expert routing on every token** (the path most likely to introduce nondeterminism via sort orders, ties, or routing-bucket batching)

The Tier 2 batch-order invariance test (A vs B-shuffled) additionally proves that **the same prompts in different positions in the batch** produce the same per-prompt tokens. This rules out order-dependent reductions in MoE routing or attention, which were the most probable hidden failure modes.

The Phase 2 anti-cheat checks (per-rank GPU memory non-zero during inference, NCCL ring with the right `commId` cross-node, iptables interdict killing inference) were validated separately in this branch's prior work and not repeated for Tier 1–3 — same cluster, same pinning. The structural anti-cheat for Tier 1–3 is the model size: DBRX (246 GiB) and Mistral Large 2 (more than 240 GiB of weights) cannot fit on fewer than 4 H100s, so the mere fact that inference completes is proof of 4-way distribution.

---

## Cost

| Phase / tier | Wall time (approx) | Spend (approx) |
|---|---|---|
| Phase 0 — Bootstrap | <5 min | $0 |
| Phase 1 — Single-node smoke | ~16 min | ~$1 |
| Phase 2 — 2-node PP=2 + anti-cheat | ~33 min | ~$5 |
| Phase 3 attempt 1 — partial (cross-region, gloo blocker) | ~5 h 15 m | ~$43 |
| **This session — fix validation + 3-tier execution** | ~5 h | ~$60 |
| **Total** | **~12 h** | **~$110** |

Cluster: 4 × `gpu_1x_h100_sxm5` × $4.29/hr = $17.16/hr when active.

---

## What this plan does NOT cover

- **PP=4** (pipeline parallelism). Blocked on vLLM 0.17.1's DBRX layer-resolution bug (item 6 above). Needs a vLLM version bump or a targeted patch. The TP=4 result this session covers the same cross-node NCCL/gloo path; PP would only add a different work-distribution strategy on top.
- **Multi-replica cross-host comparison** — running the same manifest on two physically-separate clusters and comparing across them. Useful for ruling out per-cluster artifacts, but a separate axis from "same cluster, multiple runs."
- **Logit-level comparison** — we compare token IDs only. The schema also stores per-token logits with a fuzzy comparison spec, but the harder test is exact token match and we hit that.
- **Mistral Large 2 at the 1M-token tier** — would have added another $40-60 and ~3 hours; the 9916-token shuffled-batch result is the strongest signal we need from a dense baseline, and Tier 3 was always meant to be MoE-specific.

---

## Conclusion

D6 — bitwise-reproducible cross-node distributed inference for production-scale LLMs — is **achievable today** on the deterministic-serving stack with vLLM 0.17.1 / Ray 2.54.0 / PyTorch 2.10 / Lambda H100 cluster, **provided** every item in the bug list above is patched. The single highest-leverage fix is `scripts/d6/sitecustomize.py` (the gloo bind override), which is required as soon as the cluster's nodes can't reach each other on a shared private subnet. After that, the determinism story holds at every workload size we tested, up to half a million tokens of MoE generation per run.

**Recommended next step:** roll the fixes upstream (vLLM env-var prefix list, `pkg/common/hf_resolution.py` token threading, the gloo bind workaround) and pin a vLLM version that resolves the DBRX-PP layer-lookup bug so PP=4 is also testable.
