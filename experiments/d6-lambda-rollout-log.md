# D6 Lambda Rollout — Experiment Log

This is an append-only journal for the staged D6 rollout on Lambda Cloud.
Every major action, configuration, setback, and milestone is logged here.

Plan: docs/plans/d6-lambda-staged-rollout.md

---

## Phase 0: Bootstrap

### 2026-04-13T11:34Z — Started Phase 0

Working through docs/plans/d6-lambda-staged-rollout.md.
Local environment verified: branch=multi-gpu-determinism, LAMBDALABS_API_KEY set,
~/.ssh/id_ed25519.pub present, all four multinode manifests (dbrx/mistral-large2 ×
pp4/tp4) on disk.

### 2026-04-13T11:36Z — COST: terminated 3 pre-existing leaked instances

Account inventory at start of Phase 0 showed 3 unexpected active instances
(combined ~$4.87/hr) unrelated to D6:
- 773a7845312b4ee08acc3f56969721a7  gpu_1x_gh200    us-east-3  dpdk-egress-test
- 2877a748b19843c791306577d8d53c30  gpu_1x_a10      us-west-1  pose-kexec-tight
- d20312c2e45b490c8852980cf70bba41  gpu_1x_a10      us-west-1  pose-kexec-tight

User confirmed termination. All three returned status=terminating.
Lesson: always run `lambda_cli.py list` before assuming a clean account.

### 2026-04-13T11:36Z — Verified Lambda API access

instance-types: 16 (gpu_1x_h100_sxm5 present)
instances: 0 (after termination)
ssh-keys: ['macbook 2025', 'macbook', 'arena 2022']
No key from this machine yet — will register in Task 0.4.

### 2026-04-13T11:37Z — Registered SSH key with Lambda

Key name: d6-rollout
Key id: 377c0353bad042cdb8bb81e8a1a688d1
Public key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAICbH+zsGjLDKlyelxJY6JQrtEYgGBBqSowk758eKNbbs

### 2026-04-13T11:37Z — SETBACK: lambda_cli.py first run hit Cloudflare 1010

GET /instance-types -> HTTP 403: error code: 1010

Lambda's Cloudflare WAF blocks the default `Python-urllib/3.x` User-Agent.
Curl worked from the same host (curl sets a UA Cloudflare accepts).
Fix: added `User-Agent: d6-lambda-cli/1.0` to `_auth_header()`.
After fix, types/list/keys all returned cleanly.

### 2026-04-13T11:37Z — MILESTONE: lambda_cli.py smoke test passed

Ran `poll gpu_8x_h100_sxm5 --count 1 --interval 5` under a SIGINT timeout.
Two iterations printed "no capacity", then SIGINT exited cleanly (5-line
traceback from time.sleep, within the plan's tolerance).

Note: at 11:37Z, `types` showed every GPU type with `available_in: -`
(zero capacity across all regions). Phase 1 polling may take a while.

### 2026-04-13T11:38Z — END Phase 0: bootstrap complete

Lambda API reachable, SSH key `d6-rollout` registered, lambda_cli.py in place
and tested. Account is clean (3 leaked instances terminated at start).
Ready for Phase 1 — pending user go-ahead before spending money.

---

## Phase 1: Single-Node Smoke Test

### 2026-04-13T11:42Z — COST: launched 1× H100 SXM5 on Lambda

Instance ID: a446ea16d747426a86e1c619f2a163e4
Type: gpu_1x_h100_sxm5
Region: us-south-2
Cost: $4.29/hr
Polling time: ~2.5 min (5 iters at 30s before capacity appeared)

### 2026-04-13T11:46Z — MILESTONE: SSH ready on Node 1

IP: 192.222.53.159
GPU: NVIDIA H100 80GB HBM3
Driver: 570.148.08
Time from launch to SSH-ready: ~4.5 min

### 2026-04-13T11:48Z — SETBACK: ubuntu user not in docker group

`docker pull` failed with permission denied on /var/run/docker.sock.
Workaround: prefix all docker commands with `sudo`. Did NOT add user to
docker group — sudo is fine for the duration of Phase 1.
For Phase 2/3, consider `sudo usermod -aG docker ubuntu` once per node.

### 2026-04-13T11:48Z — MILESTONE: container pulled on Node 1

Image: ghcr.io/derpyplops/deterministic-serving:multinode
Digest: sha256:0bb288d6d59391728b49fa3600e59c5a0b66f3cf72c94d59cd0b842a2b60b35d
Pull time: ~30s

### 2026-04-13T11:49Z — SETBACK: container's /usr/bin/nvidia-smi is broken

`nvidia-smi` inside the container returns "cannot execute: required file not
found". Common with Nix-built containers — the host's NVIDIA Container Toolkit
bind-mounts driver libs, but the container's nvidia-smi binary is incompatible.
Not a real problem: torch.cuda still works because libcuda is mounted in.
Workaround: skip nvidia-smi inside the container, use torch instead.

### 2026-04-13T11:50Z — MILESTONE: container has GPU access

`python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"`
→ `True 1 NVIDIA H100 80GB HBM3`

### 2026-04-13T11:50Z — SETBACK: vLLM batch_invariant requires explicit attention backend

First smoke run failed with:
  RuntimeError: VLLM batch_invariant mode requires an attention backend in
  ['FLASH_ATTN', 'TRITON_ATTN', 'FLASH_ATTN_MLA', 'TRITON_MLA'], but got 'None'.

Setting `VLLM_ATTENTION_BACKEND=FLASH_ATTN` env var did NOT propagate to vLLM
v1's attention_config (which is read independently of the env var).

Fix: pass `attention_backend="FLASH_ATTN"` as a kwarg to `LLM(...)` directly.
This matches what `cmd/runner/vllm_runner.py:123` already does. Updated
`scripts/d6/phase1_smoke.py` accordingly.

### 2026-04-13T11:51Z — MILESTONE: first inference on Lambda H100

Model: Qwen/Qwen3-0.6B
Engine init + 1 prompt × 20 tokens: ~1.3s/it after init
TOKEN_IDS: [264, 40803, 3405, 429, 702, 1293, 1012, 58574, 553, 60687, 11,
            13923, 11, 323, 68022, 13, 1084, 374, 264, 3405]
TEXT: " a philosophical question that has long been debated by philosophers,
       scientists, and thinkers. It is a question"

### 2026-04-13T11:52Z — SETBACK: Nix container has no `grep`

First determinism repeat run gave a false positive: piping `... | grep TOKEN_IDS`
*inside* the container produced empty files (grep not on PATH), and `diff` of
two empty files reported "DETERMINISTIC ✓".

Lesson: do all text filtering on the host side. The container is a python
runtime, not a userland.

Fix: redirect docker stdout to host, then `grep` on the host.

### 2026-04-13T11:53Z — MILESTONE: single-GPU determinism verified on Lambda

Two consecutive runs of Qwen3-0.6B (same seed, same prompt, same container)
produced bitwise-identical TOKEN_IDS (diff exit 0).

run-a TOKEN_IDS: [264, 40803, 3405, 429, 702, 1293, 1012, 58574, 553, 60687,
                  11, 13923, 11, 323, 68022, 13, 1084, 374, 264, 3405]
run-b TOKEN_IDS: (identical)

### 2026-04-13T11:53Z — MILESTONE: negative test passes

Same script with prompt "The opposite of hot is" produced:
  TOKEN_IDS: [9255, 11, 323, 279, 14002, 315, 9255, 374, 4017, 13, 2055, 11,
              421, 264, 1697, 374, 4017, 11, 1221, 279]
diff against run-a: differs immediately. Comparison detects mismatch.

### 2026-04-13T11:53Z — END Phase 1: single-node smoke complete

Container works on Lambda H100. Single-GPU inference is deterministic.
Determinism check is negative-tested. No unresolved SETBACKs.

Node 1 (a446ea16d747426a86e1c619f2a163e4 @ 192.222.53.159, us-south-2)
remains running and will become the Ray head node in Phase 2 (pending user
approval before launching Node 2).

Wall time Phase 1: ~16 min. Cost so far: ~$1.15.

