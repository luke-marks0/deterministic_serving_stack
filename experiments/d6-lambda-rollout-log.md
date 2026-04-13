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

