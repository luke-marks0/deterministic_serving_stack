# Capability modules

The deterministic serving stack, organized **by function**. Each subdirectory is
one capability with a documented interface; the [`Pipeline`](pipeline.py)
composes them, and [`workflows/`](../workflows/) is the recipe book of runnable
compositions.

> These modules are a **curated, stable public surface** over the primitives in
> `pkg/`, `cmd/`, and `flake.nix`. They re-export rather than relocate, so the
> underlying code (and its tests) is untouched. See
> [`docs/plans/repo-modularization.md`](../docs/plans/repo-modularization.md).

## The capability map

| Capability | What it does | Interface | Underlying code |
|---|---|---|---|
| [**build**](build/) | Hermetic, reproducible runtime + OCI image | `nix build .#oci` · `cmd/builder` | `flake.nix`, `cmd/builder`, `native/` |
| [**inference**](inference/) | Bitwise-deterministic vLLM (the c3 config) | `modules.inference` · `cmd/server` | `cmd/{server,runner}`, `pkg/manifest` |
| [**network**](network/) | Deterministic L2 egress frames | `modules.network.egress_frames(...)` | `pkg/networkdet`, `native/libnetdet` |
| [**memory**](memory/) | PoSE memory wipe + erasure attestation | `pose` package *(promoted in Phase 2)* | `experiments/memory_wipe/src/pose` |
| [**attestation**](attestation/) | Matmul / token / replay verification | `cmd/verifier` · `pkg/freivalds` *(facade Phase 2)* | `pkg/{freivalds,e2e,proverdet}`, `cmd/verifier` |
| [**utils**](utils/) | Provisioning, replay server, cloud helpers | `deploy/*` shell scripts | `deploy/`, `scripts/lambda_cli.py` |

A capability need not be a Python package — `build` and `utils` are nix + shell.
The contract is a **documented interface**, not a uniform implementation.

## The unified interface

Everything speaks the artifact spine:

```
manifest.v1  ──resolve──▶  lockfile.v1  ──build──▶  lockfile.v1(+closure)
     │                                                      │
     └──────────────────────── run ────────────────────────┘
                                 ▼
                          run_bundle.v1  ──verify──▶  verify_report.v1
```

Compose it in a few lines instead of bash:

```python
from modules import Pipeline

report = (Pipeline.from_manifest("manifests/qwen3-1.7b.manifest.json")
          .resolve()        # -> lockfile.v1
          .build()          # -> closure digest
          .run("/tmp/a")    # -> run_bundle.v1
          .run("/tmp/b")    # -> run_bundle.v1 (independent run)
          .verify())        # -> verify_report.v1  (status "conformant" iff identical)
```

## Status (Phase 1)

Fully built facades: **network**, **inference**, plus the **Pipeline**.
`build`, `memory`, `attestation`, `utils` are documented here and get Python
facades in Phase 2 (see the plan). Smoke-tested in `tests/modules/`.
