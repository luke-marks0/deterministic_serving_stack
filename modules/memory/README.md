# memory — memory wipe + erasure attestation (PoSE)

**Purpose.** Cryptographically wipe a machine's memory (DRAM / GPU HBM / disk)
and *prove* the wipe happened — a Proof of Secure Erasure (PoSE-DB): fill the
region with verifiable noise, then answer challenge–response rounds that only a
holder of the noise can pass.

**Interface (today — runs on the target box).**

```bash
# On a GPU instance, as root (HBM access):
sudo .venv/bin/python3 scripts/benchmark.py --method crypto --disk-gb 500
```

The protocol pieces (`verifier`, `prover`, noise generators, DRAM/HBM/NVMe
backends) live in a self-contained `pose` package with its own tests.

**Artifacts.** Produces an erasure report (challenge success rate, timings).
Integrates with the attestation flow via `modules/attestation/proverdet/erasure.py`
(`run_erasure`).

**Requirements.** Root on the target; a GPU for HBM wipes. Research-grade PoC.

**Underlying code.** `modules/memory/pose/` (protocol/prover/verifier/noise +
`memory/{dram,hbm,nvme}.py`). The bench/probe scripts and the research log live
on the `experiments` branch under `experiments/memory_wipe/scripts/`.

**Status.** Facade in `modules/memory/api.py`: `load_pose("protocol")`,
`load_pose("prover")`, `load_pose("memory.hbm")`, etc. (lazy — no GPU needed to
import). The `pose` package was previously kept under `experiments/memory_wipe/`
as a separately-deployable artifact; it has since been promoted into the module
and is now a regular sub-package imported via `modules.memory.pose`.
