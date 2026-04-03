# Full Wipe Experiment Plan

**Goal**: Run a complete PoSE-DB sanitization cycle on a GH200 — from serving
inference, through teardown + wipe + verification, to restored inference — and
report the five key metrics.

**Key metrics**:
1. **Coverage**: fraction of addressable memory wiped and verified
2. **Wipe time**: time to fill all memory with noise
3. **Rounds passed**: challenge-response verification (1000/1000)
4. **Total downtime**: teardown + wipe + restore
5. **Resume time**: wipe complete → first token generated

---

## Prerequisites

- A running GH200 instance on Lambda Cloud (use `scripts/provision_gh200.sh`)
- SSH access as `ubuntu@<ip>`
- The `pose` package deployed to `~/memory_wipes`

---

## Experiment Phases

```
  prep (not timed)
    │
    ▼
  baseline inference ──── docker pull + run GPT-2 + generate token
    │
    ▼
  teardown ────────────── docker rm + rmi + apt remove docker + cleanup
    │
    ▼
  wipe + prove ────────── fill DRAM/HBM/disk + 1000 challenge rounds
    │
    ▼
  restore ─────────────── apt install docker + docker pull + run GPT-2
    │
    ▼
  post-wipe inference ─── generate token (proof of recovery)
    │
    ▼
  write report
```

---

## Tasks

### Task 1: Prep Script (strip host to minimum)

**Goal**: Remove all bloat so the wipe covers maximum disk space. Run once
before the experiment. Not timed.

**File**: `scripts/run_experiment.py` — `prep()` function

**What to remove**:
```bash
# Lambda bloat (ML frameworks, cloud CLIs)
sudo apt remove -y --purge \
    python3-torch-cuda python3-tensorflow-cuda python3-jax-cuda12-plugin \
    python3-flash-attn-cuda libmagma2 azure-cli nccl-tests \
    nvidia-cuda-dev nvidia-cuda-toolkit libnccl-dev python3-jaxlib \
    libboost1.74-dev mft
sudo apt autoremove -y
sudo apt clean

# Snap packages
sudo snap remove --purge google-cloud-sdk aws-cli lxd

# Fix CUDA runtime symlink (apt remove breaks it)
sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudart.so.12 \
            /usr/lib/aarch64-linux-gnu/libcudart.so
sudo ldconfig
```

**What to keep**: kernel, systemd, network, Docker, CUDA runtime libs
(`libcudart12`, `libnvidia-compute`), Python, our `pose` package.

**Record**: `host_footprint_bytes` via `subprocess: du -sb /`.

**Done when**: `df -h /` shows ~5-8 GiB used, `nvidia-smi` works,
`python3 -c "import ctypes; ctypes.CDLL('libcudart.so')"` works,
`docker run hello-world` works.

---

### Task 2: Baseline Inference

**Goal**: Pull NGC PyTorch image, start container, load GPT-2 to GPU, generate
one token. This proves the system works before the wipe.

**File**: `scripts/run_experiment.py` — `baseline_inference()` function

**Docker image**: `nvcr.io/nvidia/pytorch:24.05-py3` (or latest compatible with
CUDA 12.8 driver). This image includes PyTorch + CUDA runtime.

**Inference script** (runs inside the container):
```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.monotonic()
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
load_time = time.monotonic() - t0

prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

t0 = time.monotonic()
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=1)
gen_time = time.monotonic() - t0

token = tokenizer.decode(output[0][-1])
mem = torch.cuda.memory_allocated()

import json
print(json.dumps({
    "load_time_s": round(load_time, 3),
    "gen_time_s": round(gen_time, 3),
    "token": token,
    "gpu_mem_used_bytes": mem,
}))
```

**Orchestrator logic**:
```python
def baseline_inference():
    t0 = time.monotonic()
    subprocess.run(["docker", "pull", IMAGE], check=True)
    pull_time = time.monotonic() - t0

    result = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all",
         "--name", "pose-inference",
         IMAGE, "python3", "-c", INFERENCE_SCRIPT],
        capture_output=True, text=True, check=True,
    )
    total_time = time.monotonic() - t0
    inference_result = json.loads(result.stdout)
    inference_result["total_time_s"] = total_time
    inference_result["pull_time_s"] = pull_time
    return inference_result
```

**Done when**: Returns a dict with `token`, `load_time_s`, `gen_time_s`,
`gpu_mem_used_bytes`.

---

### Task 3: Teardown

**Goal**: Remove the container, image, Docker itself, and all associated files.
Leave the minimum on disk for the wipe.

**File**: `scripts/run_experiment.py` — `teardown()` function

```python
def teardown():
    t0 = time.monotonic()
    disk_before = disk_used_bytes()

    # Remove container and image
    subprocess.run(["docker", "rm", "-f", "pose-inference"], check=False)
    subprocess.run(["docker", "system", "prune", "-af"], check=True)

    # Remove Docker itself
    subprocess.run(
        ["sudo", "apt", "remove", "-y", "--purge",
         "docker-ce", "docker-ce-cli", "containerd.io",
         "docker-buildx-plugin", "docker-compose-plugin"],
        check=True,
    )
    subprocess.run(["sudo", "rm", "-rf", "/var/lib/docker", "/var/lib/containerd"],
                   check=True)
    subprocess.run(["sudo", "apt", "autoremove", "-y"], check=True)

    disk_after = disk_used_bytes()
    return {
        "time_s": time.monotonic() - t0,
        "disk_freed_bytes": disk_before - disk_after,
    }
```

**Done when**: `which docker` returns non-zero. Disk usage is at the minimum
(kernel + CUDA runtime + our code).

---

### Task 4: Wipe + Prove

**Goal**: Fill all addressable memory with noise and verify via 1000
challenge-response rounds. This is the existing PoSE protocol.

**File**: `scripts/run_experiment.py` — `wipe_and_prove()` function

```python
def wipe_and_prove():
    from pose.detect import compute_ceilings
    from pose.protocol import run_protocol
    from pose.memory.dram import DramRegion
    from pose.memory.hbm import HbmRegion
    from pose.memory.nvme import NvmeRegion

    ceil = compute_ceilings(
        disk_path="/tmp",
        dram_fraction=0.98,
        hbm_fraction=0.99,
        disk_fraction=0.95,
    )

    nvme_file = "/tmp/pose_wipe.bin"
    dram = DramRegion(size_bytes=ceil.dram_wipeable, block_size=4096)
    hbm = HbmRegion(size_bytes=ceil.hbm_wipeable, block_size=4096)
    nvme = NvmeRegion(nvme_file, num_blocks=ceil.disk_wipeable // 4096, block_size=4096)

    region_info = {
        "dram": {
            "total_bytes": ceil.dram_total,
            "reserved_bytes": ceil.dram_reserved,
            "reserved_reason": ceil.dram_reserved_reason,
        },
        "hbm": {
            "total_bytes": ceil.hbm_total,
            "reserved_bytes": ceil.hbm_reserved,
            "reserved_reason": ceil.hbm_reserved_reason,
        },
        "nvme": {
            "total_bytes": ceil.disk_total,
            "reserved_bytes": ceil.disk_reserved,
            "reserved_reason": ceil.disk_reserved_reason,
        },
    }

    result = run_protocol(
        regions={"dram": dram, "hbm": hbm, "nvme": nvme},
        region_info=region_info,
        block_size=4096,
        num_rounds=1000,
    )

    if os.path.exists(nvme_file):
        os.unlink(nvme_file)

    return result  # ProtocolResult dataclass
```

**Fractions**: 98% DRAM (no NUMA bind), 99% HBM, 95% disk. These are
aggressive — if OOM, fall back to 93%/96%/90%.

**Done when**: `result.passed is True` and `result.rounds_passed == 1000`.

---

### Task 5: Restore

**Goal**: Reinstall Docker, pull the NGC image, and load the model to GPU.
This is the recovery cost.

**File**: `scripts/run_experiment.py` — `restore()` function

```python
def restore():
    t_total = time.monotonic()

    # Reinstall Docker
    t0 = time.monotonic()
    subprocess.run(
        ["sudo", "apt", "install", "-y", "docker-ce", "docker-ce-cli",
         "containerd.io"],
        check=True,
    )
    docker_install_s = time.monotonic() - t0

    # Pull image
    t0 = time.monotonic()
    subprocess.run(["docker", "pull", IMAGE], check=True)
    docker_pull_s = time.monotonic() - t0

    # Start container and load model
    t0 = time.monotonic()
    result = subprocess.run(
        ["docker", "run", "--rm", "--gpus", "all",
         "--name", "pose-inference-restored",
         IMAGE, "python3", "-c", MODEL_LOAD_SCRIPT],
        capture_output=True, text=True, check=True,
    )
    model_load_s = time.monotonic() - t0

    return {
        "time_s": time.monotonic() - t_total,
        "docker_install_s": docker_install_s,
        "docker_pull_s": docker_pull_s,
        "model_load_s": model_load_s,
    }
```

**Done when**: Container is running and model is loaded to GPU.

---

### Task 6: Post-Wipe Inference

**Goal**: Generate one token from the same prompt. Proves the system fully
recovered.

**File**: `scripts/run_experiment.py` — `post_wipe_inference()` function

Same inference script as Task 2, running in the restored container. Compare
the output token — it should be identical (same model, same prompt,
deterministic generation).

**Done when**: Returns a token and timing. Token matches baseline (or is
valid if generation is non-deterministic).

---

### Task 7: Report Generation

**Goal**: Assemble all results into `reports/experiment_report.json`.

**File**: `scripts/run_experiment.py` — `main()` function

```python
def main():
    prep()

    inventory = compute_ceilings(
        disk_path="/tmp",
        dram_fraction=0.98,
        hbm_fraction=0.99,
        disk_fraction=0.95,
    )

    report = {
        "memory_inventory": {
            "dram_total_bytes": inventory.dram_total,
            "dram_wipeable_bytes": inventory.dram_wipeable,
            "dram_reserved_bytes": inventory.dram_reserved,
            "dram_reserved_reason": inventory.dram_reserved_reason,
            "hbm_total_bytes": inventory.hbm_total,
            "hbm_wipeable_bytes": inventory.hbm_wipeable,
            "hbm_reserved_bytes": inventory.hbm_reserved,
            "hbm_reserved_reason": inventory.hbm_reserved_reason,
            "disk_total_bytes": inventory.disk_total,
            "disk_wipeable_bytes": inventory.disk_wipeable,
            "disk_reserved_bytes": inventory.disk_reserved,
            "disk_reserved_reason": inventory.disk_reserved_reason,
            "total_addressable_bytes": inventory.total_physical,
            "total_wipeable_bytes": inventory.total_wipeable,
        },
        "host_footprint_bytes": host_footprint_bytes(),
    }

    report["baseline_inference"] = baseline_inference()
    report["teardown"] = teardown()

    wipe_result = wipe_and_prove()
    report["wipe"] = {
        "time_s": wipe_result.fill_time_s,
        "coverage_pct": round(wipe_result.coverage * 100, 4),
        "rounds_passed": wipe_result.rounds_passed,
        "rounds_total": wipe_result.rounds_total,
        "verify_time_s": wipe_result.verify_time_s,
        "per_region": [
            {
                "name": rm.name,
                "wiped_bytes": rm.wiped_bytes,
                "total_bytes": rm.total_bytes,
                "reserved_bytes": rm.reserved_bytes,
                "reserved_reason": rm.reserved_reason,
                "fill_time_s": rm.fill_time_s,
                "fill_throughput_gbps": rm.fill_throughput_gbps,
            }
            for rm in wipe_result.region_metrics
        ],
    }

    report["restore"] = restore()
    report["post_wipe_inference"] = post_wipe_inference()

    report["total_downtime_s"] = (
        report["teardown"]["time_s"]
        + report["wipe"]["time_s"]
        + report["restore"]["time_s"]
    )

    os.makedirs("reports", exist_ok=True)
    with open("reports/experiment_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f" EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f" Coverage:            {report['wipe']['coverage_pct']:.2f}%")
    print(f" Challenges:          {report['wipe']['rounds_passed']}/{report['wipe']['rounds_total']}")
    print(f" Wipe time:           {report['wipe']['time_s']:.1f}s")
    print(f" Total downtime:      {report['total_downtime_s']:.1f}s")
    print(f" Baseline token:      {report['baseline_inference']['token']!r}")
    print(f" Post-wipe token:     {report['post_wipe_inference']['token']!r}")
    print(f"{'='*60}")
```

**Report file**: `reports/experiment_report.json`

---

## Execution

```bash
# From your Mac:
ssh ubuntu@<ip>
cd ~/memory_wipes

# Run the full experiment
sudo python3 scripts/run_experiment.py
```

Needs `sudo` because teardown/restore use `apt` and the wipe allocates
nearly all system memory.

---

## Failure Modes

| Failure | Cause | Recovery |
|---------|-------|----------|
| OOM during wipe | Fractions too aggressive | Fall back to 93%/96%/90% |
| Docker pull fails during restore | Network issue | Retry with backoff |
| GPU not found after wipe | CUDA driver state lost | `sudo nvidia-smi --gpu-reset` then retry |
| Model download fails | HuggingFace CDN issue | Retry or use cached weights |
| Post-wipe token differs | Non-deterministic generation | Expected — just verify token is valid text |
