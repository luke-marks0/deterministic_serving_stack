#!/usr/bin/env python3
"""Full PoSE-DB wipe experiment.

Runs the complete cycle: baseline inference → teardown → wipe + prove →
restore → post-wipe inference. Produces reports/experiment_report.json.

Usage: sudo /home/ubuntu/memory_wipes/.venv/bin/python3 scripts/run_experiment.py
"""

import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE = "nvcr.io/nvidia/pytorch:24.05-py3"
PROMPT = "The meaning of life is"
BLOCK_SIZE = 4096

# Target fractions (applied to detected free memory per region)
DRAM_FRAC = 0.90
HBM_FRAC = 0.96
DISK_FRAC = 0.95
NUM_ROUNDS = 1000

INFERENCE_SCRIPT = '''
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers<5", "accelerate"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
import torch, time, json
t0 = time.monotonic()
from transformers import AutoModelForCausalLM, AutoTokenizer
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
print(json.dumps({"load_time_s": round(load_time, 3), "gen_time_s": round(gen_time, 3), "token": token, "gpu_mem_used_bytes": mem}))
'''

DOCKER_RUN_FLAGS = ["--rm", "--gpus", "all", "--ipc=host",
                    "--ulimit", "memlock=-1", "--ulimit", "stack=67108864"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd, check=True, capture=False, timeout=None):
    """Run a shell command, print it, return result."""
    if isinstance(cmd, str):
        print(f"  $ {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=capture, text=True,
                           check=check, timeout=timeout)
    else:
        print(f"  $ {' '.join(cmd)}")
        r = subprocess.run(cmd, capture_output=capture, text=True,
                           check=check, timeout=timeout)
    return r


def disk_used_bytes():
    st = os.statvfs("/")
    return (st.f_blocks - st.f_bfree) * st.f_frsize


def timed(label):
    """Context manager that prints and returns elapsed time."""
    class Timer:
        def __enter__(self):
            print(f"\n{'='*60}")
            print(f" {label}")
            print(f"{'='*60}")
            self.t0 = time.monotonic()
            return self
        def __exit__(self, *args):
            self.elapsed = time.monotonic() - self.t0
            print(f"  [{label}] {self.elapsed:.1f}s")
    return Timer()


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def prep():
    """Strip host to minimum. Not timed."""
    print("\n" + "=" * 60)
    print(" PREP: Stripping host to minimum")
    print("=" * 60)

    # Remove Lambda bloat
    run("sudo apt remove -y --purge "
        "python3-torch-cuda python3-tensorflow-cuda python3-jax-cuda12-plugin "
        "python3-flash-attn-cuda libmagma2 azure-cli nccl-tests "
        "nvidia-cuda-dev nvidia-cuda-toolkit libnccl-dev python3-jaxlib "
        "libboost1.74-dev mft 2>/dev/null || true")
    run("sudo apt autoremove -y 2>/dev/null || true")
    run("sudo apt clean")
    run("sudo snap remove --purge google-cloud-sdk 2>/dev/null || true")
    run("sudo snap remove --purge aws-cli 2>/dev/null || true")
    run("sudo snap remove --purge lxd 2>/dev/null || true")

    # Fix CUDA runtime symlink (architecture-aware)
    run("sudo ln -sf $(find /usr/lib -name 'libcudart.so.12*' -type f 2>/dev/null | head -1) "
        "$(dirname $(find /usr/lib -name 'libcudart.so.12*' -type f 2>/dev/null | head -1))/libcudart.so "
        "2>/dev/null || true")
    run("sudo ldconfig")

    # Verify essentials
    run("nvidia-smi --query-gpu=name --format=csv,noheader")
    run("python3 -c \"import ctypes; ctypes.CDLL('libcudart.so'); print('CUDA runtime: OK')\"")
    run("docker --version")

    footprint = disk_used_bytes()
    print(f"\n  Host footprint: {footprint / (1024**3):.1f} GiB")
    return footprint


def baseline_inference():
    """Pull NGC image, run container, generate one token."""
    with timed("BASELINE INFERENCE") as t:
        # Pull
        t0 = time.monotonic()
        run(f"sudo docker pull {IMAGE}")
        pull_time = time.monotonic() - t0

        # Run inference
        r = run(
            ["sudo", "docker", "run", *DOCKER_RUN_FLAGS, "--name", "pose-baseline",
             IMAGE, "python3", "-c", INFERENCE_SCRIPT],
            capture=True,
        )
        result = json.loads(r.stdout.strip().split("\n")[-1])
        result["pull_time_s"] = round(pull_time, 3)
        result["total_time_s"] = round(t.elapsed if hasattr(t, 'elapsed') else time.monotonic() - t.t0, 3)
        return result


def teardown():
    """Remove container, image, Docker itself."""
    with timed("TEARDOWN") as t:
        disk_before = disk_used_bytes()

        run("sudo docker rm -f pose-baseline 2>/dev/null || true")
        run(f"sudo docker rmi -f {IMAGE} 2>/dev/null || true")
        run("sudo docker system prune -af 2>/dev/null || true")

        # Remove Docker
        run("sudo apt remove -y --purge docker-ce docker-ce-cli containerd.io "
            "docker-buildx-plugin docker-compose-plugin 2>/dev/null || true")
        run("sudo rm -rf /var/lib/docker /var/lib/containerd")
        run("sudo apt autoremove -y 2>/dev/null || true")

        # Remove CUDA libs we don't need (keep libcudart.so and libcuda.so)
        run("sudo find /usr/lib -name 'libnvidia-nvvm.so*' -o -name 'libnvidia-opencl.so*' "
            "-o -name 'libnvidia-ptxjitcompiler.so*' -o -name 'libcudadebugger.so*' "
            "| xargs sudo rm -f 2>/dev/null || true")

        disk_after = disk_used_bytes()
        return {
            "time_s": round(time.monotonic() - t.t0, 3),
            "disk_freed_bytes": disk_before - disk_after,
        }


def wipe_and_prove():
    """Run PoSE protocol with aggressive fractions."""
    with timed("WIPE + PROVE"):
        # Add our package to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from pose.detect import compute_ceilings
        from pose.protocol import run_protocol
        from pose.memory.dram import DramRegion
        from pose.memory.hbm import HbmRegion
        from pose.memory.nvme import NvmeRegion

        ceil = compute_ceilings(
            disk_path="/tmp",
            dram_fraction=DRAM_FRAC,
            hbm_fraction=HBM_FRAC,
            disk_fraction=DISK_FRAC,
        )

        print(f"  Ceilings: DRAM={ceil.dram_wipeable/(1024**3):.1f} GiB, "
              f"HBM={ceil.hbm_wipeable/(1024**3):.1f} GiB, "
              f"Disk={ceil.disk_wipeable/(1024**3):.1f} GiB")

        nvme_file = "/tmp/pose_wipe.bin"
        dram = DramRegion(size_bytes=ceil.dram_wipeable, block_size=BLOCK_SIZE)
        hbm = HbmRegion(size_bytes=ceil.hbm_wipeable, block_size=BLOCK_SIZE)
        nvme = NvmeRegion(nvme_file, num_blocks=ceil.disk_wipeable // BLOCK_SIZE,
                          block_size=BLOCK_SIZE)

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
            block_size=BLOCK_SIZE,
            num_rounds=NUM_ROUNDS,
        )

        if os.path.exists(nvme_file):
            os.unlink(nvme_file)

        return result, ceil


def restore():
    """Reinstall Docker, pull image, load model."""
    with timed("RESTORE") as t:
        # Reinstall Docker
        t0 = time.monotonic()
        run("sudo apt install -y docker-ce docker-ce-cli containerd.io")
        docker_install_s = time.monotonic() - t0

        # Pull image
        t0 = time.monotonic()
        run(f"sudo docker pull {IMAGE}")
        docker_pull_s = time.monotonic() - t0

        return {
            "time_s": round(time.monotonic() - t.t0, 3),
            "docker_install_s": round(docker_install_s, 3),
            "docker_pull_s": round(docker_pull_s, 3),
        }


def post_wipe_inference():
    """Generate one token to prove recovery."""
    with timed("POST-WIPE INFERENCE") as t:
        r = run(
            ["sudo", "docker", "run", *DOCKER_RUN_FLAGS, "--name", "pose-restored",
             IMAGE, "python3", "-c", INFERENCE_SCRIPT],
            capture=True,
        )
        result = json.loads(r.stdout.strip().split("\n")[-1])
        result["total_time_s"] = round(time.monotonic() - t.t0, 3)
        return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs("reports", exist_ok=True)

    # --- Prep (not timed) ---
    host_footprint = prep()

    # --- Memory inventory ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from pose.detect import compute_ceilings
    ceil = compute_ceilings(
        disk_path="/tmp",
        dram_fraction=DRAM_FRAC,
        hbm_fraction=HBM_FRAC,
        disk_fraction=DISK_FRAC,
    )

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "memory_inventory": {
            "dram_total_bytes": ceil.dram_total,
            "dram_wipeable_bytes": ceil.dram_wipeable,
            "dram_reserved_bytes": ceil.dram_reserved,
            "dram_reserved_reason": ceil.dram_reserved_reason,
            "hbm_total_bytes": ceil.hbm_total,
            "hbm_wipeable_bytes": ceil.hbm_wipeable,
            "hbm_reserved_bytes": ceil.hbm_reserved,
            "hbm_reserved_reason": ceil.hbm_reserved_reason,
            "disk_total_bytes": ceil.disk_total,
            "disk_wipeable_bytes": ceil.disk_wipeable,
            "disk_reserved_bytes": ceil.disk_reserved,
            "disk_reserved_reason": ceil.disk_reserved_reason,
            "total_addressable_bytes": ceil.total_physical,
            "total_wipeable_bytes": ceil.total_wipeable,
        },
        "host_footprint_bytes": host_footprint,
    }

    # --- Phase 1: Baseline inference ---
    report["baseline_inference"] = baseline_inference()

    # --- Phase 2: Teardown ---
    report["teardown"] = teardown()

    # --- Phase 3: Wipe + prove ---
    wipe_result, _ = wipe_and_prove()
    report["wipe"] = {
        "time_s": round(wipe_result.fill_time_s, 3),
        "coverage_pct": round(wipe_result.coverage * 100, 4),
        "rounds_passed": wipe_result.rounds_passed,
        "rounds_total": wipe_result.rounds_total,
        "verify_time_s": round(wipe_result.verify_time_s, 4),
        "per_region": [
            {
                "name": rm.name,
                "wiped_bytes": rm.wiped_bytes,
                "total_bytes": rm.total_bytes,
                "reserved_bytes": rm.reserved_bytes,
                "reserved_reason": rm.reserved_reason,
                "fill_time_s": round(rm.fill_time_s, 3),
                "fill_throughput_gbps": round(rm.fill_throughput_gbps, 2),
            }
            for rm in wipe_result.region_metrics
        ],
    }

    # --- Phase 4: Restore ---
    report["restore"] = restore()

    # --- Phase 5: Post-wipe inference ---
    report["post_wipe_inference"] = post_wipe_inference()

    # --- Computed metrics ---
    report["total_downtime_s"] = round(
        report["teardown"]["time_s"]
        + report["wipe"]["time_s"]
        + report["restore"]["time_s"],
        3,
    )
    report["resume_time_s"] = round(
        report["restore"]["time_s"]
        + report["post_wipe_inference"]["total_time_s"],
        3,
    )

    # --- Write report ---
    report_path = "reports/experiment_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f" EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    inv = report["memory_inventory"]
    print(f" Memory: {inv['total_addressable_bytes']/(1024**3):.1f} GiB total, "
          f"{inv['total_wipeable_bytes']/(1024**3):.1f} GiB wipeable")
    print(f" Coverage:          {report['wipe']['coverage_pct']:.2f}%")
    print(f" Challenges:        {report['wipe']['rounds_passed']}/{report['wipe']['rounds_total']}")
    print(f" Wipe time:         {report['wipe']['time_s']:.1f}s")
    print(f" Total downtime:    {report['total_downtime_s']:.1f}s")
    print(f" Resume time:       {report['resume_time_s']:.1f}s")
    print(f" Baseline token:    {report['baseline_inference']['token']!r}")
    print(f" Post-wipe token:   {report['post_wipe_inference']['token']!r}")
    print(f" Report:            {report_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
