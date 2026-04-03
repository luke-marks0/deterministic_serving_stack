#!/usr/bin/env python3
"""Benchmark a single PoSE-DB wipe cycle.

Usage:
    sudo .venv/bin/python3 scripts/benchmark.py --method crypto --disk-gb 500
    sudo .venv/bin/python3 scripts/benchmark.py --method devurandom --disk-gb 500
"""

import argparse
import json
import os
import secrets
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pose.tracer import Tracer
from pose.hwinfo import collect_hwinfo
from pose.detect import compute_ceilings
from pose.noise import generate_noise_bulk, generate_noise_multicore, generate_block
from pose.memory.dram import DramRegion
from pose.memory.hbm import HbmRegion
from pose.memory.nvme import NvmeRegion
from pose.verifier import Verifier
from pose.prover import Prover
from pose.devurandom import pregen_urandom, stream_from_file, verify_from_file

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

IMAGE = "nvcr.io/nvidia/pytorch:24.05-py3"
BLOCK_SIZE = 4096
NUM_ROUNDS = 1000
DRAM_FRAC = 0.80
HBM_FRAC = 0.96
DISK_FRAC = 0.95

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
        print(f"  $ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, capture_output=capture, text=True,
                           check=check, timeout=timeout)
    else:
        print(f"  $ {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, capture_output=capture, text=True,
                           check=check, timeout=timeout)
    return r


def disk_used_bytes():
    st = os.statvfs("/")
    return (st.f_blocks - st.f_bfree) * st.f_frsize


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def prep(tracer):
    """Strip host to minimum. Not included in downtime metrics but IS traced."""
    print("\n" + "=" * 60, flush=True)
    print(" PREP: Stripping host to minimum", flush=True)
    print("=" * 60, flush=True)

    with tracer.step("prep.apt_remove"):
        run("sudo apt remove -y --purge "
            "python3-torch-cuda python3-tensorflow-cuda python3-jax-cuda12-plugin "
            "python3-flash-attn-cuda libmagma2 azure-cli nccl-tests "
            "nvidia-cuda-dev nvidia-cuda-toolkit libnccl-dev python3-jaxlib "
            "libboost1.74-dev mft 2>/dev/null || true")
        run("sudo apt autoremove -y 2>/dev/null || true")
        run("sudo apt clean")

    with tracer.step("prep.snap_remove"):
        run("sudo snap remove --purge google-cloud-sdk 2>/dev/null || true")
        run("sudo snap remove --purge aws-cli 2>/dev/null || true")
        run("sudo snap remove --purge lxd 2>/dev/null || true")

    with tracer.step("prep.fix_cuda"):
        run("sudo ln -sf $(find /usr/lib -name 'libcudart.so.12*' -type f 2>/dev/null | head -1) "
            "$(dirname $(find /usr/lib -name 'libcudart.so.12*' -type f 2>/dev/null | head -1))/libcudart.so "
            "2>/dev/null || true")
        run("sudo ldconfig")

    # Verify essentials
    run("nvidia-smi --query-gpu=name --format=csv,noheader")
    run("python3 -c \"import ctypes; ctypes.CDLL('libcudart.so'); print('CUDA runtime: OK')\"")
    # Docker may have been removed by a previous run — reinstall if missing
    run("docker --version 2>/dev/null || sudo apt install -y docker-ce docker-ce-cli containerd.io")

    footprint = disk_used_bytes()
    print(f"\n  Host footprint: {footprint / (1024**3):.1f} GiB", flush=True)
    return footprint


def baseline_inference(tracer):
    """Pull NGC image, run container, generate one token."""
    print("\n" + "=" * 60, flush=True)
    print(" BASELINE INFERENCE", flush=True)
    print("=" * 60, flush=True)

    with tracer.step("baseline.docker_pull"):
        run(f"sudo docker pull {IMAGE}")

    with tracer.step("baseline.inference"):
        r = run(
            ["sudo", "docker", "run", *DOCKER_RUN_FLAGS, "--name", "pose-baseline",
             IMAGE, "python3", "-c", INFERENCE_SCRIPT],
            capture=True,
        )
    result = json.loads(r.stdout.strip().split("\n")[-1])
    return result


def teardown(tracer):
    """Remove container, image, Docker itself."""
    print("\n" + "=" * 60, flush=True)
    print(" TEARDOWN", flush=True)
    print("=" * 60, flush=True)

    disk_before = disk_used_bytes()

    with tracer.step("teardown.docker_rm"):
        run("sudo docker rm -f pose-baseline 2>/dev/null || true")

    with tracer.step("teardown.docker_rmi"):
        run(f"sudo docker rmi -f {IMAGE} 2>/dev/null || true")
        run("sudo docker system prune -af 2>/dev/null || true")

    with tracer.step("teardown.docker_remove"):
        run("sudo apt remove -y --purge docker-ce docker-ce-cli containerd.io "
            "docker-buildx-plugin docker-compose-plugin 2>/dev/null || true")
        run("sudo rm -rf /var/lib/docker /var/lib/containerd")
        run("sudo apt autoremove -y 2>/dev/null || true")

    with tracer.step("teardown.cuda_cleanup"):
        run("sudo find /usr/lib -name 'libnvidia-nvvm.so*' -o -name 'libnvidia-opencl.so*' "
            "-o -name 'libnvidia-ptxjitcompiler.so*' -o -name 'libcudadebugger.so*' "
            "| xargs sudo rm -f 2>/dev/null || true")

    disk_after = disk_used_bytes()
    return {
        "disk_freed_bytes": disk_before - disk_after,
    }


def _noise_iter(seed, start_block, num_blocks, block_size, num_workers):
    """Pick single-core or multi-core noise generation."""
    if num_workers > 1:
        return generate_noise_multicore(seed, start_block, num_blocks, block_size, num_workers)
    return generate_noise_bulk(seed, start_block, num_blocks, block_size)


def wipe_crypto(tracer, ceilings, disk_bytes, num_workers=0):
    """Wipe using AES-CTR inline generation."""
    label = f"crypto / AES-CTR / {num_workers} workers" if num_workers else "crypto / AES-CTR"
    print(f"\n{'='*60}", flush=True)
    print(f" WIPE ({label})", flush=True)
    print("=" * 60, flush=True)

    nvme_file = "/tmp/pose_wipe.bin"
    dram = DramRegion(size_bytes=ceilings.dram_wipeable, block_size=BLOCK_SIZE)
    hbm = HbmRegion(size_bytes=ceilings.hbm_wipeable, block_size=BLOCK_SIZE)
    nvme = NvmeRegion(nvme_file, num_blocks=disk_bytes // BLOCK_SIZE, block_size=BLOCK_SIZE)

    regions = {"dram": dram, "hbm": hbm, "disk": nvme}
    prover = Prover(regions=regions, block_size=BLOCK_SIZE)
    verifier = Verifier(total_blocks=prover.total_blocks, block_size=BLOCK_SIZE)

    region_metrics = []
    global_offset = 0

    # Fill DRAM
    with tracer.step("wipe.fill_dram", bytes=ceilings.dram_wipeable):
        t0 = time.monotonic()
        chunk_iter = _noise_iter(verifier.seed, global_offset, dram.num_blocks, BLOCK_SIZE, num_workers)
        prover.fill_region_bulk("dram", chunk_iter)
        elapsed = time.monotonic() - t0
    region_metrics.append({
        "name": "dram", "wiped_bytes": dram.num_blocks * BLOCK_SIZE,
        "total_bytes": ceilings.dram_total, "reserved_bytes": ceilings.dram_reserved,
        "fill_time_s": round(elapsed, 3),
        "throughput_gbps": round(dram.num_blocks * BLOCK_SIZE / elapsed / (1024**3), 2) if elapsed > 0 else 0,
    })
    global_offset += dram.num_blocks
    print(f"  DRAM: {dram.num_blocks * BLOCK_SIZE / (1024**3):.1f} GiB in {elapsed:.1f}s", flush=True)

    # Fill HBM
    with tracer.step("wipe.fill_hbm", bytes=ceilings.hbm_wipeable):
        t0 = time.monotonic()
        chunk_iter = _noise_iter(verifier.seed, global_offset, hbm.num_blocks, BLOCK_SIZE, num_workers)
        prover.fill_region_bulk("hbm", chunk_iter)
        elapsed = time.monotonic() - t0
    region_metrics.append({
        "name": "hbm", "wiped_bytes": hbm.num_blocks * BLOCK_SIZE,
        "total_bytes": ceilings.hbm_total, "reserved_bytes": ceilings.hbm_reserved,
        "fill_time_s": round(elapsed, 3),
        "throughput_gbps": round(hbm.num_blocks * BLOCK_SIZE / elapsed / (1024**3), 2) if elapsed > 0 else 0,
    })
    global_offset += hbm.num_blocks
    print(f"  HBM: {hbm.num_blocks * BLOCK_SIZE / (1024**3):.1f} GiB in {elapsed:.1f}s", flush=True)

    # Fill disk
    with tracer.step("wipe.fill_disk", bytes=disk_bytes):
        t0 = time.monotonic()
        chunk_iter = _noise_iter(verifier.seed, global_offset, nvme.num_blocks, BLOCK_SIZE, num_workers)
        prover.fill_region_bulk("disk", chunk_iter)
        elapsed = time.monotonic() - t0
    region_metrics.append({
        "name": "disk", "wiped_bytes": nvme.num_blocks * BLOCK_SIZE,
        "total_bytes": ceilings.disk_total, "reserved_bytes": ceilings.disk_reserved,
        "fill_time_s": round(elapsed, 3),
        "throughput_gbps": round(nvme.num_blocks * BLOCK_SIZE / elapsed / (1024**3), 2) if elapsed > 0 else 0,
    })
    global_offset += nvme.num_blocks
    print(f"  Disk: {nvme.num_blocks * BLOCK_SIZE / (1024**3):.1f} GiB in {elapsed:.1f}s", flush=True)

    # Challenge-response
    with tracer.step("wipe.challenge"):
        t0 = time.monotonic()
        rounds_passed = 0
        for _ in range(NUM_ROUNDS):
            idx = verifier.challenge()
            response = prover.respond(idx)
            if verifier.verify(idx, response):
                rounds_passed += 1
        verify_time = time.monotonic() - t0
    print(f"  Challenges: {rounds_passed}/{NUM_ROUNDS} in {verify_time:.3f}s", flush=True)

    # Cleanup
    dram.close()
    hbm.close()
    nvme.close()
    if os.path.exists(nvme_file):
        os.unlink(nvme_file)

    total_wiped = sum(rm["wiped_bytes"] for rm in region_metrics)
    total_physical = sum(rm["total_bytes"] for rm in region_metrics)
    total_fill_time = sum(rm["fill_time_s"] for rm in region_metrics)

    return {
        "total_time_s": round(total_fill_time + verify_time, 3),
        "coverage_pct": round(total_wiped / total_physical * 100, 4) if total_physical > 0 else 0,
        "rounds_passed": rounds_passed,
        "rounds_total": NUM_ROUNDS,
        "verify_time_s": round(verify_time, 4),
        "per_region": region_metrics,
    }


def wipe_crypto_parallel(tracer, ceilings, disk_bytes, num_workers=0):
    """Wipe using AES-CTR with all three regions filled in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print("\n" + "=" * 60, flush=True)
    print(" WIPE (crypto / AES-CTR / PARALLEL)", flush=True)
    print("=" * 60, flush=True)

    nvme_file = "/tmp/pose_wipe.bin"
    dram = DramRegion(size_bytes=ceilings.dram_wipeable, block_size=BLOCK_SIZE)
    hbm = HbmRegion(size_bytes=ceilings.hbm_wipeable, block_size=BLOCK_SIZE)
    nvme = NvmeRegion(nvme_file, num_blocks=disk_bytes // BLOCK_SIZE, block_size=BLOCK_SIZE)

    regions = {"dram": dram, "hbm": hbm, "disk": nvme}
    prover = Prover(regions=regions, block_size=BLOCK_SIZE)
    verifier = Verifier(total_blocks=prover.total_blocks, block_size=BLOCK_SIZE)

    # Each thread fills one region with its own AES-CTR stream
    region_list = [
        ("dram", dram, 0, ceilings.dram_wipeable, ceilings.dram_total, ceilings.dram_reserved),
        ("hbm", hbm, dram.num_blocks, ceilings.hbm_wipeable, ceilings.hbm_total, ceilings.hbm_reserved),
        ("disk", nvme, dram.num_blocks + hbm.num_blocks, disk_bytes, ceilings.disk_total, ceilings.disk_total - disk_bytes),
    ]

    region_metrics = []

    def fill_one(name, region, global_offset, size, total, reserved):
        t0 = time.monotonic()
        chunk_iter = _noise_iter(verifier.seed, global_offset, region.num_blocks, BLOCK_SIZE, num_workers)
        prover.fill_region_bulk(name, chunk_iter)
        elapsed = time.monotonic() - t0
        print(f"  {name}: {size / (1024**3):.1f} GiB in {elapsed:.1f}s ({size / elapsed / (1024**3):.2f} GiB/s)", flush=True)
        return {
            "name": name, "wiped_bytes": size, "total_bytes": total,
            "reserved_bytes": reserved, "fill_time_s": round(elapsed, 3),
            "throughput_gbps": round(size / elapsed / (1024**3), 2) if elapsed > 0 else 0,
        }

    # Launch all three fills in parallel
    with tracer.step("wipe.fill_parallel", bytes=ceilings.dram_wipeable + ceilings.hbm_wipeable + disk_bytes):
        wall_t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(fill_one, *args): args[0] for args in region_list}
            for future in as_completed(futures):
                region_metrics.append(future.result())
        wall_elapsed = time.monotonic() - wall_t0

    print(f"  TOTAL (parallel wall time): {wall_elapsed:.1f}s", flush=True)

    # Challenge-response
    with tracer.step("wipe.challenge"):
        t0 = time.monotonic()
        rounds_passed = 0
        for _ in range(NUM_ROUNDS):
            idx = verifier.challenge()
            response = prover.respond(idx)
            if verifier.verify(idx, response):
                rounds_passed += 1
        verify_time = time.monotonic() - t0

    # Cleanup
    for r in [dram, hbm, nvme]:
        r.close()
    if os.path.exists(nvme_file):
        os.unlink(nvme_file)

    total_wiped = sum(rm["wiped_bytes"] for rm in region_metrics)
    total_physical = sum(rm["total_bytes"] for rm in region_metrics)

    return {
        "total_time_s": round(wall_elapsed + verify_time, 3),
        "coverage_pct": round(total_wiped / total_physical * 100, 4) if total_physical > 0 else 0,
        "rounds_passed": rounds_passed,
        "rounds_total": NUM_ROUNDS,
        "verify_time_s": round(verify_time, 4),
        "per_region": region_metrics,
    }


def wipe_devurandom(tracer, ceilings, disk_bytes):
    """Wipe using pre-generated /dev/urandom file."""
    print("\n" + "=" * 60, flush=True)
    print(" WIPE (devurandom / parallel /dev/urandom)", flush=True)
    print("=" * 60, flush=True)

    noise_file = "/tmp/verifier_noise.bin"
    nvme_file = "/tmp/pose_wipe.bin"
    chunk_size = 256 * 1024 * 1024  # 256 MiB

    total_noise = ceilings.dram_wipeable + ceilings.hbm_wipeable + disk_bytes

    dram_wipeable = ceilings.dram_wipeable

    # Allocate ALL regions BEFORE pregen — this grabs physical pages before
    # the pregen fills the page cache. HBM via cudaMalloc, DRAM via mmap
    # (lazy — pages allocated on first touch), NVMe via file create.
    hbm = HbmRegion(size_bytes=ceilings.hbm_wipeable, block_size=BLOCK_SIZE)
    dram = DramRegion(size_bytes=dram_wipeable, block_size=BLOCK_SIZE)
    nvme = NvmeRegion(nvme_file, num_blocks=disk_bytes // BLOCK_SIZE, block_size=BLOCK_SIZE)

    # Pre-generate noise file
    total_noise = dram_wipeable + ceilings.hbm_wipeable + disk_bytes
    with tracer.step("wipe.pregen_urandom", bytes=total_noise):
        t0 = time.monotonic()
        pregen_urandom(noise_file, total_bytes=total_noise)
        elapsed = time.monotonic() - t0
    print(f"  Pregen: {total_noise / (1024**3):.1f} GiB in {elapsed:.1f}s", flush=True)

    # Drop page cache — stream_from_file uses O_DIRECT to avoid refilling it
    run("sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true", check=False)

    dram_offset = 0
    hbm_offset = dram_wipeable
    disk_offset = dram_wipeable + ceilings.hbm_wipeable

    region_metrics = []
    blocks_per_chunk = chunk_size // BLOCK_SIZE

    # Fill DRAM
    with tracer.step("wipe.fill_dram", bytes=dram_wipeable):
        t0 = time.monotonic()
        chunks = stream_from_file(noise_file, dram_offset, dram_wipeable, chunk_size=chunk_size)
        for i, chunk in enumerate(chunks):
            dram.write_range(i * blocks_per_chunk, chunk)
        elapsed = time.monotonic() - t0
    region_metrics.append({
        "name": "dram", "wiped_bytes": dram.num_blocks * BLOCK_SIZE,
        "total_bytes": ceilings.dram_total, "reserved_bytes": ceilings.dram_reserved,
        "fill_time_s": round(elapsed, 3),
        "throughput_gbps": round(dram.num_blocks * BLOCK_SIZE / elapsed / (1024**3), 2) if elapsed > 0 else 0,
    })
    print(f"  DRAM: {dram.num_blocks * BLOCK_SIZE / (1024**3):.1f} GiB in {elapsed:.1f}s", flush=True)

    # Fill HBM
    with tracer.step("wipe.fill_hbm", bytes=ceilings.hbm_wipeable):
        t0 = time.monotonic()
        chunks = stream_from_file(noise_file, hbm_offset, ceilings.hbm_wipeable, chunk_size=chunk_size)
        for i, chunk in enumerate(chunks):
            hbm.write_range(i * blocks_per_chunk, chunk)
        elapsed = time.monotonic() - t0
    region_metrics.append({
        "name": "hbm", "wiped_bytes": hbm.num_blocks * BLOCK_SIZE,
        "total_bytes": ceilings.hbm_total, "reserved_bytes": ceilings.hbm_reserved,
        "fill_time_s": round(elapsed, 3),
        "throughput_gbps": round(hbm.num_blocks * BLOCK_SIZE / elapsed / (1024**3), 2) if elapsed > 0 else 0,
    })
    print(f"  HBM: {hbm.num_blocks * BLOCK_SIZE / (1024**3):.1f} GiB in {elapsed:.1f}s", flush=True)

    # Fill disk
    with tracer.step("wipe.fill_disk", bytes=disk_bytes):
        t0 = time.monotonic()
        chunks = stream_from_file(noise_file, disk_offset, disk_bytes, chunk_size=chunk_size)
        for i, chunk in enumerate(chunks):
            nvme.write_range(i * blocks_per_chunk, chunk)
        elapsed = time.monotonic() - t0
    region_metrics.append({
        "name": "disk", "wiped_bytes": nvme.num_blocks * BLOCK_SIZE,
        "total_bytes": ceilings.disk_total, "reserved_bytes": ceilings.disk_reserved,
        "fill_time_s": round(elapsed, 3),
        "throughput_gbps": round(nvme.num_blocks * BLOCK_SIZE / elapsed / (1024**3), 2) if elapsed > 0 else 0,
    })
    print(f"  Disk: {nvme.num_blocks * BLOCK_SIZE / (1024**3):.1f} GiB in {elapsed:.1f}s", flush=True)

    # Challenge-response using verify_from_file
    # Build a global block map so we know which offset each region starts at
    # in the noise file.
    dram_block_offset = 0
    hbm_block_offset = dram.num_blocks
    disk_block_offset = dram.num_blocks + hbm.num_blocks
    total_blocks = dram.num_blocks + hbm.num_blocks + nvme.num_blocks

    regions = {"dram": dram, "hbm": hbm, "disk": nvme}
    prover = Prover(regions=regions, block_size=BLOCK_SIZE)

    with tracer.step("wipe.challenge"):
        t0 = time.monotonic()
        rounds_passed = 0
        for _ in range(NUM_ROUNDS):
            idx = secrets.randbelow(total_blocks)
            response = prover.respond(idx)
            expected = verify_from_file(noise_file, block_index=idx, block_size=BLOCK_SIZE)
            if response == expected:
                rounds_passed += 1
        verify_time = time.monotonic() - t0
    print(f"  Challenges: {rounds_passed}/{NUM_ROUNDS} in {verify_time:.3f}s", flush=True)

    # Cleanup
    dram.close()
    hbm.close()
    nvme.close()
    if os.path.exists(nvme_file):
        os.unlink(nvme_file)
    if os.path.exists(noise_file):
        os.unlink(noise_file)

    total_wiped = sum(rm["wiped_bytes"] for rm in region_metrics)
    total_physical = sum(rm["total_bytes"] for rm in region_metrics)
    total_fill_time = sum(rm["fill_time_s"] for rm in region_metrics)

    return {
        "total_time_s": round(total_fill_time + verify_time, 3),
        "coverage_pct": round(total_wiped / total_physical * 100, 4) if total_physical > 0 else 0,
        "rounds_passed": rounds_passed,
        "rounds_total": NUM_ROUNDS,
        "verify_time_s": round(verify_time, 4),
        "per_region": region_metrics,
    }


def restore(tracer):
    """Reinstall Docker, pull image."""
    print("\n" + "=" * 60, flush=True)
    print(" RESTORE", flush=True)
    print("=" * 60, flush=True)

    with tracer.step("restore.docker_install"):
        # Kill any stale apt locks
        run("sudo pkill -9 -f unattended-upgrade 2>/dev/null || true", check=False)
        run("sudo rm -f /var/lib/dpkg/lock-frontend /var/lib/dpkg/lock "
            "/var/lib/apt/lists/lock /var/cache/apt/archives/lock 2>/dev/null || true", check=False)
        run("sudo dpkg --configure -a 2>/dev/null || true", check=False)
        run("sudo apt install -y docker-ce docker-ce-cli containerd.io")
        # Verify Docker is running
        run("sudo systemctl start docker 2>/dev/null || true", check=False)

    with tracer.step("restore.docker_pull"):
        run(f"sudo docker pull {IMAGE}")


def post_wipe_inference(tracer):
    """Generate one token to prove recovery."""
    print("\n" + "=" * 60, flush=True)
    print(" POST-WIPE INFERENCE", flush=True)
    print("=" * 60, flush=True)

    with tracer.step("restore.inference"):
        r = run(
            ["sudo", "docker", "run", *DOCKER_RUN_FLAGS, "--name", "pose-restored",
             IMAGE, "python3", "-c", INFERENCE_SCRIPT],
            capture=True,
        )
    result = json.loads(r.stdout.strip().split("\n")[-1])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark a single PoSE-DB wipe cycle")
    parser.add_argument("--method", choices=["crypto", "devurandom"], required=True)
    parser.add_argument("--disk-gb", type=int, default=500,
                        help="Disk wipe cap in GiB (default: 500)")
    parser.add_argument("--parallel", action="store_true",
                        help="Fill all regions in parallel (crypto method only)")
    parser.add_argument("--multicore", type=int, default=0,
                        help="Number of AES worker threads per region (0=single-core)")
    args = parser.parse_args()

    disk_bytes = args.disk_gb * (1024**3)
    # Align to block size
    disk_bytes = (disk_bytes // BLOCK_SIZE) * BLOCK_SIZE

    tracer = Tracer()
    hwinfo = collect_hwinfo()

    print(f"\n{'='*60}", flush=True)
    print(f" PoSE-DB Benchmark: {args.method} method", flush=True)
    print(f" Host: {hwinfo['hostname']} ({hwinfo['arch']})", flush=True)
    print(f" GPU: {hwinfo['gpu_name']}", flush=True)
    print(f" Disk cap: {args.disk_gb} GiB", flush=True)
    print(f"{'='*60}", flush=True)

    # --- Prep ---
    prep(tracer)

    # --- Memory inventory ---
    ceilings = compute_ceilings(
        disk_path="/tmp",
        dram_fraction=DRAM_FRAC,
        hbm_fraction=HBM_FRAC,
        disk_fraction=DISK_FRAC,
    )

    print(f"\n  Ceilings: DRAM={ceilings.dram_wipeable/(1024**3):.1f} GiB, "
          f"HBM={ceilings.hbm_wipeable/(1024**3):.1f} GiB, "
          f"Disk(cap)={disk_bytes/(1024**3):.1f} GiB", flush=True)

    # --- Baseline inference ---
    baseline = baseline_inference(tracer)

    # --- Teardown ---
    teardown_result = teardown(tracer)

    # --- Wipe ---
    num_workers = args.multicore if args.multicore > 0 else 0
    if args.method == "crypto" and args.parallel:
        wipe_result = wipe_crypto_parallel(tracer, ceilings, disk_bytes, num_workers=num_workers)
    elif args.method == "crypto":
        wipe_result = wipe_crypto(tracer, ceilings, disk_bytes, num_workers=num_workers)
    else:
        wipe_result = wipe_devurandom(tracer, ceilings, disk_bytes)

    # --- Restore ---
    try:
        restore(tracer)
    except Exception as e:
        print(f"  WARNING: Restore failed: {e}", flush=True)

    # --- Post-wipe inference ---
    try:
        post_wipe = post_wipe_inference(tracer)
    except Exception as e:
        print(f"  WARNING: Post-wipe inference failed: {e}", flush=True)
        post_wipe = {"token": "FAILED", "load_time_s": 0, "gen_time_s": 0, "gpu_mem_used_bytes": 0}

    # --- Compute timing from trace ---
    events = tracer.events()

    def trace_time(prefix):
        """Sum elapsed time for all trace steps starting with prefix."""
        total = 0.0
        for e in events:
            if e["step"].startswith(prefix):
                total += e["end_ts"] - e["start_ts"]
        return total

    teardown_time = trace_time("teardown.")
    wipe_time = wipe_result["total_time_s"]
    restore_time = trace_time("restore.")
    inference_time = trace_time("restore.inference")

    total_downtime = teardown_time + wipe_time + restore_time
    resume_time = restore_time + inference_time

    # --- Build report ---
    report = {
        **hwinfo,
        "method": args.method + ("-parallel" if args.parallel else "") + (f"-{args.multicore}core" if args.multicore > 0 else ""),
        "disk_cap_gib": args.disk_gb,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "memory_inventory": {
            "dram_total_bytes": ceilings.dram_total,
            "dram_wipeable_bytes": ceilings.dram_wipeable,
            "dram_reserved_bytes": ceilings.dram_reserved,
            "dram_reserved_reason": ceilings.dram_reserved_reason,
            "hbm_total_bytes": ceilings.hbm_total,
            "hbm_wipeable_bytes": ceilings.hbm_wipeable,
            "hbm_reserved_bytes": ceilings.hbm_reserved,
            "hbm_reserved_reason": ceilings.hbm_reserved_reason,
            "disk_total_bytes": ceilings.disk_total,
            "disk_wipeable_bytes": disk_bytes,
            "disk_reserved_bytes": ceilings.disk_total - disk_bytes,
            "disk_reserved_reason": f"Capped at {args.disk_gb} GiB by --disk-gb",
            "total_addressable_bytes": ceilings.dram_total + ceilings.hbm_total + ceilings.disk_total,
            "total_wipeable_bytes": ceilings.dram_wipeable + ceilings.hbm_wipeable + disk_bytes,
        },
        "baseline_inference": baseline,
        "teardown": {
            "time_s": round(teardown_time, 3),
            "disk_freed_bytes": teardown_result["disk_freed_bytes"],
        },
        "wipe": wipe_result,
        "restore": {
            "time_s": round(restore_time, 3),
        },
        "post_wipe_inference": post_wipe,
        "total_downtime_s": round(total_downtime, 3),
        "resume_time_s": round(resume_time, 3),
        "trace": events,
    }

    # --- Write report ---
    os.makedirs("reports", exist_ok=True)
    report_file = f"reports/{hwinfo['hostname']}-{args.method}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*60}", flush=True)
    print(f" EXPERIMENT COMPLETE", flush=True)
    print(f"{'='*60}", flush=True)
    inv = report["memory_inventory"]
    print(f" Method:            {args.method}")
    print(f" Memory: {inv['total_addressable_bytes']/(1024**3):.1f} GiB total, "
          f"{inv['total_wipeable_bytes']/(1024**3):.1f} GiB wipeable")
    print(f" Coverage:          {report['wipe']['coverage_pct']:.2f}%")
    print(f" Challenges:        {report['wipe']['rounds_passed']}/{report['wipe']['rounds_total']}")
    print(f" Wipe time:         {report['wipe']['total_time_s']:.1f}s")
    print(f" Total downtime:    {report['total_downtime_s']:.1f}s")
    print(f" Resume time:       {report['resume_time_s']:.1f}s")
    print(f" Baseline token:    {report['baseline_inference']['token']!r}")
    print(f" Post-wipe token:   {report['post_wipe_inference']['token']!r}")
    print(f" Report:            {report_file}")
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
