#!/usr/bin/env python3
"""Orchestrate benchmark runs across multiple instances.

Usage:
    python3 scripts/sweep.py
"""

import json
import os
import subprocess
import sys
import time

INSTANCES = {
    "gh200": "192.222.57.125",
    "a10": "150.136.90.120",
}
METHODS = ["crypto", "devurandom"]
DISK_GB = 500


def ssh(ip, cmd, timeout=3600):
    """Run a command on the remote instance via SSH."""
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", f"ubuntu@{ip}", cmd],
        capture_output=True, text=True, timeout=timeout,
    )


def deploy(ip):
    """rsync codebase to instance."""
    subprocess.run(
        ["rsync", "-avz",
         "--exclude", ".git", "--exclude", "__pycache__",
         "--exclude", ".venv", "--exclude", "reports",
         "--exclude", "memory-sanitization",
         ".", f"ubuntu@{ip}:~/memory_wipes/"],
        check=True, capture_output=True,
    )


def setup(ip):
    """Install uv, sync deps."""
    ssh(ip, "cd ~/memory_wipes && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh 2>/dev/null && "
            "export PATH=\"$HOME/.local/bin:$PATH\" && "
            "uv sync --extra dev 2>/dev/null")


def run_benchmark(ip, method):
    """Run benchmark.py on the instance via nohup, poll for completion."""
    # Start via nohup
    log = f"/tmp/benchmark_{method}.log"
    ssh(ip, f"cd ~/memory_wipes && "
            f"sudo nohup .venv/bin/python3 -u scripts/benchmark.py "
            f"--method {method} --disk-gb {DISK_GB} "
            f"> {log} 2>&1 &")

    # Poll for completion
    while True:
        time.sleep(30)
        r = ssh(ip, f"sudo tail -1 {log}")
        if "EXPERIMENT COMPLETE" in r.stdout or "Report:" in r.stdout:
            break
        if r.returncode != 0:
            break
        # Check if process is still alive
        r2 = ssh(ip, "pgrep -f benchmark.py")
        if r2.returncode != 0:
            break

    # Print log tail
    r = ssh(ip, f"sudo cat {log}")
    print(r.stdout[-500:])


def pull_report(ip, method):
    """Pull the report JSON back to local machine."""
    r = ssh(ip, "ls ~/memory_wipes/reports/*.json 2>/dev/null")
    for line in r.stdout.strip().splitlines():
        if method in line:
            local = f"reports/{os.path.basename(line)}"
            subprocess.run(
                ["scp", f"ubuntu@{ip}:{line}", local],
                check=True, capture_output=True,
            )
            return local
    return None


def restore_instance(ip):
    """Reinstall Docker and reset GPU after a run."""
    ssh(ip, "sudo apt install -y docker-ce docker-ce-cli containerd.io 2>/dev/null")
    ssh(ip, "sudo nvidia-smi --gpu-reset 2>/dev/null")


def print_comparison(report_files):
    """Print a comparison table from the collected reports."""
    reports = {}
    for path in report_files:
        with open(path) as f:
            r = json.load(f)
        gpu = r.get("gpu_name", "unknown")
        # Extract short GPU name (e.g. "GH200" from "NVIDIA GH200")
        gpu_short = gpu.split()[-1] if gpu != "unknown" else "unknown"
        key = f"{gpu_short}-{r['method']}"
        reports[key] = r

    keys = sorted(reports.keys())
    header = f"{'':>25s}  " + "  ".join(f"{k:>18s}" for k in keys)
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    def row(label, fn):
        vals = "  ".join(f"{fn(reports[k]):>18s}" for k in keys)
        print(f"{label:>25s}  {vals}")

    row("Coverage %", lambda r: f"{r['wipe']['coverage_pct']:.2f}")
    row("Wipe time (s)", lambda r: f"{r['wipe']['total_time_s']:.1f}")
    for region in ["dram", "hbm", "disk"]:
        row(f"  {region} (GiB/s)", lambda r, rg=region: next(
            (f"{p['throughput_gbps']:.2f}" for p in r['wipe']['per_region'] if p['name'] == rg), "n/a"))
    row("Challenge (s)", lambda r: f"{r['wipe'].get('verify_time_s', 0):.3f}")
    row("Total downtime (s)", lambda r: f"{r['total_downtime_s']:.1f}")
    row("Resume (s)", lambda r: f"{r['resume_time_s']:.1f}")
    row("Tokens match", lambda r: "yes" if r['baseline_inference']['token'] == r['post_wipe_inference']['token'] else "NO")
    print(sep)


def main():
    os.makedirs("reports", exist_ok=True)
    report_files = []

    for name, ip in INSTANCES.items():
        print(f"\n{'='*60}")
        print(f" Instance: {name} ({ip})")
        print(f"{'='*60}")

        deploy(ip)
        setup(ip)

        for method in METHODS:
            print(f"\n--- {name} / {method} ---")
            restore_instance(ip)
            run_benchmark(ip, method)
            path = pull_report(ip, method)
            if path:
                report_files.append(path)
                print(f"  Report: {path}")
            else:
                print(f"  WARNING: No report found!")

    if report_files:
        print(f"\n{'='*60}")
        print(f" COMPARISON")
        print(f"{'='*60}")
        print_comparison(report_files)


if __name__ == "__main__":
    main()
