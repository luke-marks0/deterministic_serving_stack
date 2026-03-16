#!/usr/bin/env python3
"""Deterministic vLLM serving wrapper.

Validates manifest + lockfile + hardware conformance at boot,
then starts vLLM's OpenAI-compatible server with batch invariance.
All requests/responses are logged to an append-only capture file.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pkg.common.contracts import ValidationError, validate_with_schema
from pkg.common.deterministic import (
    canonical_json_bytes,
    canonical_json_text,
    compute_lockfile_digest,
    sha256_prefixed,
    utc_now_iso,
)


def _hardware_fingerprint(hw: dict[str, Any]) -> str:
    return sha256_prefixed(canonical_json_bytes(hw))


def _validate_boot(manifest: dict[str, Any], lockfile: dict[str, Any]) -> dict[str, Any]:
    """Validate manifest and lockfile at server boot. Returns conformance record."""
    validate_with_schema("manifest.v1.schema.json", manifest)
    validate_with_schema("lockfile.v1.schema.json", lockfile)

    manifest_digest = sha256_prefixed(canonical_json_bytes(manifest))
    if lockfile["manifest_digest"] != manifest_digest:
        raise ValidationError("Lockfile manifest_digest mismatch")

    expected_lockfile_digest = compute_lockfile_digest(lockfile)
    if lockfile["canonicalization"]["lockfile_digest"] != expected_lockfile_digest:
        raise ValidationError("Lockfile canonicalization.lockfile_digest mismatch")

    return {"manifest_digest": manifest_digest, "lockfile_digest": expected_lockfile_digest}


def _probe_hardware(manifest: dict[str, Any]) -> dict[str, Any]:
    """Probe GPU hardware and return conformance record."""
    expected = manifest["hardware_profile"]
    record = {
        "status": "unknown",
        "strict_hardware": manifest["runtime"]["strict_hardware"],
        "expected_fingerprint": _hardware_fingerprint(expected),
        "actual_fingerprint": "",
        "gpu_name": "",
        "compute_capability": "",
        "driver_version": "",
    }

    try:
        import torch
        if not torch.cuda.is_available():
            record["status"] = "no_gpu"
            return record

        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        record["gpu_name"] = gpu_name
        record["compute_capability"] = f"{cc[0]}.{cc[1]}"

        if cc[0] < 9:
            raise ValidationError(
                f"Batch invariance requires compute capability >= 9.0, got {cc[0]}.{cc[1]}"
            )

        # Probe via nvidia-smi for driver version
        try:
            import shutil
            if shutil.which("nvidia-smi"):
                proc = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, check=True,
                )
                record["driver_version"] = proc.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass

        record["actual_fingerprint"] = _hardware_fingerprint({
            "gpu": {"model": gpu_name, "compute_capability": f"{cc[0]}.{cc[1]}"},
        })
        record["status"] = "conformant"

        # Record software versions for provenance
        record["torch_version"] = torch.__version__
        record["cuda_version"] = torch.version.cuda or "unknown"
        try:
            import vllm
            record["vllm_version"] = getattr(vllm, "__version__", "unknown")
        except ImportError:
            record["vllm_version"] = "not_installed"

    except ImportError:
        record["status"] = "torch_not_available"

    return record


def _build_vllm_cmd(manifest: dict[str, Any], host: str, port: int) -> list[str]:
    """Build the vllm serve command from manifest settings."""
    runtime = manifest["runtime"]
    knobs = runtime["deterministic_knobs"]
    model_source = manifest["model"]["source"]
    model_id = model_source.removeprefix("hf://") if model_source.startswith("hf://") else model_source

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", os.getenv("RUNNER_MODEL_PATH", model_id),
        "--host", host,
        "--port", str(port),
        "--seed", str(knobs.get("seed", 42)),
        "--dtype", "auto",
        "--disable-log-stats",
    ]

    batch_inv = runtime.get("batch_invariance", {})
    if batch_inv.get("enforce_eager", False):
        cmd.append("--enforce-eager")

    if batch_inv.get("enabled", False):
        backend = os.getenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
        cmd.extend(["--attention-backend", backend])

    max_model_len = os.getenv("RUNNER_MAX_MODEL_LEN", "8192")
    cmd.extend(["--max-model-len", max_model_len])

    gpu_mem = os.getenv("RUNNER_GPU_MEM_UTIL", "0.90")
    cmd.extend(["--gpu-memory-utilization", gpu_mem])

    api_key = os.getenv("VLLM_API_KEY")
    if api_key:
        cmd.extend(["--api-key", api_key])

    if manifest["model"].get("trust_remote_code", False):
        cmd.append("--trust-remote-code")

    return cmd


class CaptureLog:
    """Append-only request/response log for provenance."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seq = 0
        # Write header
        with open(self.path, "w") as f:
            f.write("")

    def next_seq(self) -> int:
        """Assign a sequence number at request arrival time (under lock)."""
        with self._lock:
            self._seq += 1
            return self._seq

    def append(self, entry: dict[str, Any]) -> None:
        """Write a pre-sequenced entry. seq must already be set by next_seq()."""
        with self._lock:
            entry["captured_at"] = utc_now_iso()
            with open(self.path, "a") as f:
                f.write(canonical_json_text(entry))


class ProxyHandler(BaseHTTPRequestHandler):
    """Reverse proxy that captures requests/responses to the vLLM server."""

    vllm_port: int = 8001
    capture_log: CaptureLog | None = None
    api_key: str | None = None

    def _check_auth(self) -> bool:
        """Reject requests without valid API key (if configured)."""
        if not self.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {self.api_key}":
            return True
        error = json.dumps({"error": "Unauthorized"}).encode()
        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(error)))
        self.end_headers()
        self.wfile.write(error)
        return False

    def do_POST(self):
        if not self._check_auth():
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        # Assign sequence number at arrival time (before forwarding) so
        # logging order matches arrival order regardless of thread scheduling.
        arrival_seq = self.capture_log.next_seq() if self.capture_log else 0

        # Parse request for logging
        try:
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request_data = {"raw": body.decode("utf-8", errors="replace")}

        # Forward to vLLM
        url = f"http://127.0.0.1:{self.vllm_port}{self.path}"
        req = Request(url, data=body, method="POST")
        for key in ["Content-Type", "Authorization"]:
            val = self.headers.get(key)
            if val:
                req.add_header(key, val)

        try:
            with urlopen(req) as resp:
                resp_body = resp.read()
                status = resp.status

                # Parse response for logging
                try:
                    response_data = json.loads(resp_body)
                except json.JSONDecodeError:
                    response_data = {"raw": resp_body.decode("utf-8", errors="replace")}

                # Log the exchange with pre-assigned arrival-order seq
                if self.capture_log and self.path.startswith("/v1/"):
                    self.capture_log.append({
                        "seq": arrival_seq,
                        "endpoint": self.path,
                        "request": request_data,
                        "response": response_data,
                        "status": status,
                    })

                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)

        except URLError as exc:
            error_msg = json.dumps({"error": str(exc)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_msg)))
            self.end_headers()
            self.wfile.write(error_msg)

    def do_GET(self):
        # Allow /health without auth for load balancer probes
        if self.path != "/health" and not self._check_auth():
            return

        url = f"http://127.0.0.1:{self.vllm_port}{self.path}"
        try:
            with urlopen(url) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except URLError as exc:
            error_msg = json.dumps({"error": str(exc)}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_msg)))
            self.end_headers()
            self.wfile.write(error_msg)

    def log_message(self, format, *args):
        # Suppress default access logs
        pass


def _wait_for_vllm(port: int, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/health") as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic vLLM serving wrapper")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    parser.add_argument("--lockfile", required=True, help="Lockfile JSON path")
    parser.add_argument("--out-dir", default="/tmp/deterministic-server", help="Output directory for logs and capture")
    parser.add_argument("--host", default="0.0.0.0", help="Listen host")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (proxy)")
    parser.add_argument("--vllm-port", type=int, default=8001, help="Internal vLLM port")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    lockfile = json.loads(Path(args.lockfile).read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Validate manifest + lockfile
    print("=== Deterministic Server Boot ===")
    print(f"Model: {manifest['model']['source']}")
    print(f"Run ID: {manifest['run_id']}")
    print()

    print("[boot] Validating manifest and lockfile...")
    digests = _validate_boot(manifest, lockfile)
    print(f"  manifest_digest: {digests['manifest_digest']}")
    print(f"  lockfile_digest: {digests['lockfile_digest']}")

    # Phase 2: Hardware conformance
    print("[boot] Probing hardware...")
    hw_record = _probe_hardware(manifest)
    print(f"  GPU: {hw_record['gpu_name']}")
    print(f"  Compute capability: {hw_record['compute_capability']}")
    print(f"  Status: {hw_record['status']}")

    if hw_record["status"] != "conformant":
        if manifest["runtime"]["strict_hardware"]:
            print(f"ERROR: Hardware conformance failed: {hw_record['status']}")
            return 1
        else:
            print(f"WARNING: Hardware non-conformant ({hw_record['status']}), continuing (strict_hardware=false)")

    # Phase 3: Check batch invariance requirements
    batch_inv = manifest["runtime"].get("batch_invariance", {})
    if batch_inv.get("enabled", False):
        print("[boot] Batch invariance: ENABLED")
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    else:
        print("[boot] Batch invariance: disabled")

    # Write boot record
    boot_record = {
        "boot_time": utc_now_iso(),
        "manifest_digest": digests["manifest_digest"],
        "lockfile_digest": digests["lockfile_digest"],
        "hardware": hw_record,
        "batch_invariance": batch_inv,
        "runtime_closure_digest": lockfile.get("runtime_closure_digest", ""),
        "model": manifest["model"]["source"],
        "run_id": manifest["run_id"],
    }
    boot_path = out_dir / "boot_record.json"
    boot_path.write_text(canonical_json_text(boot_record), encoding="utf-8")
    print(f"[boot] Boot record: {boot_path}")

    # Phase 4: Set deterministic env
    knobs = manifest["runtime"]["deterministic_knobs"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(int(knobs.get("cuda_launch_blocking", True)))
    os.environ["PYTHONHASHSEED"] = "0"

    # Phase 5: Start vLLM server
    vllm_cmd = _build_vllm_cmd(manifest, "127.0.0.1", args.vllm_port)
    print(f"[boot] Starting vLLM: {' '.join(vllm_cmd)}")
    print()

    vllm_proc = subprocess.Popen(
        vllm_cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # Wait for vLLM to be ready
    print("[boot] Waiting for vLLM to be ready...")
    if not _wait_for_vllm(args.vllm_port):
        print("ERROR: vLLM server failed to start within timeout")
        vllm_proc.terminate()
        return 1

    print("[boot] vLLM is ready")
    print()

    # Phase 6: Start capture proxy
    capture_log = CaptureLog(out_dir / "capture.jsonl")
    ProxyHandler.vllm_port = args.vllm_port
    ProxyHandler.capture_log = capture_log
    ProxyHandler.api_key = os.getenv("VLLM_API_KEY")

    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        daemon_threads = True

    proxy = ThreadedHTTPServer((args.host, args.port), ProxyHandler)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()

    print(f"=== Server ready ===")
    print(f"  Endpoint: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  Model: {manifest['model']['source']}")
    print(f"  Batch invariance: {'ON' if batch_inv.get('enabled') else 'OFF'}")
    print(f"  Auth: {'API key required' if os.getenv('VLLM_API_KEY') else 'OPEN (no VLLM_API_KEY set)'}")
    print(f"  Capture log: {out_dir / 'capture.jsonl'}")
    print(f"  Boot record: {boot_path}")
    print()

    # Handle shutdown
    def shutdown(signum, frame):
        print("\n[shutdown] Stopping server...")
        proxy.shutdown()
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

        # Write shutdown record
        shutdown_record = {
            "shutdown_time": utc_now_iso(),
            "capture_entries": capture_log._seq,
            "capture_digest": sha256_prefixed(Path(capture_log.path).read_bytes()) if capture_log.path.exists() else "",
        }
        (out_dir / "shutdown_record.json").write_text(
            canonical_json_text(shutdown_record), encoding="utf-8"
        )
        print(f"[shutdown] {capture_log._seq} requests captured")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Block until vLLM exits
    try:
        vllm_proc.wait()
    except KeyboardInterrupt:
        shutdown(None, None)

    return vllm_proc.returncode or 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
