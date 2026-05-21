from __future__ import annotations

import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn

EXP_SCRIPTS = (
    Path(__file__).resolve().parents[2] / "experiments" / "prover-verifier-demo" / "scripts"
)
if str(EXP_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP_SCRIPTS))

from workloads.context import WorkloadContext  # noqa: E402
from workloads.lora_loading import LoraLoadingWorkload  # noqa: E402


class _FixedSizeHandler(BaseHTTPRequestHandler):
    """Serves N bytes of deterministic content for `data_size` from the
    server's state. Used as the fake "external" LoRA store.
    """

    def log_message(self, format: str, *args: object) -> None:
        return

    def do_GET(self) -> None:
        size = self.server.data_size  # type: ignore[attr-defined]
        body = b"L" * size
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _ThreadedServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _make_recording_ctx() -> tuple[WorkloadContext, list[bytes], list[dict]]:
    frames: list[bytes] = []
    tasks: list[dict] = []
    ctx = WorkloadContext(
        publish_frame=frames.append,
        record_task=tasks.append,
        stop_event=threading.Event(),
    )
    return ctx, frames, tasks


class TestLoraLoadingWorkload(unittest.TestCase):
    def setUp(self) -> None:
        self.lora_bytes = 4096
        self.server = _ThreadedServer(("127.0.0.1", 0), _FixedSizeHandler)
        self.server.data_size = self.lora_bytes  # type: ignore[attr-defined]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[0], self.server.server_address[1]
        self.lora_url = f"http://{host}:{port}/lora"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    def test_records_only_inference_tasks(self) -> None:
        ctx, _frames, tasks = _make_recording_ctx()
        wl = LoraLoadingWorkload(
            prompts=["a", "b"],
            lora_url=self.lora_url,
            lora_bytes=self.lora_bytes,
            seed=7,
        )
        wl.run(ctx)
        # Inferences are recorded as tasks; the LoRA download is NOT.
        self.assertEqual(len(tasks), 2)
        for t in tasks:
            self.assertEqual(t["operation"], "inference")

    def test_publishes_lora_bytes_plus_inference_frames(self) -> None:
        ctx, frames, _tasks = _make_recording_ctx()
        wl = LoraLoadingWorkload(
            prompts=["a", "b", "c"],
            lora_url=self.lora_url,
            lora_bytes=self.lora_bytes,
            seed=7,
        )
        wl.run(ctx)
        total_bytes = sum(len(f) for f in frames)
        # 3 prompts * 10 frames * 256 bytes = inference traffic.
        inference_bytes = 3 * 10 * 256
        self.assertEqual(total_bytes, self.lora_bytes + inference_bytes)

    def test_download_bytes_match_knob(self) -> None:
        ctx, _frames, _tasks = _make_recording_ctx()
        wl = LoraLoadingWorkload(
            prompts=["a"],
            lora_url=self.lora_url,
            lora_bytes=self.lora_bytes,
            seed=7,
        )
        wl.run(ctx)
        self.assertEqual(wl.download_bytes, self.lora_bytes)

    def test_unattributed_bytes_equals_download_bytes(self) -> None:
        # Unattributed = published bytes - sum(claimed_flops) where the
        # benign synth uses claimed_flops as a stand-in for bytes-on-wire.
        # The cheating gap = self.lora_bytes (i.e., the LoRA download).
        ctx, frames, tasks = _make_recording_ctx()
        wl = LoraLoadingWorkload(
            prompts=["a", "b"],
            lora_url=self.lora_url,
            lora_bytes=self.lora_bytes,
            seed=7,
        )
        wl.run(ctx)
        published = sum(len(f) for f in frames)
        claimed = sum(t["claimed_flops"] for t in tasks)
        self.assertEqual(published - claimed, self.lora_bytes)

    def test_inference_frames_are_deterministic_for_a_seed(self) -> None:
        ctx_a, fr_a, _ = _make_recording_ctx()
        ctx_b, fr_b, _ = _make_recording_ctx()
        LoraLoadingWorkload(
            prompts=["x", "y"],
            lora_url=self.lora_url,
            lora_bytes=self.lora_bytes,
            seed=7,
        ).run(ctx_a)
        LoraLoadingWorkload(
            prompts=["x", "y"],
            lora_url=self.lora_url,
            lora_bytes=self.lora_bytes,
            seed=7,
        ).run(ctx_b)
        # The LoRA bytes are constant ('L'*N) and inferences are seed-keyed,
        # so two runs with the same seed produce identical frame streams.
        self.assertEqual(fr_a, fr_b)

    def test_obeys_stop_event(self) -> None:
        ctx, frames, tasks = _make_recording_ctx()
        ctx.stop_event.set()
        wl = LoraLoadingWorkload(
            prompts=["a", "b"],
            lora_url=self.lora_url,
            lora_bytes=self.lora_bytes,
            seed=7,
        )
        wl.run(ctx)
        # Stopped before the download even starts.
        self.assertEqual(frames, [])
        self.assertEqual(tasks, [])


if __name__ == "__main__":
    unittest.main()
