"""LoRA-loading workload (adversarial — bandwidth knob).

The cheating: the workload "downloads" `lora_bytes` from a fake external
URL and republishes those bytes as outgoing frames the verifier sees on
the wire — but does NOT call `record_task` for them. From the verifier's
perspective, observed bytes-on-wire exceed the bytes the declared graph
accounts for.

This is the bandwidth analogue of the compute cheating in `mixed_lora`:
real network IO that doesn't appear in the graph. Phase 8.3's bandwidth
signal flags `(observed_bytes - sum(task.claimed_flops))` as the gap.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field

from workloads.context import WorkloadContext

_FRAMES_PER_PROMPT = 10
_FRAME_BYTES = 256


def _synth_frames_for_prompt(prompt: str, seed: int) -> list[bytes]:
    out: list[bytes] = []
    base = hashlib.sha256(f"lora_loading|{seed}|{prompt}".encode()).digest()
    for i in range(_FRAMES_PER_PROMPT):
        chunk = hashlib.sha256(base + i.to_bytes(8, "big")).digest()
        frame = bytearray()
        while len(frame) < _FRAME_BYTES:
            frame.extend(chunk)
        out.append(bytes(frame[:_FRAME_BYTES]))
    return out


def _chunk(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


@dataclass
class LoraLoadingWorkload:
    prompts: list[str]
    lora_url: str
    lora_bytes: int = 64 * 1024
    use_vllm: bool = False
    seed: int = 0
    pod_id: str = "pod-a"
    delay_per_prompt_s: float = 0.0
    download_timeout_s: float = 10.0
    # Mutated during run().
    download_bytes: int = field(default=0, init=False)

    def run(self, ctx: WorkloadContext) -> None:
        if self.use_vllm:
            return self._run_vllm(ctx)
        return self._run_synthetic(ctx)

    def _run_synthetic(self, ctx: WorkloadContext) -> None:
        # --- Cheating: download external bytes, publish on-wire, no task. ---
        if ctx.stop_event.is_set():
            return
        body = self._download(self.lora_url, self.lora_bytes)
        self.download_bytes = len(body)
        for chunk in _chunk(body, _FRAME_BYTES):
            if ctx.stop_event.is_set():
                return
            ctx.publish_frame(chunk)

        # --- Honest inference (mirrors BenignInferenceWorkload). ---
        for i, prompt in enumerate(self.prompts):
            if ctx.stop_event.is_set():
                return
            for f in _synth_frames_for_prompt(prompt, self.seed):
                ctx.publish_frame(f)
            ctx.record_task(
                {
                    "task_id": f"lora_loading-{self.seed}-{i:04d}",
                    "pod_id": self.pod_id,
                    "operation": "inference",
                    "claimed_flops": _FRAMES_PER_PROMPT * _FRAME_BYTES,
                }
            )
            if self.delay_per_prompt_s > 0 and ctx.stop_event.wait(self.delay_per_prompt_s):
                return

    def _download(self, url: str, expected_size: int) -> bytes:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self.download_timeout_s) as r:  # noqa: S310
            body = r.read(expected_size)
        return body

    def _run_vllm(self, ctx: WorkloadContext) -> None:  # pragma: no cover (GPU-only)
        import vllm  # noqa: F401

        raise NotImplementedError(
            "vLLM lora_loading path: implement when GPU is available; gated by tests."
        )
