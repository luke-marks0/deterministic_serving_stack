"""Verifier-side transcript log.

Schema: see schemas/verifier_transcript_entry.v1.schema.json.
Phase 3.2 will factor this and ProverCaptureLog onto a shared JSONL base
once both concrete shapes exist.
"""

from __future__ import annotations

import threading
from pathlib import Path

from pkg.common.contracts import validate_with_schema
from pkg.common.deterministic import (
    canonical_json_text,
    sha256_prefixed,
    utc_now_iso,
)


class TranscriptLog:
    """Append-only JSONL log for verifier-observed events."""

    SCHEMA_NAME = "verifier_transcript_entry.v1.schema.json"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._lock = threading.Lock()
        self._seq = 0

    def record(
        self,
        *,
        direction: str,
        endpoint: str,
        payload: bytes,
        status_code: int | None = None,
        payload_path: str | None = None,
    ) -> int:
        with self._lock:
            self._seq += 1
            seq = self._seq
            entry: dict[str, object] = {
                "seq": seq,
                "direction": direction,
                "endpoint": endpoint,
                "timestamp": utc_now_iso(),
                "payload_digest": sha256_prefixed(payload),
            }
            if status_code is not None:
                entry["status_code"] = status_code
            if payload_path is not None:
                entry["payload_path"] = payload_path

            # Validate against schema before write — programmer errors
            # surface early, not when the verdict engine reads the file.
            validate_with_schema(self.SCHEMA_NAME, entry)

            with self.path.open("a", encoding="utf-8") as f:
                f.write(canonical_json_text(entry))
            return seq

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
