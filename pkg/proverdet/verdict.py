"""Verdict engine for the verifier.

Three signals (replay correctness, compute budget, bandwidth) feed a
combiner that emits the final verdict. Phase 8.1 lands `replay_correctness`
+ the `SignalResult` shape; 8.2 adds `compute_budget`; 8.3 adds bandwidth
and the final combiner.

Where signals get their data:
  * `replay_correctness` reads `transcript_entries` (the standard
    transcript.jsonl) — only needs status_code per /replay/verdict/{id}.
  * `compute_budget` reads `summaries` (a sidecar `summaries.jsonl` the
    scheduler writes alongside the transcript). The transcript only stores
    payload digests, not raw bodies, so per-entry `claimed_flops` /
    `observed_flops` accounting can't come from it.

`emit_verdict` wires both. Phase 8.3 will add `bandwidth` and the
"any-failed-signal-fires-training_or_exfil" combiner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SignalResult:
    passed: bool
    reasons: list[str]


# ---- replay_correctness (Task 8.1) -----------------------------------


def replay_correctness(transcript_entries: list[dict[str, object]]) -> SignalResult:
    """`passed` iff every recorded /replay/verdict/{id} entry is status 200.

    Failed entries surface a reason naming the replay_id (extracted from
    the endpoint suffix: `/replay/verdict/<replay_id>` is the convention
    the scheduler writes).
    """
    failures: list[str] = []
    saw_any = False
    for e in transcript_entries:
        endpoint = e.get("endpoint")
        if not isinstance(endpoint, str):
            continue
        if e.get("direction") != "received":
            continue
        if not endpoint.startswith("/replay/verdict/"):
            continue
        saw_any = True
        if e.get("status_code") == 200:
            continue
        replay_id = endpoint[len("/replay/verdict/") :]
        failures.append(f"replay {replay_id} failed: status_code={e.get('status_code')}")
    if not saw_any:
        return SignalResult(passed=True, reasons=[])
    if failures:
        return SignalResult(passed=False, reasons=failures)
    return SignalResult(passed=True, reasons=[])


# ---- compute_budget (Task 8.2) ---------------------------------------


def compute_budget(summaries: list[dict[str, object]], *, tolerance: float = 0.10) -> SignalResult:
    """`passed` iff observed_flops <= (1 + tolerance) * claimed_flops.

    Scheduler writes one summary per /graph (claimed_flops_total) and one
    per /replay evidence (observed_flops). We sum across the run: a single
    breach (sum of observed exceeds the claim envelope) is enough to fail —
    that's what mixed_lora's gradient-step cheating produces.
    """
    claimed_total = 0
    observed_total = 0
    for s in summaries:
        kind = s.get("kind")
        if kind == "graph":
            v = s.get("claimed_flops_total", 0)
            if isinstance(v, int):
                claimed_total += v
        elif kind == "replay_evidence":
            v = s.get("observed_flops", 0)
            if isinstance(v, int):
                observed_total += v

    if claimed_total == 0 and observed_total == 0:
        return SignalResult(passed=True, reasons=[])

    threshold = (1.0 + tolerance) * claimed_total
    if observed_total > threshold:
        return SignalResult(
            passed=False,
            reasons=[
                f"compute budget exceeded: observed_flops={observed_total} "
                f"vs claimed_flops={claimed_total} (tolerance={tolerance:.2f})"
            ],
        )
    return SignalResult(passed=True, reasons=[])


# ---- file helpers + emit_verdict -------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def emit_verdict(transcript_path: Path) -> dict[str, object]:
    """Read the transcript + sidecars, emit a verdict.

    Until Phase 8.3 the verdict is "unknown" on no-signal-fires and
    "training_or_exfil" if any signal fires. 8.3 will add the bandwidth
    signal and finalize the combiner (and extend the signature with
    `traffic_digest_path`).
    """
    if not Path(transcript_path).exists():
        raise FileNotFoundError(f"transcript not found: {transcript_path}")
    transcript_entries = _read_jsonl(Path(transcript_path))
    summaries = _read_jsonl(Path(transcript_path).parent / "summaries.jsonl")

    correctness = replay_correctness(transcript_entries)
    budget = compute_budget(summaries)

    reasons: list[str] = []
    if not correctness.passed:
        reasons.extend(correctness.reasons)
    if not budget.passed:
        reasons.extend(budget.reasons)

    if reasons:
        return {"verdict": "training_or_exfil", "reasons": reasons}
    return {"verdict": "unknown", "reasons": []}
