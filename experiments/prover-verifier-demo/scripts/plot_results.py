#!/usr/bin/env python3
"""Plot detection rate vs knob value for the eval sweep (Task 9.2).

Reads `data/eval/results.jsonl`, computes per-knob detection rates for
the two adversarial workloads, and writes:
  - figures/mixed_lora_detection.png
  - figures/lora_loading_detection.png

We deliberately keep this small. Plots open no windows
(`matplotlib.use("Agg")`); detection rate at a single knob value is just
1.0 if the verdict is `training_or_exfil` else 0.0, so curves can have
duplicate-knob aggregation when the harness sweeps repeat seeds.

Usage:
    python3 experiments/prover-verifier-demo/scripts/plot_results.py \\
        --results data/eval/results.jsonl \\
        --figures-dir experiments/prover-verifier-demo/figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)


def _load_rows(results_path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line
    ]


def _detection_curve(rows: list[dict[str, object]], workload: str) -> tuple[list[int], list[float]]:
    """Average `verdict == training_or_exfil` per knob value, sorted by knob."""
    buckets: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r.get("workload") != workload:
            continue
        knob = r.get("knob_value")
        verdict = r.get("verdict")
        if not isinstance(knob, int):
            continue
        buckets[knob].append(1 if verdict == "training_or_exfil" else 0)
    if not buckets:
        return [], []
    keys = sorted(buckets)
    rates = [sum(buckets[k]) / len(buckets[k]) for k in keys]
    return keys, rates


def _plot_curve(
    *,
    knob_values: list[int],
    detection_rates: list[float],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(knob_values, detection_rates, marker="o", color="#c0392b")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("detection rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot eval sweep detection curves")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figures-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = _load_rows(args.results)

    g_knobs, g_rates = _detection_curve(rows, "mixed_lora")
    if g_knobs:
        _plot_curve(
            knob_values=g_knobs,
            detection_rates=g_rates,
            title="mixed_lora — detection vs gradient_steps",
            xlabel="gradient_steps",
            out_path=args.figures_dir / "mixed_lora_detection.png",
        )

    b_knobs, b_rates = _detection_curve(rows, "lora_loading")
    if b_knobs:
        _plot_curve(
            knob_values=b_knobs,
            detection_rates=b_rates,
            title="lora_loading — detection vs lora_bytes",
            xlabel="lora_bytes",
            out_path=args.figures_dir / "lora_loading_detection.png",
        )

    print(
        f"[plot] wrote figures to {args.figures_dir} "
        f"(mixed_lora={len(g_knobs)} pts, lora_loading={len(b_knobs)} pts)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
