#!/usr/bin/env python3
"""Generate overhead figures for the paper from overhead.jsonl data."""

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "overhead.jsonl"
OUT = ROOT / "results" / "figures"
OUT.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────
rows = [json.loads(l) for l in DATA.read_text().splitlines()]

MODELS = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen 2.5 1.5B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B",
}
CONFIGS = ["c0", "c1", "c2", "c3"]
CONFIG_LABELS = {
    "c0": "Baseline",
    "c1": "+ Eager mode",
    "c2": "+ Deterministic cuBLAS",
    "c3": "+ Batch invariance",
}
BATCH_SIZES = [1, 4, 16, 64, 128]
SEQ_LENS = [16, 128, 512, 2048]

# Build lookup: (model, config, batch, seq) -> tok_per_s
lookup = {}
for r in rows:
    key = (r["model"], r["config"], r["batch_size"], r["max_tokens"])
    lookup[key] = r["tok_per_s"]


# ── Figure 1: Throughput by batch size (one subplot per model, fixed seq=128) ─
def plot_throughput_by_batch(seq_len=128):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = ["#2196F3", "#FF9800", "#F44336", "#4CAF50"]

    for ax, (model_id, model_name) in zip(axes, MODELS.items()):
        for ci, cfg in enumerate(CONFIGS):
            throughputs = [lookup.get((model_id, cfg, bs, seq_len), 0)
                          for bs in BATCH_SIZES]
            ax.plot(range(len(BATCH_SIZES)), throughputs,
                    "o-", color=colors[ci], label=CONFIG_LABELS[cfg],
                    linewidth=2, markersize=6)
        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput (tokens/s)")
        ax.set_title(model_name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    fig.suptitle(f"Throughput vs. Determinism Level (seq_len={seq_len})", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "throughput_by_batch.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'throughput_by_batch.png'}")
    plt.close()


# ── Figure 2: Overhead % (c0→c3) by batch size, one line per seq len ─────────
def plot_overhead_pct():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["#9C27B0", "#E91E63", "#FF5722", "#795548"]

    for ax, (model_id, model_name) in zip(axes, MODELS.items()):
        for si, sl in enumerate(SEQ_LENS):
            overheads = []
            for bs in BATCH_SIZES:
                c0 = lookup.get((model_id, "c0", bs, sl), 1)
                c3 = lookup.get((model_id, "c3", bs, sl), 0)
                overheads.append((c3 / c0 - 1) * 100)
            ax.plot(range(len(BATCH_SIZES)), overheads,
                    "s-", color=colors[si], label=f"seq={sl}",
                    linewidth=2, markersize=6)

        ax.set_xticks(range(len(BATCH_SIZES)))
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput change (%)")
        ax.set_title(model_name)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Full Determinism Overhead (c0 → c3)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "overhead_pct.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'overhead_pct.png'}")
    plt.close()


# ── Figure 3: Stacked incremental cost (averaged across seq lens) ────────────
def plot_incremental_cost():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    flag_labels = [
        "Eager mode\n(disable CUDAGraphs)",
        "Deterministic\ncuBLAS",
        "Batch\ninvariance",
    ]

    for ax, (model_id, model_name) in zip(axes, MODELS.items()):
        # For each batch size, compute avg incremental cost across seq lens
        eager_costs = []
        cublas_costs = []
        batch_costs = []

        for bs in BATCH_SIZES:
            e_vals, cu_vals, bi_vals = [], [], []
            for sl in SEQ_LENS:
                c0 = lookup.get((model_id, "c0", bs, sl), 1)
                c1 = lookup.get((model_id, "c1", bs, sl), 0)
                c2 = lookup.get((model_id, "c2", bs, sl), 0)
                c3 = lookup.get((model_id, "c3", bs, sl), 0)
                e_vals.append((c0 - c1) / c0 * 100)
                cu_vals.append((c1 - c2) / c0 * 100)
                bi_vals.append((c2 - c3) / c0 * 100)
            eager_costs.append(np.mean(e_vals))
            cublas_costs.append(np.mean(cu_vals))
            batch_costs.append(np.mean(bi_vals))

        x = np.arange(len(BATCH_SIZES))
        width = 0.6
        ax.bar(x, eager_costs, width, color=colors[0], label=flag_labels[0])
        ax.bar(x, cublas_costs, width, bottom=eager_costs, color=colors[1],
               label=flag_labels[1])
        bottoms = [e + c for e, c in zip(eager_costs, cublas_costs)]
        ax.bar(x, batch_costs, width, bottom=bottoms, color=colors[2],
               label=flag_labels[2])

        ax.set_xticks(x)
        ax.set_xticklabels(BATCH_SIZES)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput lost (% of baseline)")
        ax.set_title(model_name)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Incremental Cost of Each Determinism Flag", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "incremental_cost.png", dpi=200, bbox_inches="tight")
    print(f"Saved {OUT / 'incremental_cost.png'}")
    plt.close()


if __name__ == "__main__":
    plot_throughput_by_batch()
    plot_overhead_pct()
    plot_incremental_cost()
    print("Done.")
