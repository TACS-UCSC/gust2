"""Pull sc341 training curves + final losses from wandb and render plots.

Two figures, written to plots/sc341_local/:
  1. final_loss_vs_params.png     — scaling law (final train loss vs NSP params)
  2. train_curves.png             — per-epoch training loss for all 10 networks

Reads from gust2-nsp-robust-scaling-bridges, filters to small-sc341 +
medium-sc341 (10 finished runs at the time of writing). Run names follow
`<size>-sc341-nsp-<label>` and label `s<N>` encodes ~params in M.

Usage:
    ~/llm/bin/python plot_sc341_train.py
    ~/llm/bin/python plot_sc341_train.py --output_dir plots/sc341_local
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "bigpseud-ucsc"
PROJECT = "gust2-nsp-robust-scaling-bridges"

LABEL_ORDER = ["s06", "s09", "s13", "s18", "s24"]
SIZE_COLORS = {"small": "#0072B2", "medium": "#D55E00"}   # Wong blue + vermillion
SIZE_MARKERS = {"small": "o", "medium": "s"}

# Per-network color for training curves: gradient within each VQ size.
def label_color(size, label):
    base = plt.cm.Blues if size == "small" else plt.cm.Oranges
    idx = LABEL_ORDER.index(label) if label in LABEL_ORDER else 0
    return base(0.4 + 0.13 * idx)   # ramp 0.4 -> 0.92


def label_to_params_M(label):
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def fetch_runs():
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    rows = []
    for r in runs:
        parts = r.name.split("-")
        if len(parts) < 4 or parts[2] != "nsp":
            continue
        size, sc, label = parts[0], parts[1], parts[3]
        if size not in ("small", "medium") or sc != "sc341":
            continue
        if r.state != "finished":
            print(f"  [skip] {r.name}: state={r.state}")
            continue
        params_M = label_to_params_M(label)
        if params_M is None:
            continue
        # History: per-epoch loss. samples=2000 is plenty for 400 epochs.
        hist = r.history(keys=["epoch", "epoch/loss"], samples=2000)
        rows.append({
            "name":      r.name,
            "size":      size,
            "label":     label,
            "params_M":  params_M,
            "final_loss": r.summary.get("loss"),
            "epochs":    hist["epoch"].to_numpy(),
            "losses":    hist["epoch/loss"].to_numpy(),
        })
    rows.sort(key=lambda x: (x["size"], x["params_M"]))
    return rows


def plot_scaling_law(rows, output_path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for size in ("small", "medium"):
        subset = [r for r in rows if r["size"] == size]
        if not subset:
            continue
        x = np.array([r["params_M"] for r in subset])
        y = np.array([r["final_loss"] for r in subset])
        ax.plot(x, y, marker=SIZE_MARKERS[size], color=SIZE_COLORS[size],
                linewidth=1.8, markersize=9, alpha=0.9,
                label=f"VQ {size}")
        for r in subset:
            ax.annotate(r["label"], (r["params_M"], r["final_loss"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=8, color="0.4")

    ax.set_xlabel("NSP params (M, label-derived)", fontsize=11)
    ax.set_ylabel("Final train loss (epoch 400)", fontsize=11)
    ax.set_title("sc341 scaling law — train loss vs NSP params",
                 fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def plot_train_curves(rows, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, size in zip(axes, ("small", "medium")):
        subset = [r for r in rows if r["size"] == size]
        for r in subset:
            ax.plot(r["epochs"], r["losses"],
                    color=label_color(size, r["label"]),
                    linewidth=1.4, alpha=0.95,
                    label=f"{r['label']} ({r['params_M']:.0f}M)")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(f"VQ {size}-sc341 — training loss", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right", title="NSP arch")
    axes[0].set_ylabel("Train loss (epoch-mean)", fontsize=11)
    fig.suptitle("sc341 training curves", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/sc341_local")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Fetching runs from {ENTITY}/{PROJECT}...")
    rows = fetch_runs()
    print(f"\nUsing {len(rows)} finished runs:")
    for r in rows:
        n_pts = len(r["losses"])
        print(f"  {r['name']:<28}  {r['params_M']:>4.0f}M  "
              f"final={r['final_loss']:.3f}  history points={n_pts}")

    if not rows:
        print("No runs to plot.")
        return

    print("\n--- Scaling-law plot ---")
    plot_scaling_law(rows, os.path.join(args.output_dir, "final_loss_vs_params.png"))
    print("\n--- Training curves plot ---")
    plot_train_curves(rows, os.path.join(args.output_dir, "train_curves.png"))
    print(f"\nAll plots in {args.output_dir}/")


if __name__ == "__main__":
    main()
