"""Pull all available sc341 metrics from wandb and render scaling plots.

Combines:
  - Train loss   from gust2-nsp-robust-scaling-bridges (one number per network)
  - EMD/nsp + TKE-RSE/nsp  from gust2-analysis-bridges-scaling
    (one number per (network, kind); kind is rollout (2000-step AR) or
    eval (single-step). Only vs-raw-pixel-GT metrics are plotted.)

Output: one figure per VQ size, each with the same 3-row layout:
  row 0: train CE (one panel)
  row 1: EMD vs raw pixel GT — single-step | rollout (per-panel y-scale)
  row 2: TKE RSE vs raw pixel GT — single-step | rollout (per-panel y-scale)

  plots/sc341_local/all_metrics_small.png
  plots/sc341_local/all_metrics_medium.png

The s09 (1-layer trunk) network is excluded as an aspect-ratio outlier (d/L=384;
see scaling-plot diagnostic note in the writeup).

Usage:
    ~/llm/bin/python plot_sc341_all.py
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import wandb

ENTITY = "bigpseud-ucsc"
TRAIN_PROJECT = "gust2-nsp-robust-scaling-bridges"
ANALYSIS_PROJECT = "gust2-analysis-bridges-scaling"

LABEL_ORDER = ["s06", "s13", "s18", "s24"]   # s09 excluded (1-layer trunk outlier)
SIZE_COLORS = {"small": "#0072B2", "medium": "#D55E00"}     # Wong blue + vermillion
SIZE_MARKERS = {"small": "o", "medium": "s"}
SIZE_TITLES = {"small": "VQ small (D=5)", "medium": "VQ medium (D=10)"}
KIND_TITLES = {"eval": "single-step (2000 pairs)",
               "rollout": "AR rollout (2000 steps)"}


def label_to_params_M(label):
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def parse_train_name(name):
    parts = name.split("-")
    if len(parts) < 4 or parts[2] != "nsp": return None
    return parts[0], parts[1], parts[3]   # size, sc, label


def parse_analysis_name(name):
    parts = name.split("-")
    if len(parts) < 4 or parts[2] != "nsp": return None
    size, sc, label = parts[0], parts[1], parts[3]
    kind = "eval" if (len(parts) >= 5 and parts[4] == "eval") else "rollout"
    return size, sc, label, kind


def fetch_all():
    api = wandb.Api()
    # Indexed by (size, sc, label, kind) for analysis; (size, sc, label, "train") for training.
    table = defaultdict(dict)

    # Training runs.
    print(f"Fetching {TRAIN_PROJECT}...")
    for r in api.runs(f"{ENTITY}/{TRAIN_PROJECT}"):
        p = parse_train_name(r.name)
        if p is None: continue
        size, sc, label = p
        if sc != "sc341" or size not in ("small", "medium"): continue
        if r.state != "finished": continue
        params_M = label_to_params_M(label)
        if params_M is None: continue
        table[(size, sc, label, "train")] = {
            "name":     r.name,
            "size":     size,
            "sc":       sc,
            "label":    label,
            "params_M": params_M,
            "train_loss": r.summary.get("loss"),
        }

    # Analysis runs.
    try:
        analysis_runs = list(api.runs(f"{ENTITY}/{ANALYSIS_PROJECT}"))
    except ValueError:
        analysis_runs = []
        print(f"({ANALYSIS_PROJECT} not found yet — analysis panels will be empty)")
    print(f"Fetching {ANALYSIS_PROJECT}... ({len(analysis_runs)} runs)")
    for r in analysis_runs:
        p = parse_analysis_name(r.name)
        if p is None: continue
        size, sc, label, kind = p
        if sc != "sc341" or size not in ("small", "medium"): continue
        if r.state != "finished": continue
        params_M = label_to_params_M(label)
        if params_M is None: continue
        table[(size, sc, label, kind)] = {
            "name":     r.name,
            "size":     size,
            "sc":       sc,
            "label":    label,
            "kind":     kind,
            "params_M": params_M,
            "emd/nsp":       r.summary.get("emd/nsp"),
            "tke_rse/nsp":   r.summary.get("tke_rse/nsp"),
        }

    return table


def plot_train_loss(ax, table, size):
    pts = [table[(size, "sc341", l, "train")]
           for l in LABEL_ORDER if (size, "sc341", l, "train") in table]
    if not pts: return
    pts.sort(key=lambda r: r["params_M"])
    x = [r["params_M"] for r in pts]
    y = [r["train_loss"] for r in pts]
    ax.plot(x, y, marker=SIZE_MARKERS[size], color=SIZE_COLORS[size],
            linewidth=1.8, markersize=9, alpha=0.9)


def plot_metric_kind(ax, table, key, kind, size):
    pts = [table[(size, "sc341", l, kind)]
           for l in LABEL_ORDER
           if (size, "sc341", l, kind) in table
           and table[(size, "sc341", l, kind)].get(key) is not None]
    if not pts: return
    pts.sort(key=lambda r: r["params_M"])
    x = [r["params_M"] for r in pts]
    y = [r[key] for r in pts]
    ax.plot(x, y, marker=SIZE_MARKERS[size], color=SIZE_COLORS[size],
            linewidth=1.6, markersize=8, alpha=0.9)


def render_size(table, size, output_path):
    """Render one 3-row figure for the given VQ size (sc341 only)."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.subplots_adjust(hspace=0.38, wspace=0.22)

    # ─ Row 0: train CE in left cell, info card on right ─
    ax_ce = axes[0, 0]
    plot_train_loss(ax_ce, table, size)
    ax_ce.set_xlabel("NSP params (M)", fontsize=11)
    ax_ce.set_ylabel("Train CE (epoch 400)", fontsize=11)
    ax_ce.set_title("Train CE per token", fontsize=12, fontweight="bold")
    ax_ce.set_xscale("log")
    ax_ce.grid(True, alpha=0.3)

    ax_info = axes[0, 1]
    ax_info.axis("off")
    info_text = (f"VQ size: {size}\n"
                 f"sc-config: sc341\n"
                 f"NSP archs: {', '.join(LABEL_ORDER)}\n"
                 f"(s09 excluded — 1-layer\n"
                 f" trunk aspect-ratio outlier)")
    ax_info.text(0.5, 0.5, info_text, ha="center", va="center",
                 fontsize=12, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6",
                           facecolor=SIZE_COLORS[size], alpha=0.10,
                           edgecolor=SIZE_COLORS[size]))

    # ─ Row 1: EMD vs raw pixel GT (per-panel y-scale) ─
    for col, kind in enumerate(("eval", "rollout")):
        ax = axes[1, col]
        plot_metric_kind(ax, table, "emd/nsp", kind, size)
        ax.set_xlabel("NSP params (M)", fontsize=11)
        ax.set_ylabel("Pixel EMD vs raw GT", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Pixel EMD — {KIND_TITLES[kind]}",
                     fontsize=12, fontweight="bold")

    # ─ Row 2: TKE RSE vs raw pixel GT (per-panel y-scale) ─
    for col, kind in enumerate(("eval", "rollout")):
        ax = axes[2, col]
        plot_metric_kind(ax, table, "tke_rse/nsp", kind, size)
        ax.set_xlabel("NSP params (M)", fontsize=11)
        ax.set_ylabel("TKE RSE vs raw GT", fontsize=11)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"TKE RSE — {KIND_TITLES[kind]}",
                     fontsize=12, fontweight="bold")

    fig.suptitle(f"sc341 / {SIZE_TITLES[size]} — CE, EMD, TKE-RSE vs raw pixel GT",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/sc341_local")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    table = fetch_all()

    print("\nCoverage (for plotted networks):")
    for size in ("small", "medium"):
        avail = []
        for label in LABEL_ORDER:
            kinds = [k for k in ("train", "rollout", "eval")
                     if (size, "sc341", label, k) in table]
            avail.append(f"{label}=[{','.join(kinds) if kinds else '--'}]")
        print(f"  {size}-sc341:  " + "  ".join(avail))

    print()
    for size in ("small", "medium"):
        out = os.path.join(args.output_dir, f"all_metrics_{size}.png")
        render_size(table, size, out)


if __name__ == "__main__":
    main()
