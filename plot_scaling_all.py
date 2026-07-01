"""All-in-one scaling-curves figure: every (VQ-size, sc-config) curve.

3×3 grid:
  rows    = VQ size       (small / medium / large)
  columns = metric        (Train CE / Rollout pixel-EMD / Rollout TKE-RSE)

Each panel overlays the sc341 and sc917 curves. VQ-VAE floors drawn as
dashed reference lines (color-matched to the sc-config). Shared legend
at the bottom of the figure.

Output:
  plots/sc341_local/scaling_all.png

Usage:
    ~/llm/bin/python plot_scaling_all.py
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb

ENTITY = "bigpseud-ucsc"
TRAIN_PROJECT = "gust2-nsp-robust-scaling-bridges"
ANALYSIS_PROJECT = "gust2-analysis-bridges-scaling"

LABELS = {
    "sc341": ["s06", "s09", "s13", "s18", "s24"],
    "sc917": ["s13", "s22", "s34", "s50", "s74"],
}
SC_COLORS = {"sc341": "#1F4E79", "sc917": "#FFC107"}
SC_MARKERS = {"sc341": "o", "sc917": "s"}
SC_LABELS = {"sc341": "sc341 (341 tok/frame)", "sc917": "sc917 (917 tok/frame)"}
SIZES = ("small", "medium", "large")
SIZE_TITLES = {
    "small":  "VQ small (D=5)",
    "medium": "VQ medium (D=10)",
    "large":  "VQ large (D=20)",
}


def label_to_params_M(label):
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def fetch():
    api = wandb.Api()
    table = defaultdict(dict)

    print(f"Fetching {TRAIN_PROJECT}...")
    for r in api.runs(f"{ENTITY}/{TRAIN_PROJECT}"):
        parts = r.name.split("-")
        if len(parts) < 4 or parts[2] != "nsp": continue
        size, sc, label = parts[0], parts[1], parts[3]
        if r.state != "finished": continue
        if size not in SIZES: continue
        if sc not in ("sc341", "sc917"): continue
        params_M = label_to_params_M(label)
        if params_M is None: continue
        table[(size, sc, label, "train")] = {
            "params_M": params_M,
            "train_loss": r.summary.get("loss"),
        }

    try:
        analysis_runs = list(api.runs(f"{ENTITY}/{ANALYSIS_PROJECT}"))
    except ValueError:
        analysis_runs = []
    print(f"Fetching {ANALYSIS_PROJECT}... ({len(analysis_runs)} runs)")
    for r in analysis_runs:
        parts = r.name.split("-")
        if len(parts) < 4 or parts[2] != "nsp": continue
        size, sc, label = parts[0], parts[1], parts[3]
        kind = "eval" if (len(parts) >= 5 and parts[4] == "eval") else "rollout"
        if r.state != "finished": continue
        if size not in SIZES: continue
        if sc not in ("sc341", "sc917"): continue
        params_M = label_to_params_M(label)
        if params_M is None: continue
        table[(size, sc, label, kind)] = {
            "params_M": params_M,
            "emd/nsp":       r.summary.get("emd/nsp"),
            "tke_rse/nsp":   r.summary.get("tke_rse/nsp"),
            "emd/vqvae":     r.summary.get("emd/vqvae"),
            "tke_rse/vqvae": r.summary.get("tke_rse/vqvae"),
        }
    return table


def vqvae_floor(table, size, sc, kind, key):
    vals = []
    for label in LABELS[sc]:
        rec = table.get((size, sc, label, kind))
        if rec is None: continue
        v = rec.get(key)
        if v is not None: vals.append(v)
    if not vals: return None
    vals.sort()
    return vals[len(vals) // 2]


def gather(table, size, sc, kind, key=None):
    pts = []
    for label in LABELS[sc]:
        rec = table.get((size, sc, label, kind))
        if rec is None: continue
        if key is not None and rec.get(key) is None: continue
        pts.append(rec)
    pts.sort(key=lambda r: r["params_M"])
    return pts


def plot_panel(ax, table, size, kind, key, floor_key=None):
    for sc in ("sc341", "sc917"):
        pts = gather(table, size, sc, kind, key)
        if pts:
            x = [r["params_M"] for r in pts]
            y = [r[key] if key else r["train_loss"] for r in pts]
            ax.plot(x, y, marker=SC_MARKERS[sc], color=SC_COLORS[sc],
                    linewidth=1.8, markersize=7, alpha=0.92)
        if floor_key is not None:
            floor = vqvae_floor(table, size, sc, kind, floor_key)
            if floor is not None:
                ax.axhline(floor, color=SC_COLORS[sc], linestyle="--",
                           linewidth=1.1, alpha=0.55)
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/sc341_local")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    table = fetch()

    print("\nCoverage (train | rollout):")
    for size in SIZES:
        for sc in ("sc341", "sc917"):
            t = sum(1 for l in LABELS[sc] if (size, sc, l, "train") in table)
            r = sum(1 for l in LABELS[sc] if (size, sc, l, "rollout") in table)
            print(f"  {size}-{sc}: train {t}/{len(LABELS[sc])}  rollout {r}/{len(LABELS[sc])}")

    fig, axes = plt.subplots(len(SIZES), 3, figsize=(15, 12), sharex=False)

    col_titles = ["Train CE per token", "Rollout pixel EMD", "Rollout TKE RSE"]
    col_ylabels = ["Train CE (epoch 400)", "Pixel EMD vs raw GT", "TKE RSE vs raw GT"]

    for i, size in enumerate(SIZES):
        plot_panel(axes[i, 0], table, size, "train",   None)
        plot_panel(axes[i, 1], table, size, "rollout", "emd/nsp",
                   floor_key="emd/vqvae")
        plot_panel(axes[i, 2], table, size, "rollout", "tke_rse/nsp",
                   floor_key="tke_rse/vqvae")

        # Row label on the left edge.
        axes[i, 0].set_ylabel(f"{SIZE_TITLES[size]}\n\n{col_ylabels[0]}",
                              fontsize=11)
        axes[i, 1].set_ylabel(col_ylabels[1], fontsize=10)
        axes[i, 2].set_ylabel(col_ylabels[2], fontsize=10)

        # Column titles only on the top row.
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(t, fontsize=12, fontweight="bold")

        # X labels only on the bottom row.
        if i == len(SIZES) - 1:
            for j in range(3):
                axes[i, j].set_xlabel("NSP params (M)", fontsize=10)

    handles = [Line2D([0], [0], color=SC_COLORS[sc], marker=SC_MARKERS[sc],
                      linewidth=2.2, markersize=9, label=SC_LABELS[sc])
               for sc in ("sc341", "sc917")]
    handles.append(Line2D([0], [0], color="0.4", linestyle="--",
                          linewidth=1.6, label="VQ-VAE floor (per sc)"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=False)

    fig.suptitle("Scaling curves — every (VQ size × sc-config) combination",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    out = os.path.join(args.output_dir, "scaling_all.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    main()
