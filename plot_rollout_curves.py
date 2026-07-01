"""Compare sc341 and sc917 rollout curves at fixed VQ-VAE size.

One figure per VQ size (small / medium); each figure has three panels:
  - Train CE per token (epoch 400)         — context only
  - Rollout pixel EMD vs raw pixel GT
  - Rollout TKE RSE vs raw pixel GT

Each panel overlays the sc341 and sc917 curves; the x-axis is NSP params
(M, derived from the label suffix). Single-step metrics omitted by user
request.

s09 (sc341, 1-layer trunk) excluded as an aspect-ratio outlier; sc917's
labels (s13, s22, s34, s50, s74) all have L >= 3 so no exclusion there.

Output:
  plots/sc341_local/rollout_curves_small.png
  plots/sc341_local/rollout_curves_medium.png

Usage:
    ~/llm/bin/python plot_rollout_curves.py
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
# Picked for severe-colorblindness safety: very different luminances
# (dark navy ~70 vs bright amber ~190) so they remain distinguishable
# even under monochromacy. Different hue families too (cool vs warm)
# so they read as separate even with full red-green deficiency.
SC_COLORS = {"sc341": "#1F4E79", "sc917": "#FFC107"}
SC_MARKERS = {"sc341": "o", "sc917": "s"}
SC_LABELS = {"sc341": "sc341 (341 tok/frame)", "sc917": "sc917 (917 tok/frame)"}
VQVAE_TITLES = {
    "small":  "VQ small (D=5)",
    "medium": "VQ medium (D=10)",
    "large":  "VQ large (D=20)",
}
SIZES = ("small", "medium", "large")


def label_to_params_M(label):
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def fetch():
    """table[(size, sc, label, kind)] -> {name, params_M, train_loss?, emd/nsp?, tke_rse/nsp?}"""
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
            "name": r.name, "params_M": params_M,
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
            "name": r.name, "params_M": params_M,
            "emd/nsp":       r.summary.get("emd/nsp"),
            "tke_rse/nsp":   r.summary.get("tke_rse/nsp"),
            "emd/vqvae":     r.summary.get("emd/vqvae"),
            "tke_rse/vqvae": r.summary.get("tke_rse/vqvae"),
        }
    return table


def vqvae_floor(table, size, sc, kind, key):
    """Median VQ-VAE-floor value across all rollout runs in (size, sc).

    The vqvae floor is the same field for every NSP arch (same VQ-VAE +
    same GT window), so we just take the median for robustness against
    a single bad run.
    """
    vals = []
    for label in LABELS[sc]:
        rec = table.get((size, sc, label, kind))
        if rec is None:
            continue
        v = rec.get(key)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def gather(table, size, sc, kind, key=None):
    pts = []
    for label in LABELS[sc]:
        rec = table.get((size, sc, label, kind))
        if rec is None:
            continue
        if key is not None and rec.get(key) is None:
            continue
        pts.append(rec)
    pts.sort(key=lambda r: r["params_M"])
    return pts


def plot_panel(ax, table, size, kind, key, ylabel, title, floor_key=None):
    drawn_any = False
    for sc in ("sc341", "sc917"):
        pts = gather(table, size, sc, kind, key)
        if pts:
            x = [r["params_M"] for r in pts]
            y = [r[key] if key else r["train_loss"] for r in pts]
            ax.plot(x, y, marker=SC_MARKERS[sc], color=SC_COLORS[sc],
                    linewidth=1.8, markersize=8, alpha=0.92,
                    label=SC_LABELS[sc])
            drawn_any = True

        # VQ-VAE floor: same field-of-data → same vqvae value for every
        # NSP arch in the (size, sc) group. Draw it as a dashed reference.
        if floor_key is not None:
            floor = vqvae_floor(table, size, sc, kind, floor_key)
            if floor is not None:
                ax.axhline(floor, color=SC_COLORS[sc], linestyle="--",
                           linewidth=1.2, alpha=0.55,
                           label=f"{sc} VQ-VAE floor")

    if not drawn_any:
        ax.text(0.5, 0.5, "(no data yet)", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
    ax.set_xlabel("NSP params (M)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)


def render_size(table, size, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    plot_panel(axes[0], table, size, "train", None,
               ylabel="Train CE (epoch 400)",
               title="Train CE per token")
    plot_panel(axes[1], table, size, "rollout", "emd/nsp",
               ylabel="Pixel EMD vs raw GT",
               title="Rollout pixel EMD",
               floor_key="emd/vqvae")
    plot_panel(axes[2], table, size, "rollout", "tke_rse/nsp",
               ylabel="TKE RSE vs raw GT",
               title="Rollout TKE RSE",
               floor_key="tke_rse/vqvae")

    handles = [Line2D([0], [0], color=SC_COLORS[sc], marker=SC_MARKERS[sc],
                      linewidth=2.2, markersize=9, label=SC_LABELS[sc])
               for sc in ("sc341", "sc917")]
    handles.append(Line2D([0], [0], color="0.4", linestyle="--",
                          linewidth=1.6, label="VQ-VAE floor (per sc)"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), fontsize=11, frameon=False)

    fig.suptitle(f"{VQVAE_TITLES[size]} — sc341 vs sc917, rollout metrics",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


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

    print()
    for size in SIZES:
        out = os.path.join(args.output_dir, f"rollout_curves_{size}.png")
        render_size(table, size, out)


if __name__ == "__main__":
    main()
