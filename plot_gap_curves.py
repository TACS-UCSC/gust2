"""Normalized-gap (rollout / VQ-VAE-floor) curves for sc341 vs sc917.

Companion to plot_rollout_curves.py. The absolute EMD and TKE-RSE values
encode VQ-VAE reconstruction quality (sc917's floor is lower than sc341's)
in addition to AR error. To isolate the AR error, this script plots the
multiplicative excess over the VQ-VAE floor:

    gap = rollout_metric / vqvae_floor_metric

Reading:
  gap = 1.0  → AR rollout is indistinguishable from the VQ-VAE round-trip
              (model has consumed all available headroom)
  gap = 2.0  → AR rollout is 2× worse than the VQ-VAE floor
  gap > 5    → near-catastrophic divergence

Y-axis is log-scaled so catastrophic-collapse points (sc917 small-NSP)
don't crush the rest of the curve.

Output:
  plots/sc341_local/gap_curves_small.png
  plots/sc341_local/gap_curves_medium.png

Usage:
    ~/llm/bin/python plot_gap_curves.py
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import wandb

ENTITY = "bigpseud-ucsc"
ANALYSIS_PROJECT = "gust2-analysis-bridges-scaling"

LABELS = {
    "sc341": ["s06", "s09", "s13", "s18", "s24"],
    "sc917": ["s13", "s22", "s34", "s50", "s74"],
}
SC_COLORS = {"sc341": "#1F4E79", "sc917": "#FFC107"}     # CVD-safe (matches plot_rollout_curves.py)
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
    api = wandb.Api()
    table = defaultdict(dict)
    runs = list(api.runs(f"{ENTITY}/{ANALYSIS_PROJECT}"))
    print(f"Fetching {ANALYSIS_PROJECT}... ({len(runs)} runs)")
    for r in runs:
        parts = r.name.split("-")
        if len(parts) < 4 or parts[2] != "nsp": continue
        size, sc, label = parts[0], parts[1], parts[3]
        kind = "eval" if (len(parts) >= 5 and parts[4] == "eval") else "rollout"
        if kind != "rollout": continue
        if r.state != "finished": continue
        if size not in SIZES: continue
        if sc not in ("sc341", "sc917"): continue
        params_M = label_to_params_M(label)
        if params_M is None: continue
        table[(size, sc, label)] = {
            "params_M": params_M,
            "emd/nsp":       r.summary.get("emd/nsp"),
            "tke_rse/nsp":   r.summary.get("tke_rse/nsp"),
            "emd/vqvae":     r.summary.get("emd/vqvae"),
            "tke_rse/vqvae": r.summary.get("tke_rse/vqvae"),
        }
    return table


def vqvae_floor(table, size, sc, key):
    """Median floor across NSP archs in (size, sc) — same field, same VQ-VAE."""
    vals = [table[(size, sc, l)][key] for l in LABELS[sc]
            if (size, sc, l) in table and table[(size, sc, l)].get(key) is not None]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def gather_gap(table, size, sc, num_key, denom_key):
    floor = vqvae_floor(table, size, sc, denom_key)
    if floor is None or floor <= 0:
        return [], []
    pts = []
    for label in LABELS[sc]:
        rec = table.get((size, sc, label))
        if rec is None: continue
        v = rec.get(num_key)
        if v is None: continue
        pts.append((rec["params_M"], v / floor))
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def plot_gap_panel(ax, table, size, num_key, denom_key, ylabel, title):
    drawn_any = False
    for sc in ("sc341", "sc917"):
        x, y = gather_gap(table, size, sc, num_key, denom_key)
        if x:
            ax.plot(x, y, marker=SC_MARKERS[sc], color=SC_COLORS[sc],
                    linewidth=1.8, markersize=8, alpha=0.92,
                    label=SC_LABELS[sc])
            drawn_any = True

    # gap = 1.0 means at the VQ-VAE floor (best possible).
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2, alpha=0.7,
               label="floor (gap=1.0)")

    if not drawn_any:
        ax.text(0.5, 0.5, "(no data yet)", ha="center", va="center",
                transform=ax.transAxes, color="0.5")
    ax.set_xlabel("NSP params (M)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)


def render_size(table, size, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    plot_gap_panel(axes[0], table, size, "emd/nsp", "emd/vqvae",
                   ylabel="rollout EMD / VQ-VAE-floor EMD",
                   title="Pixel-EMD multiplicative gap")
    plot_gap_panel(axes[1], table, size, "tke_rse/nsp", "tke_rse/vqvae",
                   ylabel="rollout TKE-RSE / VQ-VAE-floor TKE-RSE",
                   title="TKE-RSE multiplicative gap")

    handles = [Line2D([0], [0], color=SC_COLORS[sc], marker=SC_MARKERS[sc],
                      linewidth=2.2, markersize=9, label=SC_LABELS[sc])
               for sc in ("sc341", "sc917")]
    handles.append(Line2D([0], [0], color="0.35", linestyle="--",
                          linewidth=1.4, label="floor (gap=1.0)"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04), fontsize=11, frameon=False)

    fig.suptitle(f"{VQVAE_TITLES[size]} — gap from VQ-VAE floor (rollout)",
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

    print("\nFloors (median across NSP archs):")
    for size in SIZES:
        for sc in ("sc341", "sc917"):
            f_emd = vqvae_floor(table, size, sc, "emd/vqvae")
            f_tke = vqvae_floor(table, size, sc, "tke_rse/vqvae")
            print(f"  {size}-{sc}: emd/vqvae={f_emd}  tke_rse/vqvae={f_tke}")

    print()
    for size in SIZES:
        out = os.path.join(args.output_dir, f"gap_curves_{size}.png")
        render_size(table, size, out)


if __name__ == "__main__":
    main()
