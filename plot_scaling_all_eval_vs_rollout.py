"""Side-by-side scaling figure: single-step eval vs 2000-step rollout.

3×4 grid:
  rows = VQ size (small / medium / large)
  cols = Eval EMD | Rollout EMD | Eval TKE-RSE | Rollout TKE-RSE

Used to answer "does the sc341 < sc917 ordering on rollout metrics also
hold for single-step eval?" — if it does, the ordering is intrinsic to
the (NSP × VQ) pair, not a rollout-divergence artifact.

Output:
  plots/sc341_local/scaling_all_eval_vs_rollout.png

Usage:
    ~/llm/bin/python plot_scaling_all_eval_vs_rollout.py
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

    try:
        runs = list(api.runs(f"{ENTITY}/{ANALYSIS_PROJECT}"))
    except ValueError:
        runs = []
    print(f"Fetching {ANALYSIS_PROJECT}... ({len(runs)} runs)")
    for r in runs:
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


def gather(table, size, sc, kind, key):
    pts = []
    for label in LABELS[sc]:
        rec = table.get((size, sc, label, kind))
        if rec is None: continue
        if rec.get(key) is None: continue
        pts.append(rec)
    pts.sort(key=lambda r: r["params_M"])
    return pts


def plot_panel(ax, table, size, kind, key, floor_key):
    for sc in ("sc341", "sc917"):
        pts = gather(table, size, sc, kind, key)
        if pts:
            x = [r["params_M"] for r in pts]
            y = [r[key] for r in pts]
            ax.plot(x, y, marker=SC_MARKERS[sc], color=SC_COLORS[sc],
                    linewidth=1.8, markersize=7, alpha=0.92)
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

    print("\nCoverage (eval | rollout):")
    for size in SIZES:
        for sc in ("sc341", "sc917"):
            e = sum(1 for l in LABELS[sc] if (size, sc, l, "eval") in table)
            r = sum(1 for l in LABELS[sc] if (size, sc, l, "rollout") in table)
            print(f"  {size}-{sc}: eval {e}/{len(LABELS[sc])}  rollout {r}/{len(LABELS[sc])}")

    fig, axes = plt.subplots(len(SIZES), 4, figsize=(18, 12), sharex=False)

    col_specs = [
        ("eval",    "emd/nsp",     "emd/vqvae",     "Single-step pixel EMD"),
        ("rollout", "emd/nsp",     "emd/vqvae",     "Rollout pixel EMD"),
        ("eval",    "tke_rse/nsp", "tke_rse/vqvae", "Single-step TKE RSE"),
        ("rollout", "tke_rse/nsp", "tke_rse/vqvae", "Rollout TKE RSE"),
    ]

    for i, size in enumerate(SIZES):
        for j, (kind, key, floor_key, title) in enumerate(col_specs):
            plot_panel(axes[i, j], table, size, kind, key, floor_key)
            if i == 0:
                axes[i, j].set_title(title, fontsize=12, fontweight="bold")
            if i == len(SIZES) - 1:
                axes[i, j].set_xlabel("NSP params (M)", fontsize=10)

        ylabel_prefix = f"{SIZE_TITLES[size]}\n"
        axes[i, 0].set_ylabel(f"{ylabel_prefix}\nPixel EMD vs raw GT", fontsize=10)
        axes[i, 1].set_ylabel("Pixel EMD vs raw GT", fontsize=10)
        axes[i, 2].set_ylabel("TKE RSE vs raw GT", fontsize=10)
        axes[i, 3].set_ylabel("TKE RSE vs raw GT", fontsize=10)

    handles = [Line2D([0], [0], color=SC_COLORS[sc], marker=SC_MARKERS[sc],
                      linewidth=2.2, markersize=9, label=SC_LABELS[sc])
               for sc in ("sc341", "sc917")]
    handles.append(Line2D([0], [0], color="0.4", linestyle="--",
                          linewidth=1.6, label="VQ-VAE floor (per sc)"))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=False)

    fig.suptitle("Single-step eval vs 2000-step rollout — all (VQ × sc) curves",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    out = os.path.join(args.output_dir, "scaling_all_eval_vs_rollout.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    main()
