"""Companion to plot_rollout_curves.py — overlays single-step eval and
2000-step rollout on the same EMD / TKE-RSE panels so the two regimes
can be compared directly.

One figure per VQ size (small / medium / large), 3 panels each:
  - Train CE per token                (context, single line per sc)
  - Pixel EMD vs raw GT               (eval AND rollout per sc)
  - TKE RSE vs raw GT                 (eval AND rollout per sc)

Convention:
  - Color = sc-config (sc341 navy, sc917 amber, both CVD-safe)
  - Rollout = solid line, filled marker
  - Eval    = dotted line, open marker
  - VQ-VAE floor (same field per group) = dashed reference

Output:
  plots/sc341_local/rollout_vs_eval_curves_small.png
  plots/sc341_local/rollout_vs_eval_curves_medium.png
  plots/sc341_local/rollout_vs_eval_curves_large.png

Usage:
    ~/llm/bin/python plot_rollout_vs_eval_curves.py
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
SC_COLORS  = {"sc341": "#1F4E79", "sc917": "#FFC107"}
SC_MARKERS = {"sc341": "o",       "sc917": "s"}
SC_LABELS  = {"sc341": "sc341 (341 tok/frame)", "sc917": "sc917 (917 tok/frame)"}
SIZES = ("small", "medium", "large")
SIZE_TITLES = {
    "small":  "VQ small (D=5)",
    "medium": "VQ medium (D=10)",
    "large":  "VQ large (D=20)",
}

# kind → (linestyle, fillstyle, label suffix)
KIND_STYLE = {
    "rollout": ("-",  "full", "rollout"),
    "eval":    (":",  "none", "single-step"),
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


def plot_train_panel(ax, table, size):
    for sc in ("sc341", "sc917"):
        pts = gather(table, size, sc, "train")
        if not pts: continue
        x = [r["params_M"] for r in pts]
        y = [r["train_loss"] for r in pts]
        ax.plot(x, y, marker=SC_MARKERS[sc], color=SC_COLORS[sc],
                linewidth=1.8, markersize=8, alpha=0.92)
    ax.set_xscale("log")
    ax.set_xlabel("NSP params (M)", fontsize=11)
    ax.set_ylabel("Train CE (epoch 400)", fontsize=11)
    ax.set_title("Train CE per token", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def plot_metric_panel(ax, table, size, key, floor_key, ylabel, title):
    for sc in ("sc341", "sc917"):
        for kind in ("rollout", "eval"):
            pts = gather(table, size, sc, kind, key)
            if not pts: continue
            x = [r["params_M"] for r in pts]
            y = [r[key] for r in pts]
            ls, fs, _ = KIND_STYLE[kind]
            ax.plot(x, y, marker=SC_MARKERS[sc], color=SC_COLORS[sc],
                    linewidth=1.8, markersize=8, alpha=0.92,
                    linestyle=ls, fillstyle=fs,
                    markerfacecolor=(SC_COLORS[sc] if fs == "full" else "white"),
                    markeredgewidth=1.6, markeredgecolor=SC_COLORS[sc])

        # VQ-VAE floor — same field per (size, sc); use the rollout floor
        # since rollout coverage is complete.
        floor = vqvae_floor(table, size, sc, "rollout", floor_key)
        if floor is not None:
            ax.axhline(floor, color=SC_COLORS[sc], linestyle="--",
                       linewidth=1.2, alpha=0.55)

    ax.set_xscale("log")
    ax.set_xlabel("NSP params (M)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)


def render_size(table, size, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    plot_train_panel(axes[0], table, size)
    plot_metric_panel(axes[1], table, size, "emd/nsp", "emd/vqvae",
                      ylabel="Pixel EMD vs raw GT",
                      title="Pixel EMD: single-step vs rollout")
    plot_metric_panel(axes[2], table, size, "tke_rse/nsp", "tke_rse/vqvae",
                      ylabel="TKE RSE vs raw GT",
                      title="TKE RSE: single-step vs rollout")

    handles = []
    for sc in ("sc341", "sc917"):
        for kind in ("rollout", "eval"):
            ls, fs, suffix = KIND_STYLE[kind]
            handles.append(Line2D(
                [0], [0], color=SC_COLORS[sc], marker=SC_MARKERS[sc],
                linestyle=ls, linewidth=2.0, markersize=9,
                markerfacecolor=(SC_COLORS[sc] if fs == "full" else "white"),
                markeredgewidth=1.6, markeredgecolor=SC_COLORS[sc],
                label=f"{SC_LABELS[sc]} — {suffix}"))
    handles.append(Line2D([0], [0], color="0.4", linestyle="--",
                          linewidth=1.6, label="VQ-VAE floor (per sc)"))

    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05), fontsize=10, frameon=False)
    fig.suptitle(f"{SIZE_TITLES[size]} — single-step vs rollout metrics",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/sc341_local")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    table = fetch()

    print("\nCoverage (train | eval | rollout):")
    for size in SIZES:
        for sc in ("sc341", "sc917"):
            t = sum(1 for l in LABELS[sc] if (size, sc, l, "train")   in table)
            e = sum(1 for l in LABELS[sc] if (size, sc, l, "eval")    in table)
            r = sum(1 for l in LABELS[sc] if (size, sc, l, "rollout") in table)
            print(f"  {size}-{sc}: train {t}/{len(LABELS[sc])}  "
                  f"eval {e}/{len(LABELS[sc])}  rollout {r}/{len(LABELS[sc])}")

    print()
    for size in SIZES:
        out = os.path.join(args.output_dir, f"rollout_vs_eval_curves_{size}.png")
        render_size(table, size, out)


if __name__ == "__main__":
    main()
