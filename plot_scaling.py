"""Pull metrics from wandb and generate scaling law plots.

Each experiment is treated as a unified model (VQ-VAE + NSP).
Two figures:
  1. Metric vs total params — color = tokens/sample, marker = VQ-VAE size
  2. Metric vs tokens/sample — color = NSP size, marker = VQ-VAE size

Usage:
    source ~/work/ml/bin/activate
    python plot_scaling.py [--output_dir plots/scaling]
"""

import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import wandb

# ── Constants ────────────────────────────────────────────────────────────────

ENTITY = "bigpseud-ucsc"
ANALYSIS_PROJECT = "gust2-analysis"
EVAL_PROJECT = "gust2-eval"

VQVAE_PARAMS = {"small": 31e6, "medium": 57e6, "large": 109e6}
NSP_PARAMS = {"small": 63e6, "medium": 115e6, "large": 215e6}
TOKENS_PER_SAMPLE = {"sc341": 341, "sc917": 917, "sc1941": 1941}

# ── Colorblind-safe palette (Wong 2011) ──────────────────────────────────────
# Safe for deuteranopia, protanopia, and tritanopia.
# Distinguishable via both hue and luminance.

WONG_BLUE = "#0072B2"
WONG_ORANGE = "#E69F00"
WONG_PINK = "#CC79A7"
WONG_VERMILLION = "#D55E00"
WONG_SKYBLUE = "#56B4E9"
WONG_GREEN = "#009E73"

SC_COLORS = {"sc341": WONG_BLUE, "sc917": WONG_ORANGE, "sc1941": WONG_PINK}
SC_LABELS = {"sc341": "341 tok", "sc917": "917 tok", "sc1941": "1941 tok"}

NSP_COLORS = {"small": WONG_SKYBLUE, "medium": WONG_VERMILLION, "large": WONG_GREEN}
NSP_LABELS = {"small": "NSP 63M", "medium": "NSP 115M", "large": "NSP 215M"}

VQVAE_MARKERS = {"small": "o", "medium": "s", "large": "D"}
VQVAE_LABELS = {"small": "VQ 31M", "medium": "VQ 57M", "large": "VQ 109M"}

NSP_SIZES_ORDERED = ["small", "medium", "large"]
VQVAE_SIZES_ORDERED = ["small", "medium", "large"]
SC_ORDERED = ["sc341", "sc917", "sc1941"]

# Row 0 = single-step metrics together, then rollout metrics, EMD last
METRICS = [
    ("cross_entropy",      "Cross-Entropy (nats)"),
    ("pixel_rmse",         "Pixel RMSE"),
    ("tke_rse_nsp",        "TKE Relative Spectral Error"),
    ("enstrophy_rse_nsp",  "Enstrophy RSE"),
    ("emd_nsp",            "Pixel EMD"),
]


def parse_run_name(name):
    parts = name.split("-")
    return parts[0], parts[1], parts[3]


# ── Data fetching ────────────────────────────────────────────────────────────


def fetch_rollout_metrics():
    """Fetch rollout spectral metrics from gust2-analysis + CE/RMSE from gust2-eval."""
    api = wandb.Api()

    analysis = {}
    for r in api.runs(f"{ENTITY}/{ANALYSIS_PROJECT}"):
        name = r.name
        if not any(sz in name for sz in ["small", "medium", "large"]):
            continue
        s = r.summary
        analysis[name] = {
            "tke_rse_nsp": s.get("tke_rse/nsp"),
            "enstrophy_rse_nsp": s.get("enstrophy_rse/nsp"),
            "emd_nsp": s.get("emd/nsp"),
        }

    eval_data = {}
    for r in api.runs(f"{ENTITY}/{EVAL_PROJECT}"):
        name = r.name
        if not any(sz in name for sz in ["small", "medium", "large"]):
            continue
        s = r.summary
        eval_data[name] = {
            "cross_entropy": s.get("cross_entropy"),
            "pixel_rmse": s.get("pixel_rmse"),
        }

    rows = []
    for name in sorted(set(analysis) | set(eval_data)):
        try:
            vqvae_size, sc_config, nsp_size = parse_run_name(name)
        except (IndexError, ValueError):
            continue
        row = {
            "name": name,
            "vqvae_size": vqvae_size,
            "sc_config": sc_config,
            "nsp_size": nsp_size,
            "tokens_per_sample": TOKENS_PER_SAMPLE[sc_config],
            "total_params": VQVAE_PARAMS[vqvae_size] + NSP_PARAMS[nsp_size],
        }
        row.update(analysis.get(name, {}))
        row.update(eval_data.get(name, {}))
        rows.append(row)

    print(f"Fetched {len(rows)} rollout experiments")
    return rows


def fetch_single_step_metrics():
    """Fetch all single-step metrics (CE, RMSE, spectra, EMD) from gust2-eval."""
    api = wandb.Api()

    rows = []
    for r in api.runs(f"{ENTITY}/{EVAL_PROJECT}"):
        name = r.name
        if not any(sz in name for sz in ["small", "medium", "large"]):
            continue
        try:
            vqvae_size, sc_config, nsp_size = parse_run_name(name)
        except (IndexError, ValueError):
            continue
        s = r.summary
        row = {
            "name": name,
            "vqvae_size": vqvae_size,
            "sc_config": sc_config,
            "nsp_size": nsp_size,
            "tokens_per_sample": TOKENS_PER_SAMPLE[sc_config],
            "total_params": VQVAE_PARAMS[vqvae_size] + NSP_PARAMS[nsp_size],
            "cross_entropy": s.get("cross_entropy"),
            "pixel_rmse": s.get("pixel_rmse"),
            "tke_rse_nsp": s.get("tke_rse/nsp"),
            "enstrophy_rse_nsp": s.get("enstrophy_rse/nsp"),
            "emd_nsp": s.get("emd/nsp"),
        }
        rows.append(row)

    print(f"Fetched {len(rows)} single-step experiments")
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────────


def fig_vs_total_params(rows, output_path, title="Scaling vs Total Parameters"):
    """3x2 figure: metric vs total params.
    Color = tokens/sample, marker = VQ-VAE size."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes_flat = axes.flatten()

    for idx, (key, ylabel) in enumerate(METRICS):
        ax = axes_flat[idx]
        for sc in SC_ORDERED:
            for vq_sz in VQVAE_SIZES_ORDERED:
                subset = [r for r in rows
                          if r["sc_config"] == sc
                          and r["vqvae_size"] == vq_sz
                          and r.get(key) is not None]
                if not subset:
                    continue
                subset.sort(key=lambda r: r["total_params"])
                x = np.array([r["total_params"] / 1e6 for r in subset])
                y = np.array([r[key] for r in subset])
                ax.plot(x, y,
                        marker=VQVAE_MARKERS[vq_sz],
                        color=SC_COLORS[sc],
                        linewidth=1.5, markersize=8, alpha=0.85,
                        linestyle="-")

        ax.set_xlabel("Total params (M)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    # Hide unused 6th subplot, use for legend
    ax_legend = axes_flat[5]
    ax_legend.axis("off")

    handles = []
    for sc in SC_ORDERED:
        handles.append(Line2D([0], [0], color=SC_COLORS[sc], linewidth=2.5,
                              label=SC_LABELS[sc]))
    handles.append(Line2D([0], [0], color="none", label=""))
    for vq_sz in VQVAE_SIZES_ORDERED:
        handles.append(Line2D([0], [0], marker=VQVAE_MARKERS[vq_sz],
                              color="0.3", linestyle="none", markersize=9,
                              label=VQVAE_LABELS[vq_sz]))

    ax_legend.legend(handles=handles, loc="center", fontsize=13,
                     frameon=True, fancybox=True, shadow=False,
                     edgecolor="0.8", title="Legend", title_fontsize=13)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


def fig_vs_tokens(rows, output_path, title="Scaling vs Sequence Length"):
    """3x2 figure: metric vs tokens/sample.
    Color = NSP size, marker = VQ-VAE size."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    axes_flat = axes.flatten()

    for idx, (key, ylabel) in enumerate(METRICS):
        ax = axes_flat[idx]
        for nsp_sz in NSP_SIZES_ORDERED:
            for vq_sz in VQVAE_SIZES_ORDERED:
                subset = [r for r in rows
                          if r["nsp_size"] == nsp_sz
                          and r["vqvae_size"] == vq_sz
                          and r.get(key) is not None]
                if not subset:
                    continue
                subset.sort(key=lambda r: r["tokens_per_sample"])
                x = np.array([r["tokens_per_sample"] for r in subset])
                y = np.array([r[key] for r in subset])
                ax.plot(x, y,
                        marker=VQVAE_MARKERS[vq_sz],
                        color=NSP_COLORS[nsp_sz],
                        linewidth=1.5, markersize=8, alpha=0.85,
                        linestyle="-")

        ax.set_xlabel("Tokens / sample", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.set_xticks(sorted(TOKENS_PER_SAMPLE.values()))
        ax.grid(True, alpha=0.3)

    # Legend in 6th subplot
    ax_legend = axes_flat[5]
    ax_legend.axis("off")

    handles = []
    for nsp_sz in NSP_SIZES_ORDERED:
        handles.append(Line2D([0], [0], color=NSP_COLORS[nsp_sz], linewidth=2.5,
                              label=NSP_LABELS[nsp_sz]))
    handles.append(Line2D([0], [0], color="none", label=""))
    for vq_sz in VQVAE_SIZES_ORDERED:
        handles.append(Line2D([0], [0], marker=VQVAE_MARKERS[vq_sz],
                              color="0.3", linestyle="none", markersize=9,
                              label=VQVAE_LABELS[vq_sz]))

    ax_legend.legend(handles=handles, loc="center", fontsize=13,
                     frameon=True, fancybox=True, shadow=False,
                     edgecolor="0.8", title="Legend", title_fontsize=13)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/scaling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rollout_rows = fetch_rollout_metrics()
    single_step_rows = fetch_single_step_metrics()

    print("\n--- Figure 1: rollout metric vs total params ---")
    fig_vs_total_params(rollout_rows,
                        os.path.join(args.output_dir, "rollout_vs_params.png"),
                        title="Rollout Scaling vs Total Parameters")

    print("\n--- Figure 2: rollout metric vs tokens/sample ---")
    fig_vs_tokens(rollout_rows,
                  os.path.join(args.output_dir, "rollout_vs_tokens.png"),
                  title="Rollout Scaling vs Sequence Length")

    print("\n--- Figure 3: single-step metric vs total params ---")
    fig_vs_total_params(single_step_rows,
                        os.path.join(args.output_dir, "single_step_vs_params.png"),
                        title="Single-Step Scaling vs Total Parameters")

    print("\n--- Figure 4: single-step metric vs tokens/sample ---")
    fig_vs_tokens(single_step_rows,
                  os.path.join(args.output_dir, "single_step_vs_tokens.png"),
                  title="Single-Step Scaling vs Sequence Length")

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
