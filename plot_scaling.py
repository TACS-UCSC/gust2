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
ANALYSIS_PROJECT = "gust2-analysis-derecho"
SAMPLING_PROJECT = "gust2-sampling-derecho"
EVAL_PROJECT = "gust2-eval-derecho"

VQVAE_PARAMS = {"small": 31e6, "medium": 57e6, "large": 109e6}
NSP_PARAMS = {
    "nano":   2e6,
    "micro":  6e6,
    "mini":   14e6,
    "small":  63e6,
    "medium": 115e6,
    "large":  215e6,
}
TOKENS_PER_SAMPLE = {"sc341": 341, "sc917": 917, "sc1941": 1941}

# Trainable scales per sc config (1x1 is free context; we predict from 2x2 up).
SC_TRAINABLE_SCALES = {
    "sc341":  [2, 4, 8, 16],
    "sc917":  [2, 4, 8, 16, 24],
    "sc1941": [2, 4, 8, 16, 24, 32],
}

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

# Tol-inspired extensions for the ablation sizes; Wong-6 kept for small/medium/large.
TOL_ROSE   = "#882255"
TOL_GOLD   = "#DDAA33"
TOL_TEAL   = "#44AA99"

NSP_COLORS = {
    "nano":   TOL_ROSE,
    "micro":  TOL_GOLD,
    "mini":   TOL_TEAL,
    "small":  WONG_SKYBLUE,
    "medium": WONG_VERMILLION,
    "large":  WONG_GREEN,
}
NSP_LABELS = {
    "nano":   "NSP 2M",
    "micro":  "NSP 6M",
    "mini":   "NSP 14M",
    "small":  "NSP 63M",
    "medium": "NSP 115M",
    "large":  "NSP 215M",
}

VQVAE_MARKERS = {"small": "o", "medium": "s", "large": "D"}
VQVAE_LABELS = {"small": "VQ 31M", "medium": "VQ 57M", "large": "VQ 109M"}

# Used when --sc filter collapses to a single config, so SC-as-color becomes
# degenerate and we color by VQ-VAE instead. Wong-palette colors not used by
# NSP_COLORS so the two legends remain visually distinct.
VQVAE_COLORS = {
    "small":  WONG_BLUE,        # deep blue
    "medium": WONG_PINK,        # pink
    "large":  WONG_ORANGE,      # orange
}

NSP_SIZES_ORDERED = ["nano", "micro", "mini", "small", "medium", "large"]
VQVAE_SIZES_ORDERED = ["small", "medium", "large"]
SC_ORDERED = ["sc341", "sc917", "sc1941"]

# Row 0 = single-step metrics together, then rollout metrics, EMD last
METRICS = [
    ("ce_per_token",       "Cross-Entropy per Token (nats)"),
    ("pixel_rmse",         "Pixel RMSE"),
    ("tke_rse_nsp",        "TKE Relative Spectral Error"),
    ("enstrophy_rse_nsp",  "Enstrophy RSE"),
    ("emd_nsp",            "Pixel EMD"),
]


def compute_ce_per_token(summary, sc_config):
    """Aggregate per-scale CEs into a per-token mean.

    Prefers the server-side `ce_per_token` field if eval_single_step.py logged
    it (runs after 2026-04-23). Falls back to the legacy client-side
    computation (token-weighted mean of per-scale CEs) for older runs.
    Returns None if neither source is available.
    """
    ce_logged = summary.get("ce_per_token")
    if ce_logged is not None:
        return ce_logged

    scales = SC_TRAINABLE_SCALES[sc_config]
    total_nats = 0.0
    total_tokens = 0
    for s in scales:
        ce = summary.get(f"ce/scale_{s}x{s}")
        if ce is None:
            return None
        total_nats += s * s * ce
        total_tokens += s * s
    return total_nats / total_tokens


def parse_run_name(name):
    parts = name.split("-")
    return parts[0], parts[1], parts[3]


# ── Data fetching ────────────────────────────────────────────────────────────


def _latest_per_name(project):
    """Return newest run per name (dedupes retrained runs)."""
    api = wandb.Api()
    runs = list(api.runs(f"{ENTITY}/{project}"))
    runs.sort(key=lambda r: r.created_at, reverse=True)
    seen = {}
    for r in runs:
        if r.name not in seen:
            seen[r.name] = r
    return seen


def fetch_rollout_metrics():
    """Fetch rollout spectral metrics from gust2-analysis + CE/RMSE from gust2-eval."""
    analysis_runs = _latest_per_name(ANALYSIS_PROJECT)
    eval_runs = _latest_per_name(EVAL_PROJECT)

    analysis = {}
    for name, r in analysis_runs.items():
        if not any(sz in name for sz in ["small", "medium", "large"]):
            continue
        s = r.summary
        analysis[name] = {
            "tke_rse_nsp": s.get("tke_rse/nsp"),
            "enstrophy_rse_nsp": s.get("enstrophy_rse/nsp"),
            "emd_nsp": s.get("emd/nsp"),
        }

    eval_data = {}
    for name, r in eval_runs.items():
        if not any(sz in name for sz in ["small", "medium", "large"]):
            continue
        try:
            _, sc_config, _ = parse_run_name(name)
        except (IndexError, ValueError):
            continue
        s = r.summary
        eval_data[name] = {
            "cross_entropy": s.get("cross_entropy"),
            "ce_per_token": compute_ce_per_token(s, sc_config),
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


def fetch_sampling_rollout_metrics(temperature=None):
    """Pull rollouts from gust2-sampling (temperature sweep).

    If ``temperature`` is None, reduces across T per (vqvae, sc, nsp) group by
    taking the min per metric independently (the Pareto lower bound from T
    tuning). If a specific ``temperature`` is given, returns only runs at
    that T, no reduction.

    Run names follow ``<vqvae>-<sc>-nsp-<nsp>-T<temp>-s<seed>``.
    CE/RMSE come from gust2-eval (deterministic, no T).
    """
    sampling_runs = _latest_per_name(SAMPLING_PROJECT)
    eval_runs = _latest_per_name(EVAL_PROJECT)

    # Group sampling runs by (vqvae_size, sc_config, nsp_size) across T.
    groups = {}
    for name, r in sampling_runs.items():
        parts = name.split("-")
        if len(parts) < 6:
            continue
        vqvae_size, sc_config, _, nsp_size, temp_tag, _ = parts[:6]
        try:
            temp = float(temp_tag.lstrip("T"))
        except ValueError:
            continue
        if temperature is not None and abs(temp - temperature) > 1e-6:
            continue
        s = r.summary
        metrics = {
            "tke_rse_nsp":       s.get("tke_rse/nsp"),
            "enstrophy_rse_nsp": s.get("enstrophy_rse/nsp"),
            "emd_nsp":           s.get("emd/nsp"),
        }
        key = (vqvae_size, sc_config, nsp_size)
        groups.setdefault(key, []).append((temp, metrics))

    rollout_rows = []
    for (vqvae_size, sc_config, nsp_size), entries in groups.items():
        best = {}
        best_t = {}
        for m in ("tke_rse_nsp", "enstrophy_rse_nsp", "emd_nsp"):
            vals = [(t, e[m]) for t, e in entries if e[m] is not None]
            if not vals:
                continue
            t, v = min(vals, key=lambda tv: tv[1])
            best[m] = v
            best_t[m] = t

        name = f"{vqvae_size}-{sc_config}-nsp-{nsp_size}"
        if temperature is not None:
            name = f"{name}-T{temperature}"
        row = {
            "name": name,
            "vqvae_size": vqvae_size,
            "sc_config": sc_config,
            "nsp_size": nsp_size,
            "tokens_per_sample": TOKENS_PER_SAMPLE[sc_config],
            "total_params": VQVAE_PARAMS[vqvae_size] + NSP_PARAMS[nsp_size],
            "best_temps": best_t,
        }
        row.update(best)

        # CE/RMSE from deterministic eval (no T suffix there).
        lookup_name = f"{vqvae_size}-{sc_config}-nsp-{nsp_size}"
        for ev_name, ev in eval_runs.items():
            if ev_name == lookup_name:
                s = ev.summary
                row["cross_entropy"] = s.get("cross_entropy")
                row["ce_per_token"] = compute_ce_per_token(s, sc_config)
                row["pixel_rmse"] = s.get("pixel_rmse")
                break

        rollout_rows.append(row)

    print(f"Fetched {len(rollout_rows)} sampling-rollout experiments "
          f"(best-T per metric across {len(sampling_runs)} runs)")
    return rollout_rows


def fetch_single_step_metrics():
    """Fetch all single-step metrics (CE, RMSE, spectra, EMD) from gust2-eval."""
    eval_runs = _latest_per_name(EVAL_PROJECT)

    rows = []
    for name, r in eval_runs.items():
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
            "ce_per_token": compute_ce_per_token(s, sc_config),
            "pixel_rmse": s.get("pixel_rmse"),
            "tke_rse_nsp": s.get("tke_rse/nsp"),
            "enstrophy_rse_nsp": s.get("enstrophy_rse/nsp"),
            "emd_nsp": s.get("emd/nsp"),
        }
        rows.append(row)

    print(f"Fetched {len(rows)} single-step experiments")
    return rows


# ── Plotting ─────────────────────────────────────────────────────────────────


def fig_vs_total_params(rows, output_path, title="Scaling vs Total Parameters",
                         color_by="sc"):
    """3x2 figure: metric vs total params.

    color_by:
      "sc"    — color = tokens/sample (original), marker = VQ-VAE size.
      "vqvae" — color = VQ-VAE size. Useful when the plot is filtered to a
                single sc config (SC-color would be degenerate) and you want
                per-VQ traces visually distinct.
    """
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
                if color_by == "vqvae":
                    color = VQVAE_COLORS[vq_sz]
                else:
                    color = SC_COLORS[sc]
                ax.plot(x, y,
                        marker=VQVAE_MARKERS[vq_sz],
                        color=color,
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
    if color_by == "vqvae":
        for vq_sz in VQVAE_SIZES_ORDERED:
            handles.append(Line2D([0], [0],
                                  marker=VQVAE_MARKERS[vq_sz],
                                  color=VQVAE_COLORS[vq_sz],
                                  linewidth=2.5, markersize=9,
                                  label=VQVAE_LABELS[vq_sz]))
    else:
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
    parser.add_argument("--vqvae", type=str, default=None,
                        help="Filter to VQ-VAE size (small, medium, large)")
    parser.add_argument("--sc", type=str, default=None, nargs="+",
                        choices=list(SC_ORDERED),
                        help="Filter to one or more sc configs (default: all). "
                             "When exactly one sc config is selected, "
                             "vs_tokens plots are skipped (degenerate), and "
                             "vs_params lines are colored by VQ-VAE size.")
    parser.add_argument("--nsp", type=str, default=None, nargs="+",
                        choices=list(NSP_SIZES_ORDERED),
                        help="Filter to a subset of NSP sizes (default: all). "
                             "Example: --nsp nano micro mini small to drop "
                             "the heavily-overfit medium/large cells.")
    parser.add_argument("--sampling_rollout", action="store_true",
                        help="Use gust2-sampling (temperature sweep) for "
                             "rollout metrics; takes best T per metric.")
    parser.add_argument("--per_temperature", action="store_true",
                        help="Emit one set of sampling-rollout plots per "
                             "temperature instead of reducing across T.")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.7, 1.0, 1.2],
                        help="Temperatures to plot when --per_temperature.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    per_temperature_runs = []
    if args.per_temperature:
        for T in args.temperatures:
            rows = fetch_sampling_rollout_metrics(temperature=T)
            per_temperature_runs.append(
                (rows, f" (T={T})", f"_T{T}"))
        rollout_rows = None
    elif args.sampling_rollout:
        rollout_rows = fetch_sampling_rollout_metrics()
        rollout_title_suffix = " (best-T sampling)"
        rollout_file_suffix = "_sampling"
        print("\nBest temperature per (model, metric):")
        print(f"  {'config':<28}{'tke':>5}{'ens':>5}{'emd':>5}")
        for r in sorted(rollout_rows,
                        key=lambda r: (r["vqvae_size"], r["sc_config"], r["nsp_size"])):
            bt = r.get("best_temps", {})
            print(f"  {r['name']:<28}"
                  f"{bt.get('tke_rse_nsp', '-'):>5}"
                  f"{bt.get('enstrophy_rse_nsp', '-'):>5}"
                  f"{bt.get('emd_nsp', '-'):>5}")
    else:
        rollout_rows = fetch_rollout_metrics()
        rollout_title_suffix = ""
        rollout_file_suffix = ""
    single_step_rows = fetch_single_step_metrics()

    if args.vqvae:
        if rollout_rows is not None:
            rollout_rows = [r for r in rollout_rows if r["vqvae_size"] == args.vqvae]
        per_temperature_runs = [
            ([r for r in rows if r["vqvae_size"] == args.vqvae], ts, fs)
            for rows, ts, fs in per_temperature_runs
        ]
        single_step_rows = [r for r in single_step_rows if r["vqvae_size"] == args.vqvae]
        print(f"Filtered to vqvae={args.vqvae}")

    if args.sc:
        sc_filter = set(args.sc)
        if rollout_rows is not None:
            rollout_rows = [r for r in rollout_rows if r["sc_config"] in sc_filter]
        per_temperature_runs = [
            ([r for r in rows if r["sc_config"] in sc_filter], ts, fs)
            for rows, ts, fs in per_temperature_runs
        ]
        single_step_rows = [r for r in single_step_rows if r["sc_config"] in sc_filter]
        print(f"Filtered to sc={sorted(sc_filter)}")

    if args.nsp:
        nsp_filter = set(args.nsp)
        if rollout_rows is not None:
            rollout_rows = [r for r in rollout_rows if r["nsp_size"] in nsp_filter]
        per_temperature_runs = [
            ([r for r in rows if r["nsp_size"] in nsp_filter], ts, fs)
            for rows, ts, fs in per_temperature_runs
        ]
        single_step_rows = [r for r in single_step_rows if r["nsp_size"] in nsp_filter]
        print(f"Filtered to nsp={sorted(nsp_filter)}")

    skip_vs_tokens = args.sc is not None and len(args.sc) == 1
    color_by = "vqvae" if skip_vs_tokens else "sc"

    if args.per_temperature:
        for rows, ts, fs in per_temperature_runs:
            print(f"\n--- Rollout plots for T{fs[2:]} ({len(rows)} experiments) ---")
            fig_vs_total_params(
                rows,
                os.path.join(args.output_dir, f"rollout_vs_params{fs}.png"),
                title=f"Rollout Scaling vs Total Parameters{ts}",
                color_by=color_by)
            if not skip_vs_tokens:
                fig_vs_tokens(
                    rows,
                    os.path.join(args.output_dir, f"rollout_vs_tokens{fs}.png"),
                    title=f"Rollout Scaling vs Sequence Length{ts}")
    else:
        print("\n--- Figure 1: rollout metric vs total params ---")
        fig_vs_total_params(
            rollout_rows,
            os.path.join(args.output_dir, f"rollout_vs_params{rollout_file_suffix}.png"),
            title=f"Rollout Scaling vs Total Parameters{rollout_title_suffix}",
            color_by=color_by)

        if not skip_vs_tokens:
            print("\n--- Figure 2: rollout metric vs tokens/sample ---")
            fig_vs_tokens(
                rollout_rows,
                os.path.join(args.output_dir, f"rollout_vs_tokens{rollout_file_suffix}.png"),
                title=f"Rollout Scaling vs Sequence Length{rollout_title_suffix}")

    print("\n--- Figure 3: single-step metric vs total params ---")
    fig_vs_total_params(single_step_rows,
                        os.path.join(args.output_dir, "single_step_vs_params.png"),
                        title="Single-Step Scaling vs Total Parameters",
                        color_by=color_by)

    if not skip_vs_tokens:
        print("\n--- Figure 4: single-step metric vs tokens/sample ---")
        fig_vs_tokens(single_step_rows,
                      os.path.join(args.output_dir, "single_step_vs_tokens.png"),
                      title="Single-Step Scaling vs Sequence Length")

    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
