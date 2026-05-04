"""Pull metrics from gust2-analysis-bridges-scaling and render scaling plots.

Two figures (one per analysis type) showing 4 metrics per (sc-config × VQ size)
curve:
  - emd/vqvae        (pixel histogram EMD vs tokenizer GT)
  - emd/nsp          (pixel histogram EMD vs raw pixel GT)
  - tke_rse/vqvae    (TKE relative spectral error vs tokenizer GT)
  - tke_rse/nsp      (TKE relative spectral error vs raw pixel GT)

Run names follow:
  <size>-<sc>-nsp-<label>           rollout analysis (analyze_rollout.py)
  <size>-<sc>-nsp-<label>-eval      single-step (eval_single_step.py)

Per-network NSP parameter count is taken from the trailing numeric portion of
``label`` (s06 → 6 M, s139 → 139 M). The labels were chosen to encode this
intentionally; using the label avoids re-deriving an approximation from arch.

Usage:
    ~/llm/bin/python plot_scaling_bridges.py
    ~/llm/bin/python plot_scaling_bridges.py --output_dir plots/scaling_bridges
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
PROJECT = "gust2-analysis-bridges-scaling"

VQVAE_SIZES = ("small", "medium", "large")
SC_CONFIGS = ("sc341", "sc917", "sc1941")

# Wong colorblind-safe (matches plot_scaling.py)
SC_COLORS = {"sc341": "#0072B2", "sc917": "#E69F00", "sc1941": "#CC79A7"}
VQVAE_MARKERS = {"small": "o", "medium": "s", "large": "D"}
SC_LABELS = {"sc341": "341 tok", "sc917": "917 tok", "sc1941": "1941 tok"}
VQVAE_LABELS = {"small": "VQ small (D=5)", "medium": "VQ medium (D=10)", "large": "VQ large (D=20)"}

METRICS = [
    ("emd/vqvae",     "Pixel EMD vs tokenizer GT"),
    ("emd/nsp",       "Pixel EMD vs raw pixel GT"),
    ("tke_rse/vqvae", "TKE RSE vs tokenizer GT"),
    ("tke_rse/nsp",   "TKE RSE vs raw pixel GT"),
]


def label_to_params_M(label):
    """s06 -> 6.0, s139 -> 139.0; returns None for unrecognized labels."""
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def parse_run_name(name):
    """Returns (size, sc, label, kind) or None if name doesn't match.

    kind is "rollout" for `<size>-<sc>-nsp-<label>` and "eval" for
    `<size>-<sc>-nsp-<label>-eval`.
    """
    parts = name.split("-")
    if len(parts) < 4:
        return None
    size, sc = parts[0], parts[1]
    if size not in VQVAE_SIZES or sc not in SC_CONFIGS:
        return None
    if parts[2] != "nsp":
        return None
    label = parts[3]
    kind = "eval" if (len(parts) >= 5 and parts[4] == "eval") else "rollout"
    return size, sc, label, kind


def fetch_runs():
    """Pull all runs and bucket them by analysis kind."""
    api = wandb.Api()
    try:
        runs = list(api.runs(f"{ENTITY}/{PROJECT}"))
    except ValueError as e:
        if "Could not find project" in str(e):
            print(f"Project {ENTITY}/{PROJECT} doesn't exist yet — "
                  "submit at least one analysis job first.")
            return {"rollout": [], "eval": []}
        raise
    runs.sort(key=lambda r: r.created_at, reverse=True)

    # Dedupe by name (keep newest).
    seen = {}
    for r in runs:
        if r.name not in seen:
            seen[r.name] = r

    by_kind = {"rollout": [], "eval": []}
    for name, r in seen.items():
        parsed = parse_run_name(name)
        if parsed is None:
            continue
        size, sc, label, kind = parsed
        params_M = label_to_params_M(label)
        if params_M is None:
            continue
        s = r.summary
        row = {
            "name": name,
            "size": size,
            "sc": sc,
            "label": label,
            "params_M": params_M,
        }
        for key, _ in METRICS:
            row[key] = s.get(key)
        by_kind[kind].append(row)

    print(f"Fetched {len(by_kind['rollout'])} rollout + "
          f"{len(by_kind['eval'])} single-step runs from {PROJECT}")
    return by_kind


def plot_one(rows, kind, output_path, title):
    """2×2 panel grid; one curve per (sc-config × VQ-size)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes_flat = axes.flatten()

    drawn_curves = 0
    for idx, (key, ylabel) in enumerate(METRICS):
        ax = axes_flat[idx]
        for sc in SC_CONFIGS:
            for size in VQVAE_SIZES:
                subset = [r for r in rows
                          if r["sc"] == sc and r["size"] == size
                          and r.get(key) is not None]
                if not subset:
                    continue
                subset.sort(key=lambda r: r["params_M"])
                x = np.array([r["params_M"] for r in subset])
                y = np.array([r[key] for r in subset])
                ax.plot(x, y,
                        marker=VQVAE_MARKERS[size],
                        color=SC_COLORS[sc],
                        linewidth=1.5, markersize=8, alpha=0.85,
                        linestyle="-")
                drawn_curves += 1

        ax.set_xlabel("NSP params (M)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)

    handles = []
    for sc in SC_CONFIGS:
        handles.append(Line2D([0], [0], color=SC_COLORS[sc], linewidth=2.5,
                              label=SC_LABELS[sc]))
    handles.append(Line2D([0], [0], color="none", label=""))
    for size in VQVAE_SIZES:
        handles.append(Line2D([0], [0], marker=VQVAE_MARKERS[size],
                              color="0.3", linestyle="none", markersize=9,
                              label=VQVAE_LABELS[size]))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=False)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path}  ({drawn_curves} curves drawn from {len(rows)} runs)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/scaling_bridges")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    by_kind = fetch_runs()

    if by_kind["rollout"]:
        print("\n--- Rollout (2000-step AR) ---")
        plot_one(by_kind["rollout"], "rollout",
                 os.path.join(args.output_dir, "rollout_vs_params.png"),
                 title="Rollout (2000 steps) — Scaling vs NSP Params")
    else:
        print("\n--- Rollout (no runs yet — skipping plot) ---")

    if by_kind["eval"]:
        print("\n--- Single-step (2000 teacher-forced pairs) ---")
        plot_one(by_kind["eval"], "eval",
                 os.path.join(args.output_dir, "single_step_vs_params.png"),
                 title="Single-step — Scaling vs NSP Params")
    else:
        print("\n--- Single-step (no runs yet — skipping plot) ---")

    # Also dump a flat table so we can sanity-check before plotting.
    print("\n--- Run table ---")
    for kind in ("rollout", "eval"):
        rows = sorted(by_kind[kind],
                      key=lambda r: (r["size"], r["sc"], r["params_M"]))
        if not rows:
            continue
        print(f"\n{kind} ({len(rows)} runs)")
        hdr = f"{'name':<32} {'pM':>5}  " + "  ".join(f"{k:>16}" for k, _ in METRICS)
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            vals = []
            for k, _ in METRICS:
                v = r.get(k)
                vals.append(f"{v:>16.4g}" if v is not None else f"{'--':>16}")
            print(f"{r['name']:<32} {r['params_M']:>5.0f}  " + "  ".join(vals))


if __name__ == "__main__":
    main()
