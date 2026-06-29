"""plot_scale_drift.py — per-scale token-distribution drift vs temperature, per
token count. The refined companion to plot_climate_temp_band: resolves the
single pixel-EMD into WHERE (which scale) and WHICH WAY (collapse vs over-spread)
a rollout leaves the manifold.

Pulls gust2-drift-{small,medium,large} (logged by measure_drift_sweep.py). Each
run has summary drift/js/s<scale>, drift/dH/s<scale> + config {size,sc,arch,T}.
Top row: JS(scale) vs T, mean over the cells at each T (drift magnitude).
Bottom row: signed entropy-gap dH(scale) vs T (<0 collapse, >0 over-spread).
A-priori T* per token count marked (sc341 1.0 / sc917 1.7 / sc1941 1.8).

  ~/llm/bin/python plot_scale_drift.py
"""
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "bigpseud-ucsc"
SCS = ["sc341", "sc917", "sc1941"]
APRIORI_T = {"sc341": 1.0, "sc917": 1.7, "sc1941": 1.8}


def fetch(sizes):
    """data[sc][scale][T] = list of js ; and same for dH."""
    api = wandb.Api()
    js = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dH = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for size in sizes:
        try:
            runs = list(api.runs(f"{ENTITY}/gust2-drift-{size}"))
        except ValueError:
            print(f"[warn] gust2-drift-{size} not found"); continue
        for r in runs:
            sc = r.config.get("sc"); T = r.config.get("T")
            if sc is None or T is None:
                continue
            for k, v in r.summary.items():
                if k.startswith("drift/js/s"):
                    js[sc][int(k.split("s")[-1])][float(T)].append(float(v))
                elif k.startswith("drift/dH/s"):
                    dH[sc][int(k.split("s")[-1])][float(T)].append(float(v))
    return js, dH


def _mean_curve(scaleT):
    Ts = sorted(scaleT)
    return Ts, [float(np.mean(scaleT[t])) for t in Ts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    ap.add_argument("--output", default="plots/scale_drift/scale_drift.png")
    a = ap.parse_args()
    js, dH = fetch(a.sizes)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex="col")
    for ci, sc in enumerate(SCS):
        scales = sorted(js[sc])
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(scales)))
        for col, (metric, ylab, ax) in enumerate([
                (js, "JS divergence (drift magnitude)", axes[0, ci]),
                (dH, "signed entropy gap dH (bits)", axes[1, ci])]):
            for sci, s in enumerate(scales):
                if s not in metric[sc]:
                    continue
                Ts, ys = _mean_curve(metric[sc][s])
                ax.plot(Ts, ys, marker="o", ms=4, color=cmap[sci], label=f"scale {s}")
            ax.axvline(APRIORI_T[sc], color="k", ls="--", lw=1.4, alpha=0.7)
            if metric is dH:
                ax.axhline(0, color="0.6", lw=0.8)
            ax.grid(alpha=0.3)
            if ci == 0:
                ax.set_ylabel(ylab)
        axes[0, ci].set_title(f"{sc}  (a-priori T*={APRIORI_T[sc]})")
        axes[1, ci].set_xlabel("sampling temperature T")
        axes[0, ci].legend(fontsize=7, ncol=2)
    axes[0, 2].text(0.98, 0.95, "high JS at coarse scale = collapse;\n"
                    "high JS at fine scale = over-spread", transform=axes[0, 2].transAxes,
                    fontsize=8, ha="right", va="top", color="0.3")
    fig.suptitle("Per-scale token-distribution drift vs temperature (mean over cells), by token count.\n"
                 "Refined companion to pixel-EMD: localizes drift to coarse-vs-fine scales and resolves collapse (dH<0) vs over-spread (dH>0).",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    fig.savefig(a.output, dpi=140, bbox_inches="tight")
    print(f"saved {a.output}")


if __name__ == "__main__":
    main()
