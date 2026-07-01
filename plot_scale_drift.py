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


# Colorblind-safe, redundant encoding (robust to SEVERE CVD incl. monochromacy):
# cividis is CVD-designed + lightness-ordered (so scale order survives even with
# zero color perception), AND each scale gets a distinct MARKER so the series are
# separable by shape alone. Fixed scale->(color,marker) map so a given scale looks
# identical across all three panels. Scale 1 (global token, JS=dH=0) is dropped.
ALL_SCALES = [2, 4, 8, 16, 24, 32]
_MARKERS = ["o", "s", "^", "D", "v", "P"]
_COLORS = plt.cm.cividis(np.linspace(0.04, 0.96, len(ALL_SCALES)))
SCALE_STYLE = {s: (_COLORS[i], _MARKERS[i]) for i, s in enumerate(ALL_SCALES)}


def render(js, dH, output, mean_overlay):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ci, sc in enumerate(SCS):
        ax = axes[ci]
        scales = [s for s in sorted(js[sc]) if s in SCALE_STYLE]   # drop scale 1
        for s in scales:
            if s not in js[sc]:
                continue
            col, mk = SCALE_STYLE[s]
            Ts, ys = _mean_curve(js[sc][s])
            ax.plot(Ts, ys, marker=mk, ms=7, mew=0.6, mec="0.15",
                    color=col, lw=1.8, alpha=0.5, label=f"scale {s}")
        # mean-JS overlay: heavy BLACK line (max contrast under all CVD), opaque so
        # it pops against the faded per-scale lines. Mean over shown scales (>=2).
        if mean_overlay:
            Tset = sorted({t for s in scales for t in js[sc][s]})
            ym = [float(np.mean([np.mean(js[sc][s][t])
                                 for s in scales if t in js[sc][s]])) for t in Tset]
            ax.plot(Tset, ym, color="black", lw=3.0, marker="*", ms=11,
                    label="mean (scales >=2)", zorder=10)
        ax.axvline(APRIORI_T[sc], color="0.45", ls="--", lw=1.6, alpha=0.9, zorder=1)
        ax.grid(alpha=0.3)
        ax.set_title(f"{sc}  (a-priori T*={APRIORI_T[sc]})")
        ax.set_xlabel("sampling temperature T")
        if ci == 0:
            ax.set_ylabel("JS divergence (drift magnitude)")
    # single legend, in the sc1941 (rightmost) panel, handles forced opaque so the
    # faded per-scale lines are still legible in the key.
    leg = axes[2].legend(fontsize=8, ncol=2, loc="upper right")
    for lh in getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])):
        lh.set_alpha(1.0)
    fig.suptitle("Per-scale token-distribution drift vs temperature (mean over cells), by token count."
                 + (" [mean-JS overlay]" if mean_overlay else ""),
                 fontsize=12, y=1.02)
    fig.tight_layout()
    import os
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {output}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    ap.add_argument("--output_dir", default="plots/scale_drift")
    a = ap.parse_args()
    js, dH = fetch(a.sizes)
    render(js, dH, f"{a.output_dir}/scale_drift_cb.png", mean_overlay=False)
    render(js, dH, f"{a.output_dir}/scale_drift_cb_mean.png", mean_overlay=True)


if __name__ == "__main__":
    main()
