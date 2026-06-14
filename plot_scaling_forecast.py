"""plot_scaling_forecast.py — short-horizon forecast-skill scaling figures.

The short-term companion to plot_scaling_tempopt.py. Reads the forecast sweep
(gust2-scaling-forecast-{small,medium,large}, runs named
<size>-<sc>-nsp-<arch>-T<temp>), which logs per-lead-time metrics
emd/nsp/k<k>, tke_rse/nsp/k<k> for k in {1,2,5,10}. For each (cell, horizon)
it picks the best temperature by min pixel-EMD at that lead, then plots the
forecast scaling law and compares it to the long-run climate scaling law from
plot_scaling_tempopt.py.

Figures -> --output_dir (default plots/scaling_forecast/):
  scaling_emd_k<k>.png        pixel-EMD vs NSP params at lead k (panel per VQ
                              size, line per sc, VQ floor) — mirrors the
                              long-term scaling_emd.png so you can flip between
  scaling_tke_k<k>.png        TKE-RSE vs NSP params at lead k
  forecast_vs_longterm_emd.png  3x3 grid (sc rows × VQ-size cols); per cell,
                              EMD-vs-params for k=1,2,5,10 (viridis) overlaid
                              with the long-term curve (black) — does the
                              scaling law change shape at short horizons?
  forecast_best_temperature.png  3x3 grid; best-T vs params per horizon +
                              long-term — does the short-term optimum diverge
                              (colder) from the long-term one?
  scaling_forecast.csv        per (cell, horizon) best T + metrics

Run locally after the forecast sweep's analysis runs have logged:
  ~/llm/bin/python plot_scaling_forecast.py
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb

# Reuse parsing + constants + the long-term-style metric-vs-params figure.
from plot_scaling_tempopt import (
    ENTITY, SIZES, SCS, SC_COLOR, VQVAE_PARAMS_M, ARCH_LAYERS,
    label_to_params_M, parse_run_name, fig_metric_vs_params,
)


def fetch_forecast(projects, entity, horizons):
    """Per-horizon best-T tables.

    Returns tables[k][(size, sc, arch)] = {params_M, best_T, n_T, n_layer,
    emd, tke, emd_vq}, with best T chosen per (cell, horizon) by min
    emd/nsp/k<k>. Schema matches what fig_metric_vs_params expects.
    """
    api = wandb.Api()
    runs = []
    for project in projects:
        try:
            pr = list(api.runs(f"{entity}/{project}"))
        except ValueError:
            print(f"[warn] project {project} not found, skipping")
            continue
        print(f"{project}: {len(pr)} runs")
        runs += pr

    per_cell = defaultdict(list)            # (size,sc,arch) -> [(T, rec)]
    for r in runs:
        if r.state != "finished":
            continue
        parsed = parse_run_name(r.name)
        if parsed is None:
            continue
        size, sc, arch, temp = parsed
        rec = {"emd_vq": r.summary.get("emd/vqvae"), "per_k": {}}
        for k in horizons:
            emd = r.summary.get(f"emd/nsp/k{k}")
            if emd is None:
                continue
            rec["per_k"][k] = {
                "emd": float(emd),
                "tke": r.summary.get(f"tke_rse/nsp/k{k}"),
            }
        if rec["per_k"]:
            per_cell[(size, sc, arch)].append((temp, rec))

    tables = {k: {} for k in horizons}
    for cell, entries in per_cell.items():
        size, sc, arch = cell
        pm = label_to_params_M(arch)
        if pm is None:
            continue
        for k in horizons:
            cand = [(T, rec) for (T, rec) in entries if k in rec["per_k"]]
            if not cand:
                continue
            best_T, best = min(cand, key=lambda e: e[1]["per_k"][k]["emd"])
            tables[k][cell] = {
                "params_M": pm, "best_T": best_T, "n_T": len(cand),
                "n_layer": ARCH_LAYERS.get((sc, arch)),
                "emd": best["per_k"][k]["emd"],
                "tke": best["per_k"][k]["tke"],
                "emd_vq": best["emd_vq"],
            }
    for k in horizons:
        print(f"  k={k}: {len(tables[k])} cells with metrics")
    return tables


def load_lt_csv(path):
    """Long-term per-cell record from plot_scaling_tempopt's CSV, or None."""
    if not path or not os.path.exists(path):
        print(f"[warn] long-term CSV not found at {path!r}; skipping overlays")
        return None

    def fl(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    lt = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            lt[(row["size"], row["sc"], row["arch"])] = {
                "params_M": fl(row["params_M"]),
                "best_T": fl(row["best_T"]),
                "emd_nsp": fl(row["emd_nsp"]),
                "tke_rse": fl(row["tke_rse"]),
            }
    print(f"loaded long-term CSV: {len(lt)} cells from {path}")
    return lt


def fig_overlay(tables, lt, horizons, metric, lt_key, ylabel, title, out):
    """3x3 grid (sc rows × VQ-size cols). Per cell: metric vs params for each
    horizon (viridis) + the long-term curve (black). Long-term omitted if
    lt is None."""
    fig, axes = plt.subplots(len(SCS), len(SIZES), figsize=(15, 12),
                             sharex=False, sharey="row",
                             constrained_layout=True)
    hcolors = plt.cm.viridis(np.linspace(0.15, 0.9, len(horizons)))
    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            ax = axes[i, j]
            for hi, k in enumerate(horizons):
                pts = sorted(
                    [(v["params_M"], v[metric])
                     for (s, c, a), v in tables[k].items()
                     if s == size and c == sc and v.get(metric) is not None],
                    key=lambda x: x[0])
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            "-o", color=hcolors[hi], lw=1.5, ms=4,
                            label=f"k={k}")
            if lt is not None:
                lp = sorted(
                    [(v["params_M"], v[lt_key])
                     for (s, c, a), v in lt.items()
                     if s == size and c == sc
                     and v.get(lt_key) is not None and v.get("params_M")],
                    key=lambda x: x[0])
                if lp:
                    ax.plot([p[0] for p in lp], [p[1] for p in lp],
                            "-s", color="black", lw=1.8, ms=5,
                            label="long-term")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"VQ-VAE {size} ({VQVAE_PARAMS_M[size]:g}M)",
                             fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{sc}\n{ylabel}", fontsize=10,
                              color=SC_COLOR[sc])
            if i == len(SCS) - 1:
                ax.set_xlabel("NSP params (M)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return fig


def fig_per_family(tables, lt, horizons, metric, lt_key, ylabel, title, out,
                   floor=False, annotate=False):
    """One panel per (size, sc) model family; within each, a curve per lead
    time k (metric vs NSP params), the long-run climate as a dashed black
    reference, and (EMD only) the horizon-independent VQ-VAE floor. 3x3 grid,
    sc rows x VQ-size cols, y shared per row. This is the transpose of the
    per-horizon scaling_emd_k<k>.png figures: there a panel fixes k and varies
    sc/size; here a panel fixes the family and overlays every k."""
    fig, axes = plt.subplots(len(SCS), len(SIZES), figsize=(16, 13),
                             sharex=False, sharey="row",
                             constrained_layout=True)
    hcolors = plt.cm.viridis(np.linspace(0.12, 0.88, len(horizons)))
    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            ax = axes[i, j]
            for hi, k in enumerate(horizons):
                pts = sorted(
                    [(v["params_M"], v[metric], v.get("n_layer"))
                     for (s, c, a), v in tables[k].items()
                     if s == size and c == sc and v.get(metric) is not None],
                    key=lambda x: x[0])
                if not pts:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o",
                        color=hcolors[hi], lw=1.7, ms=5, label=f"k={k}")
                if annotate and hi == len(horizons) - 1:
                    for px, py, nl in pts:
                        if nl is not None:
                            ax.annotate(f"{nl}L", (px, py), fontsize=6,
                                        color="0.3", xytext=(0, 4),
                                        textcoords="offset points", ha="center")
            if lt is not None:
                lp = sorted(
                    [(v["params_M"], v[lt_key])
                     for (s, c, a), v in lt.items()
                     if s == size and c == sc
                     and v.get(lt_key) is not None and v.get("params_M")],
                    key=lambda x: x[0])
                if lp:
                    ax.plot([p[0] for p in lp], [p[1] for p in lp], "--s",
                            color="black", lw=1.6, ms=4, alpha=0.85,
                            label="long-run")
            if floor:
                vqs, seen = [], set()
                for k in horizons:
                    for (s, c, a), v in tables[k].items():
                        if (s == size and c == sc and (s, c, a) not in seen
                                and v.get("emd_vq") is not None):
                            seen.add((s, c, a)); vqs.append(v["emd_vq"])
                if vqs:
                    ax.axhline(np.median(vqs), color="0.5", ls=":", lw=1,
                               label="VQ floor")
            ax.set_xscale("log")
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{size}-{sc} ({VQVAE_PARAMS_M[size]:g}M VQ)",
                         fontsize=10, color=SC_COLOR[sc])
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(SCS) - 1:
                ax.set_xlabel("NSP params (M)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=13)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return fig


def write_csv(tables, horizons, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "sc", "arch", "params_M", "horizon", "best_T",
                    "n_T", "emd_nsp", "emd_vqvae", "tke_rse"])
        for k in horizons:
            for (size, sc, arch), v in sorted(tables[k].items()):
                w.writerow([size, sc, arch, v["params_M"], k, v["best_T"],
                            v["n_T"], v["emd"], v.get("emd_vq"), v.get("tke")])
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--projects", nargs="+",
                   default=[f"gust2-scaling-forecast-{s}" for s in SIZES],
                   help="wandb project(s) to read (default: the 3 per-tier "
                        "scaling-forecast projects)")
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--horizons", default="1,2,5,10",
                   help="comma-separated lead times to plot")
    p.add_argument("--longterm_csv", default="plots/scaling_tempopt/scaling_tempopt.csv",
                   help="long-term scaling CSV from plot_scaling_tempopt.py "
                        "(for the short-vs-long overlay); skipped if missing")
    p.add_argument("--output_dir", default="plots/scaling_forecast")
    p.add_argument("--no_annotate", action="store_true")
    args = p.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    tables = fetch_forecast(args.projects, args.entity, horizons)
    if not any(tables[k] for k in horizons):
        raise SystemExit("No cells with forecast metrics found.")
    lt = load_lt_csv(args.longterm_csv)

    # Textual readout: forecast EMD per cell per horizon.
    print(f"\n{'cell':28s} " + " ".join(f"k{k:>2}_emd" for k in horizons))
    cells = sorted({c for k in horizons for c in tables[k]},
                   key=lambda c: (c[0], c[1], label_to_params_M(c[2]) or 0))
    for (size, sc, arch) in cells:
        vals = []
        for k in horizons:
            v = tables[k].get((size, sc, arch))
            vals.append(f"{v['emd']:7.3f}" if v else f"{'--':>7}")
        print(f"  {size+'-'+sc+'-'+arch:26s} " + " ".join(vals))

    # (a) Per-horizon scaling figures (mirror the long-term layout).
    for k in horizons:
        if not tables[k]:
            continue
        fig_metric_vs_params(
            tables[k], "emd", f"pixel-EMD vs GT (best T, lead k={k})",
            f"Forecast scaling: lead-{k} pixel-EMD vs NSP size "
            "(temperature-optimal, posmask)",
            os.path.join(args.output_dir, f"scaling_emd_k{k}.png"),
            floor=True, annotate=not args.no_annotate)
        fig_metric_vs_params(
            tables[k], "tke", f"TKE-RSE (best T, lead k={k})",
            f"Forecast scaling: lead-{k} TKE spectral RSE vs NSP size",
            os.path.join(args.output_dir, f"scaling_tke_k{k}.png"),
            annotate=not args.no_annotate)

    # (b) Short-vs-long overlay.
    fig_overlay(
        tables, lt, horizons, "emd", "emd_nsp", "pixel-EMD",
        "Forecast (k=1,2,5,10) vs long-run climate (black): EMD scaling law",
        os.path.join(args.output_dir, "forecast_vs_longterm_emd.png"))

    # (c) Best-temperature divergence.
    fig_overlay(
        tables, lt, horizons, "best_T", "best_T", "best T",
        "Optimal temperature: short-horizon (k=1,2,5,10) vs long-run (black) "
        "— does the optimum diverge?",
        os.path.join(args.output_dir, "forecast_best_temperature.png"))

    # (d) Per-family view: one panel per (size, sc), one curve per lead k
    # (transpose of the per-horizon figures). Both metrics.
    fig_per_family(
        tables, lt, horizons, "emd", "emd_nsp", "pixel-EMD vs GT (best T)",
        "Forecast EMD scaling per model family "
        "(curve per lead k=1,2,5,10; long-run dashed; VQ floor dotted)",
        os.path.join(args.output_dir, "family_scaling_emd.png"),
        floor=True, annotate=not args.no_annotate)
    fig_per_family(
        tables, lt, horizons, "tke", "tke_rse", "TKE-RSE (best T)",
        "Forecast TKE-RSE scaling per model family "
        "(curve per lead k=1,2,5,10; long-run dashed)",
        os.path.join(args.output_dir, "family_scaling_tke.png"),
        annotate=not args.no_annotate)

    write_csv(tables, horizons,
              os.path.join(args.output_dir, "scaling_forecast.csv"))
    print(f"\nDone — figures in {args.output_dir}")


if __name__ == "__main__":
    main()
