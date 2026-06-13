"""plot_scaling_tempopt.py — temperature-optimal scaling-law figures.

Reads the temperature-bracket scaling sweep (gust2-scaling-tempopt-bridges,
runs named <size>-<sc>-nsp-<arch>-T<temp>), picks each cell's best temperature
by pixel-EMD vs GT, and plots the honest scaling metric vs NSP parameters —
the replacement for the old T=1.0 / posmask-off scaling figures whose
sc917/sc1941 numbers were cold-collapse artifacts.

Figures -> --output_dir (default plots/scaling_tempopt/):
  scaling_emd.png     pixel-EMD vs NSP params, panel per VQ size, line per sc,
                      dashed VQ-VAE floor per config  (THE scaling law)
  scaling_tke.png     TKE-RSE vs NSP params (same layout) at each cell's best T
  best_temperature.png  best T vs NSP params — does the optimum drift with
                      model size / VQ tier? (justifies / refutes T transfer)
  scaling_tempopt.csv   per-cell best T + metrics

Reads the 3 per-VQ-tier projects gust2-scaling-tempopt-{small,medium,large}
by default. Run locally after the sweep's analysis runs have logged:
  ~/llm/bin/python plot_scaling_tempopt.py
  ~/llm/bin/python plot_scaling_tempopt.py --projects gust2-analysis-bridges-scaling  # smoke test on old single-T data
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "bigpseud-ucsc"
SIZES = ["small", "medium", "large"]
SCS = ["sc341", "sc917", "sc1941"]
# VQ-VAE params per tier (matches plot_scaling.py / CLAUDE.md model table).
VQVAE_PARAMS_M = {"small": 31.0, "medium": 57.0, "large": 109.0}
SC_COLOR = {"sc341": "C0", "sc917": "C1", "sc1941": "C2"}


def label_to_params_M(label):
    """NSP params in millions from the arch label (sNN -> NN). Canonical
    convention, same as plot_scaling_all.py:46."""
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def temp_from_token(tok):
    """'T1p6' -> 1.6 ; returns None if not a temperature token."""
    m = re.match(r"T(\d+)p(\d+)$", tok)
    return float(f"{m.group(1)}.{m.group(2)}") if m else None


def parse_run_name(name):
    """<size>-<sc>-nsp-<arch>[-T<temp>] -> (size, sc, arch, temp) or None.
    Skips -eval runs. temp defaults to 1.0 for single-T (old) projects."""
    parts = name.split("-")
    if len(parts) < 4 or parts[2] != "nsp":
        return None
    if "eval" in parts:
        return None
    size, sc, arch = parts[0], parts[1], parts[3]
    temp = 1.0
    if len(parts) >= 5:
        t = temp_from_token(parts[4])
        if t is None:
            return None        # some other suffix we don't handle
        temp = t
    return size, sc, arch, temp


def fetch(projects, entity):
    """Return per-cell best-T record:
    table[(size, sc, arch)] = {params_M, best_T, emd, tke, ens, emd_vq, n_T}
    best T chosen by min pixel-EMD (emd/nsp). Reads across one or more wandb
    projects (the per-VQ-tier scaling-tempopt projects)."""
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

    # collect per (cell, T)
    per_cell = defaultdict(list)            # (size,sc,arch) -> [(T, metrics)]
    for r in runs:
        if r.state != "finished":
            continue
        parsed = parse_run_name(r.name)
        if parsed is None:
            continue
        size, sc, arch, temp = parsed
        emd = r.summary.get("emd/nsp")
        if emd is None:
            continue
        per_cell[(size, sc, arch)].append((temp, {
            "emd": float(emd),
            "tke": r.summary.get("tke_rse/nsp"),
            "ens": r.summary.get("enstrophy_rse/nsp"),
            "emd_vq": r.summary.get("emd/vqvae"),
        }))

    table = {}
    for cell, entries in per_cell.items():
        size, sc, arch = cell
        pm = label_to_params_M(arch)
        if pm is None:
            continue
        best_T, best = min(entries, key=lambda e: e[1]["emd"])
        table[cell] = {
            "params_M": pm, "best_T": best_T, "n_T": len(entries), **best,
        }
    print(f"  -> {len(table)} cells with metrics")
    return table


def _panel_lines(ax, table, size, metric, floor=False):
    """One VQ-size panel: metric vs NSP params, a line per sc-config."""
    for sc in SCS:
        pts = sorted(
            [(v["params_M"], v[metric], v.get("emd_vq"))
             for (s, c, a), v in table.items()
             if s == size and c == sc and v.get(metric) is not None],
            key=lambda x: x[0])
        if not pts:
            continue
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        ax.plot(x, y, "-o", color=SC_COLOR[sc], lw=1.8, ms=5, label=sc)
        if floor:
            vq = [p[2] for p in pts if p[2] is not None]
            if vq:
                ax.axhline(np.median(vq), color=SC_COLOR[sc], ls=":", lw=1,
                           alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("NSP params (M)")
    ax.set_title(f"VQ-VAE {size} ({VQVAE_PARAMS_M[size]:g}M)", fontsize=11)
    ax.grid(True, alpha=0.3)


def fig_metric_vs_params(table, metric, ylabel, title, out, floor=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True,
                             constrained_layout=True)
    for ax, size in zip(axes, SIZES):
        _panel_lines(ax, table, size, metric, floor=floor)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(title="token budget", fontsize=9)
    if floor:
        axes[-1].plot([], [], ls=":", color="0.4", label="VQ-VAE floor")
        axes[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle(title)
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return fig


def fig_best_temperature(table, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True,
                             constrained_layout=True)
    for ax, size in zip(axes, SIZES):
        for sc in SCS:
            pts = sorted(
                [(v["params_M"], v["best_T"])
                 for (s, c, a), v in table.items() if s == size and c == sc],
                key=lambda x: x[0])
            if not pts:
                continue
            x = [p[0] for p in pts]
            y = [p[1] for p in pts]
            ax.plot(x, y, "-o", color=SC_COLOR[sc], lw=1.8, ms=5, label=sc)
        ax.set_xscale("log")
        ax.set_xlabel("NSP params (M)")
        ax.set_title(f"VQ-VAE {size}", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("best temperature (min pixel-EMD)")
    axes[0].legend(title="token budget", fontsize=9)
    fig.suptitle("Optimal temperature vs model size — drift check "
                 "(flat per config => T transfers)")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")
    return fig


def write_csv(table, out):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "sc", "arch", "params_M", "best_T", "n_T",
                    "emd_nsp", "emd_vqvae", "tke_rse", "enstrophy_rse"])
        for (size, sc, arch), v in sorted(table.items()):
            w.writerow([size, sc, arch, v["params_M"], v["best_T"], v["n_T"],
                        v["emd"], v.get("emd_vq"), v.get("tke"), v.get("ens")])
    print(f"saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--projects", nargs="+",
                   default=[f"gust2-scaling-tempopt-{s}" for s in SIZES],
                   help="wandb project(s) to read (default: the 3 per-tier "
                        "scaling-tempopt projects)")
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--output_dir", default="plots/scaling_tempopt")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    table = fetch(args.projects, args.entity)
    if not table:
        raise SystemExit("No cells with metrics found.")

    # quick textual readout (best T per cell)
    print(f"\n{'cell':28s} {'params':>7} {'bestT':>6} {'emd':>7} {'tke':>6}")
    for (size, sc, arch), v in sorted(
            table.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[1]["params_M"])):
        tke = v.get("tke")
        print(f"  {size+'-'+sc+'-'+arch:26s} {v['params_M']:6.0f}M "
              f"{v['best_T']:6.2f} {v['emd']:7.3f} "
              f"{tke if tke is None else round(tke,3):>6}")

    fig_metric_vs_params(
        table, "emd", "pixel-EMD vs GT (best T)",
        "Scaling law: long-rollout pixel-EMD vs NSP size (temperature-optimal, posmask)",
        os.path.join(args.output_dir, "scaling_emd.png"), floor=True)
    fig_metric_vs_params(
        table, "tke", "TKE-RSE (best T)",
        "Scaling law: TKE spectral RSE vs NSP size (temperature-optimal)",
        os.path.join(args.output_dir, "scaling_tke.png"))
    fig_best_temperature(
        table, os.path.join(args.output_dir, "best_temperature.png"))
    write_csv(table, os.path.join(args.output_dir, "scaling_tempopt.csv"))
    print(f"\nDone — figures in {args.output_dir}")


if __name__ == "__main__":
    main()
