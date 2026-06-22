"""plot_inference_samplers.py — anchored samplers vs the swept-T yardstick.

The payoff figure for paper section B. Reads the inference-sampler sweep
(gust2-inference-samplers-{small,medium,large}, runs <size>-<sc>-nsp-<arch>-<arm>)
and joins each arm's long-rollout pixel-EMD against the per-cell swept-temperature
optimum (plots/scaling_tempopt_n128/scaling_tempopt.csv) and the VQ-VAE EMD floor.

The claim under test: the parameter-free entropy-target homeostat (arm `etpool`)
lands NON-COLLAPSED (EMD < 2x VQ-VAE floor = the analyze_rollout collapse
threshold) and within ballpark of the cell's best-T EMD — with NO per-config
tuning. The entropy-shrinking controls (minp, invedt) and truncation-only tophabs
are expected to collapse. Ranked by EMD (NEVER TKE RSE — the over-diffusive mode
games it).

Figures -> --output_dir (default plots/inference_samplers/):
  samplers_emd_grid.png   3x3 (size x sc): EMD vs NSP params, line per arm,
                          dashed best-T yardstick, dotted VQ-VAE floor.
  samplers_headline.png   homeostat (etpool) vs best-T vs floor, panel per size.
  inference_samplers.csv  per (cell, arm): emd, best_T_emd, gap, ratio_to_floor,
                          collapsed?

Run locally after the sweep's analysis runs have logged:
  ~/llm/bin/python plot_inference_samplers.py
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "bigpseud-ucsc"
SIZES = ["small", "medium", "large"]
SCS = ["sc341", "sc917", "sc1941"]

ARMS = ["etmodel", "etpool", "etpos", "tophabs", "typical", "minp", "invedt"]
ARM_LABEL = {
    "etmodel": "entropy-target (model H)",
    "etpool": "entropy-target (pooled)",
    "etpos": "entropy-target (per-pos)",
    "tophabs": "Top-H abs (trunc-only)",
    "typical": "locally typical",
    "minp": "min-p (control)",
    "invedt": "inverted-EDT (control)",
}
# (color, marker, linestyle)
ARM_STYLE = {
    "etmodel": ("C6", "P", "-"),
    "etpool": ("C2", "o", "-"),
    "etpos": ("C0", "s", "-"),
    "tophabs": ("C4", "^", "--"),
    "typical": ("C1", "D", "-"),
    "minp": ("C3", "v", ":"),
    "invedt": ("C5", "x", ":"),
}
# etmodel (model conditional-entropy anchor) is the principled headline — the
# tokenizer-marginal anchor (etpool) over-warms cold-optimal sc341.
HEADLINE_ARM = "etmodel"


def label_to_params_M(label):
    """'s139' -> 139.0 (canonical convention, matches plot_scaling_tempopt.py)."""
    import re
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def parse_run_name(name):
    """<size>-<sc>-nsp-<arch>-<arm> -> (size, sc, arch, arm) or None."""
    parts = name.split("-")
    if len(parts) < 5 or parts[2] != "nsp" or "eval" in parts:
        return None
    size, sc, arch, arm = parts[0], parts[1], parts[3], parts[4]
    if arm not in ARMS:
        return None
    return size, sc, arch, arm


def fetch(projects, entity):
    """table[(size, sc, arch)][arm] = {emd, tke, emd_vq, collapse_rate}."""
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

    table = defaultdict(dict)
    for r in runs:
        if r.state != "finished":
            continue
        parsed = parse_run_name(r.name)
        if parsed is None:
            continue
        size, sc, arch, arm = parsed
        emd = r.summary.get("emd/nsp")
        if emd is None:
            continue
        table[(size, sc, arch)][arm] = {
            "emd": float(emd),
            "tke": r.summary.get("tke_rse/nsp"),
            "emd_vq": r.summary.get("emd/vqvae"),
            "collapse_rate": r.summary.get("collapse_rate"),
        }
    n_arms = sum(len(v) for v in table.values())
    print(f"  -> {len(table)} cells, {n_arms} (cell,arm) records")
    return table


def load_yardstick(path):
    """CSV -> ystk[(size, sc, arch)] = {best_T, emd_nsp, emd_vqvae, tke_rse}."""
    if not os.path.exists(path):
        print(f"[warn] yardstick CSV not found: {path} (no best-T overlay)")
        return {}
    ystk = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["size"], row["sc"], row["arch"])
            def fnum(k):
                v = row.get(k, "")
                return float(v) if v not in ("", "None", None) else None
            ystk[key] = {
                "best_T": fnum("best_T"),
                "emd_nsp": fnum("emd_nsp"),
                "emd_vqvae": fnum("emd_vqvae"),
                "tke_rse": fnum("tke_rse"),
            }
    print(f"  yardstick: {len(ystk)} cells from {path}")
    return ystk


def fig_emd_grid(table, ystk, out):
    """3x3 (size rows x sc cols): EMD vs params, line per arm + yardstick + floor."""
    fig, axes = plt.subplots(len(SIZES), len(SCS), figsize=(15, 12),
                             sharex=False, constrained_layout=True)
    for ri, size in enumerate(SIZES):
        for ci, sc in enumerate(SCS):
            ax = axes[ri][ci]
            cells = sorted(
                [(arch, v) for (s, c, arch), v in table.items()
                 if s == size and c == sc],
                key=lambda t: (label_to_params_M(t[0]) or 0))
            if not cells:
                ax.set_visible(False)
                continue
            xs = [label_to_params_M(a) for a, _ in cells]
            for arm in ARMS:
                color, marker, ls = ARM_STYLE[arm]
                pts = [(label_to_params_M(a), v[arm]["emd"])
                       for a, v in cells if arm in v]
                if not pts:
                    continue
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts],
                        marker=marker, ls=ls, color=color, lw=1.6, ms=5,
                        label=ARM_LABEL[arm])
            # swept-T yardstick + VQ floor from the CSV
            yb = [(label_to_params_M(a), ystk.get((size, sc, a), {}).get("emd_nsp"))
                  for a, _ in cells]
            yb = [(x, y) for x, y in yb if y is not None]
            if yb:
                yb.sort()
                ax.plot([p[0] for p in yb], [p[1] for p in yb], "k--", lw=1.5,
                        label="best-T (sweep)")
            fl = [ystk.get((size, sc, a), {}).get("emd_vqvae") for a, _ in cells]
            fl = [f for f in fl if f is not None]
            if fl:
                ax.axhline(np.median(fl), color="0.4", ls=":", lw=1.2,
                           label="VQ-VAE floor")
                ax.axhline(2 * np.median(fl), color="0.7", ls=":", lw=1.0,
                           label="collapse thr (2x floor)")
            ax.set_xscale("log")
            ax.set_title(f"{size} / {sc}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if ci == 0:
                ax.set_ylabel("pixel-EMD vs GT")
            if ri == len(SIZES) - 1:
                ax.set_xlabel("NSP params (M)")
    # one shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Anchored samplers vs swept-T optimum — long-rollout pixel-EMD "
                 "(below collapse threshold = non-collapsed)", fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {out}")


def fig_headline(table, ystk, out):
    """homeostat (etpool) EMD vs best-T vs floor — one panel per VQ size."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(15, 4.6), sharey=False,
                             constrained_layout=True)
    for ax, size in zip(axes, SIZES):
        for sc in SCS:
            color = {"sc341": "C0", "sc917": "C1", "sc1941": "C2"}[sc]
            cells = sorted(
                [(arch, v) for (s, c, arch), v in table.items()
                 if s == size and c == sc and HEADLINE_ARM in v],
                key=lambda t: (label_to_params_M(t[0]) or 0))
            if not cells:
                continue
            x = [label_to_params_M(a) for a, _ in cells]
            y = [v[HEADLINE_ARM]["emd"] for _, v in cells]
            ax.plot(x, y, "-o", color=color, lw=1.8, ms=5, label=f"{sc} homeostat")
            yb = [(label_to_params_M(a), ystk.get((size, sc, a), {}).get("emd_nsp"))
                  for a, _ in cells]
            yb = [(xx, yy) for xx, yy in yb if yy is not None]
            if yb:
                yb.sort()
                ax.plot([p[0] for p in yb], [p[1] for p in yb], "--",
                        color=color, lw=1.2, alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("NSP params (M)")
        ax.set_title(f"VQ-VAE {size}", fontsize=11)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("pixel-EMD vs GT")
    axes[0].plot([], [], "k-o", label="homeostat (etpool)")
    axes[0].plot([], [], "k--", label="best-T (sweep)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Headline: parameter-free entropy-target homeostat vs swept-T "
                 "optimum (solid vs dashed = within ballpark)")
    fig.savefig(out, dpi=130)
    print(f"saved {out}")


def write_summary(table, ystk, out):
    rows = []
    for (size, sc, arch), arms in sorted(table.items()):
        y = ystk.get((size, sc, arch), {})
        best_emd = y.get("emd_nsp")
        floor = y.get("emd_vqvae")
        thr = 2 * floor if floor is not None else None
        for arm in ARMS:
            if arm not in arms:
                continue
            emd = arms[arm]["emd"]
            gap = (emd - best_emd) if best_emd is not None else None
            ratio = (emd / floor) if floor else None
            collapsed = (emd > thr) if thr is not None else None
            rows.append({
                "size": size, "sc": sc, "arch": arch, "arm": arm,
                "emd": emd, "best_T_emd": best_emd, "gap": gap,
                "floor": floor, "ratio_to_floor": ratio,
                "collapse_thr": thr, "collapsed": collapsed,
                "collapse_rate": arms[arm].get("collapse_rate"),
            })
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["size", "sc", "arch", "arm", "emd"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved {out}")

    # textual readout: the headline arm per cell
    print(f"\n{'cell':30s} {'arm':9s} {'emd':>7} {'best-T':>7} "
          f"{'gap':>7} {'x_floor':>7} {'collapsed':>9}")
    for r in rows:
        if r["arm"] != HEADLINE_ARM:
            continue
        cell = f"{r['size']}-{r['sc']}-{r['arch']}"
        def s(x, f="{:.3f}"):
            return "  -  " if x is None else f.format(x)
        print(f"  {cell:28s} {r['arm']:9s} {s(r['emd'])} {s(r['best_T_emd'])} "
              f"{s(r['gap'], '{:+.3f}')} {s(r['ratio_to_floor'], '{:.2f}')} "
              f"{str(r['collapsed']):>9}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--projects", nargs="+",
                   default=[f"gust2-inference-samplers-{s}" for s in SIZES])
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--yardstick", default="plots/scaling_tempopt_n128/scaling_tempopt.csv")
    p.add_argument("--output_dir", default="plots/inference_samplers")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    table = fetch(args.projects, args.entity)
    if not table:
        raise SystemExit("No (cell, arm) records found in the sampler projects.")
    ystk = load_yardstick(args.yardstick)

    fig_emd_grid(table, ystk, os.path.join(args.output_dir, "samplers_emd_grid.png"))
    fig_headline(table, ystk, os.path.join(args.output_dir, "samplers_headline.png"))
    write_summary(table, ystk, os.path.join(args.output_dir, "inference_samplers.csv"))
    print(f"\nDone — figures + CSV in {args.output_dir}")


if __name__ == "__main__":
    main()
