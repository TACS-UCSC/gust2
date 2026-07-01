"""plot_snapshot_tempgrid.py — temperature × lead-time snapshot grid from the
already-logged wandb snapshots (no Bridges round-trip, no token decode).

analyze_rollout.py logs a GT | VQ-VAE | NSP triptych per (cell, temp) run at
~11 lead times (snapshot/t1..t1500). This script downloads those triptychs for
one cell across its temperature bracket, auto-crops the three panels (by white-
gutter detection — robust to constrained_layout), and assembles an NSP-per-cell
grid (rows = temperature, cols = lead time) with GT + VQ-VAE reference rows and
each NSP row annotated with that run's collapse_rate. The figure shows the
temperature collapse Pareto: cold drifts to large-scale diffusive bands, hot
collapses too, only the mid sweet-spot sustains GT-like turbulence.

NOTE: snapshots are trajectory-0 only (the analyzer's representative). They are
single-trajectory; collapse is an ensemble property, so each row is annotated
with the full-ensemble collapse_rate to ground the single-frame view. For the
collapsed-vs-survivor ENSEMBLE spread, use multitraj_snapshot_grid.py on Bridges.

Usage:
  ~/llm/bin/python plot_snapshot_tempgrid.py \
      --project gust2-scaling-tempopt-n128-medium \
      --cell medium-sc917-nsp-s22 \
      --temps 1.2 1.4 1.6 1.8 2.0 --best_t 1.6 \
      --times 10 100 1000 1500
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import wandb

ENTITY = "bigpseud-ucsc"


def _tp(temp):
    """1.6 -> 'T1p6' (wandb run-name temperature token)."""
    return "T" + f"{temp:.1f}".replace(".", "p")


def _nonwhite(a):
    return (a < 240).any(axis=2)


def crop_panels(path):
    """Return {gt,vqvae,nsp} -> cropped RGB field (titles/colorbar excluded).

    The triptych is 3 equal imshow panels + a thin colorbar. Panel x-bands are
    the wide (>100px) columns with high non-white fraction; the colorbar strip
    is too narrow to survive the width filter. Within each band the dense field
    rows (non-white fraction > 0.3) exclude the sparse title text.
    """
    a = np.asarray(Image.open(path).convert("RGB"))
    col = _nonwhite(a).mean(axis=0) > 0.15
    bands, i, W = [], 0, len(col)
    while i < W:
        if col[i]:
            j = i
            while j < W and col[j]:
                j += 1
            if j - i > 100:
                bands.append((i, j))
            i = j
        else:
            i += 1
    out = {}
    for name, (x0, x1) in zip(["gt", "vqvae", "nsp"], bands[:3]):
        row = _nonwhite(a[:, x0:x1]).mean(axis=1) > 0.3
        ys = np.where(row)[0]
        out[name] = a[ys.min():ys.max() + 1, x0:x1]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", required=True, help="wandb project (one VQ tier)")
    p.add_argument("--cell", required=True, help="run stem, e.g. medium-sc917-nsp-s22")
    p.add_argument("--temps", type=float, nargs="+",
                   default=[1.2, 1.4, 1.6, 1.8, 2.0])
    p.add_argument("--best_t", type=float, default=None,
                   help="temperature to highlight (e.g. the best-EMD T)")
    p.add_argument("--times", type=int, nargs="+", default=[10, 100, 1000, 1500])
    p.add_argument("--entity", default=ENTITY)
    p.add_argument("--cache", default="/tmp/snap_tempgrid")
    p.add_argument("--output_dir", default="plots/snapshots_n128_tempcollapse")
    args = p.parse_args()

    os.makedirs(args.cache, exist_ok=True)
    api = wandb.Api()
    runs = {r.name: r for r in api.runs(f"{args.entity}/{args.project}")}

    cr, emd = {}, {}
    for tv in args.temps:
        s = runs[f"{args.cell}-{_tp(tv)}"].summary
        cr[tv], emd[tv] = s.get("collapse_rate"), s.get("emd/nsp")

    def triptych(tv, t):
        local = f"{args.cache}/{args.cell}-{_tp(tv)}-t{t}.png"
        if not os.path.exists(local):
            r = runs[f"{args.cell}-{_tp(tv)}"]
            f = next(f for f in r.files() if f"/snapshot/t{t}_" in f.name)
            dl = f.download(root=args.cache, replace=True)
            Image.open(dl.name).convert("RGB").save(local)
        return local

    ref_tv = args.best_t if args.best_t in args.temps else args.temps[len(args.temps) // 2]
    rows = [("GT", "gt", None), ("VQ-VAE", "vqvae", None)] + \
           [(f"T={tv:.1f}", "nsp", tv) for tv in args.temps]
    ncol, nrow = len(args.times), len(rows)
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.55),
                             squeeze=False)
    for ri, (label, which, tv) in enumerate(rows):
        src_tv = ref_tv if which != "nsp" else tv
        for ci, t in enumerate(args.times):
            ax = axes[ri][ci]
            ax.imshow(crop_panels(triptych(src_tv, t))[which])
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t = {t}", fontsize=13)
        is_best = which == "nsp" and args.best_t is not None and abs(tv - args.best_t) < 1e-9
        if which == "nsp":
            crv = cr.get(tv)
            lab = label + (f"\ncollapse {crv*100:.0f}%" if crv is not None else "")
            if is_best:
                lab += "\n(best-T)"
            color = "tab:red" if is_best else ("firebrick" if (crv or 0) >= 0.5 else "black")
        else:
            lab = label + ("\n(quant. floor)" if which == "vqvae" else "\n(truth)")
            color = "tab:blue"
        axes[ri][0].set_ylabel(lab, fontsize=11, color=color,
                               fontweight="bold" if (is_best or which != "nsp") else "normal")
        if is_best:
            for ci in range(ncol):
                for sp in axes[ri][ci].spines.values():
                    sp.set_color("tab:red"); sp.set_linewidth(2.5)

    sub = ""
    if cr.get(args.temps[0]) is not None and args.best_t is not None:
        sub = (f"\ncold T={args.temps[0]:.1f} (collapse {cr[args.temps[0]]*100:.0f}%) "
               f"→ diffusive banding   |   best-T={args.best_t:.1f} "
               f"(collapse {cr[args.best_t]*100:.0f}%) → stays turbulent")
    fig.suptitle(f"{args.cell}  ·  N=128 traj0  ·  temperature × lead-time{sub}",
                 fontsize=13.5, y=1.004)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(args.output_dir, exist_ok=True)
    dest = f"{args.output_dir}/{args.cell}_temp_collapse.png"
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    print(f"saved {dest}")
    print("collapse_rate per T:", {tv: cr[tv] for tv in args.temps})
    print("emd per T:", {tv: (round(emd[tv], 3) if emd[tv] is not None else None)
                          for tv in args.temps})


if __name__ == "__main__":
    main()
