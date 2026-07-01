"""plot_per_scale_temp.py — per-scale-temperature replication report.

Reads the per-scale-temp arms (fineT*/coarseT*/calibPS/calibPC) from
gust2-inference-samplers-{small,medium,large} and asks the replication question:
does fine-scale heating generalize across the sc1941 column, and where does
coarse-heating / calibration land?

Judged by EMD BAND + the swept best-T yardstick — NOT collapse_rate (whose
2x-VQ-floor bar over-flags the low-floor sc1941 configs; see
feedback_judge_by_emd_band). Bands (EMD vs GT): <=1.2x best-T = matches/beats
the hand-tuned sweep; >1.0 abs = exploded (off-manifold). The snapshot remains
the final arbiter — build the snapshot montage separately (plot_snapshot_tempgrid
or the arm-snapshot script).

  ~/llm/bin/python plot_per_scale_temp.py                       # sc1941, all tiers
  ~/llm/bin/python plot_per_scale_temp.py --sc sc917
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "bigpseud-ucsc"
SIZES = ["small", "medium", "large"]
EXPLODED = 1.0           # EMD above this = off-manifold explosion (see snapshots)


def label_to_params_M(label):
    m = re.match(r"s(\d+)$", label)
    return float(m.group(1)) if m else None


def arm_style(arm):
    """(color, marker) by arm family; None if not a per-scale arm."""
    if re.match(r"fineT\d", arm):
        return ("tab:green", "o")
    if re.match(r"coarseT\d", arm):
        return ("tab:red", "x")
    if arm in ("calibPS", "calibPC"):
        return ("tab:blue", "s")
    return None


def fetch(projects, sc):
    """table[(size, arch)][arm] = {emd, emd_vq}. Only per-scale arms, only `sc`."""
    api = wandb.Api()
    table = defaultdict(dict)
    for project in projects:
        try:
            runs = list(api.runs(f"{ENTITY}/{project}"))
        except ValueError:
            print(f"[warn] {project} not found"); continue
        for r in runs:
            if r.state != "finished":
                continue
            p = r.name.split("-")
            if len(p) < 5 or p[2] != "nsp" or "eval" in p:
                continue
            size, rsc, arch, arm = p[0], p[1], p[3], p[4]
            if rsc != sc or arm_style(arm) is None:
                continue
            emd = r.summary.get("emd/nsp")
            if emd is None:
                continue
            table[(size, arch)][arm] = {
                "emd": float(emd), "emd_vq": r.summary.get("emd/vqvae"),
            }
    return table


def load_yardstick(path, sc):
    """(size, arch) -> best-T EMD from the swept-T CSV."""
    y = {}
    if not os.path.exists(path):
        print(f"[warn] yardstick {path} missing"); return y
    for row in csv.DictReader(open(path)):
        if row["sc"] != sc:
            continue
        try:
            y[(row["size"], row["arch"])] = {
                "emd": float(row["emd_nsp"]), "best_T": row.get("best_T"),
                "floor": float(row["emd_vqvae"]) if row.get("emd_vqvae") not in ("", "None") else None,
            }
        except (ValueError, KeyError):
            pass
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects", nargs="+",
                    default=[f"gust2-inference-samplers-{s}" for s in SIZES])
    ap.add_argument("--sc", default="sc1941")
    ap.add_argument("--yardstick", default="plots/scaling_tempopt_n128/scaling_tempopt.csv")
    ap.add_argument("--output_dir", default="plots/per_scale_temp")
    args = ap.parse_args()

    table = fetch(args.projects, args.sc)
    ystk = load_yardstick(args.yardstick, args.sc)
    arms = sorted({a for cell in table.values() for a in cell},
                  key=lambda a: (a[:5], a))
    print(f"{args.sc}: {len(table)} cells with per-scale arms; arms={arms}")

    # ---- figure: per-tier EMD vs params, line per arm, yardstick + floor ----
    fig, axes = plt.subplots(1, len(SIZES), figsize=(16, 5), sharey=True)
    for ci, size in enumerate(SIZES):
        ax = axes[ci]
        cells = sorted([(arch, table[(size, arch)]) for (s, arch) in table if s == size],
                       key=lambda x: label_to_params_M(x[0]))
        ax.axhspan(EXPLODED, 1e2, color="0.9", zorder=0)
        ax.text(0.02, 0.96, "exploded (EMD>1)", transform=ax.transAxes,
                fontsize=8, color="0.4", va="top")
        for arm in arms:
            col, mk = arm_style(arm)
            xs, ys = [], []
            for arch, d in cells:
                if arm in d:
                    xs.append(label_to_params_M(arch)); ys.append(d[arm]["emd"])
            if xs:
                ax.plot(xs, ys, marker=mk, color=col, label=arm, lw=1.6, ms=7, alpha=0.85)
        # yardstick best-T + floor
        yx = sorted([(label_to_params_M(a), ystk[(size, a)]) for (s, a) in ystk if s == size])
        if yx:
            px = [p for p, _ in yx]
            ax.plot(px, [v["emd"] for _, v in yx], "k--", lw=1.5, label="swept best-T", zorder=5)
            fl = [v["floor"] for _, v in yx if v["floor"]]
            if fl:
                ax.plot([p for p, v in yx if v["floor"]], fl, ":", color="purple", lw=1.3, label="VQ floor")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(f"{size} VQ"); ax.set_xlabel("NSP params (M)")
        if ci == 0:
            ax.set_ylabel("pixel-EMD vs GT (log)")
        ax.grid(alpha=0.3, which="both")
        if ci == len(SIZES) - 1:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Per-scale-temp replication ({args.sc}): does FINE-scale heating beat collapse across the column?\n"
                 f"green=fine-heat · red=coarse-heat (control) · blue=calibration · judged by EMD band (NOT collapse_rate)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    dest = f"{args.output_dir}/per_scale_temp_{args.sc}.png"
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    print(f"saved {dest}")

    # ---- text summary: best per-scale arm vs best-T, fine-vs-coarse control ----
    print(f"\n{'cell':22s} {'bestT':>6} {'best per-scale arm':>20} {'EMD':>7} {'vs bestT':>9} {'fine<coarse?':>12}")
    for (size, arch) in sorted(table, key=lambda k: (SIZES.index(k[0]), label_to_params_M(k[1]))):
        d = table[(size, arch)]
        ba = min(d, key=lambda a: d[a]["emd"])
        y = ystk.get((size, arch), {})
        bt = y.get("emd")
        ratio = f"{d[ba]['emd']/bt:.2f}x" if bt else "  --"
        fineE = min((d[a]["emd"] for a in d if a.startswith("fine")), default=None)
        coarseE = min((d[a]["emd"] for a in d if a.startswith("coarse")), default=None)
        fc = (f"{'YES' if fineE < coarseE else 'no':>4} ({fineE:.2f}/{coarseE:.2f})"
              if (fineE is not None and coarseE is not None) else "  --")
        print(f"{size+'-'+args.sc+'-'+arch:22s} {str(y.get('best_T','--')):>6} "
              f"{ba:>20} {d[ba]['emd']:7.3f} {ratio:>9} {fc:>12}")


if __name__ == "__main__":
    main()
