"""measure_drift_sweep.py — sweep the per-scale token-DISTRIBUTION drift metric
(scale_distribution_drift.compute_drift) over a whole scaling-tempopt rollout tree.

Pure numpy + wandb (NO jax/GPU) — runs on a CPU node. For every
  {sweep_root}/<size>-<sc>-nsp-<arch>/T<tp>/rollout/rollout_tokens.npz
it computes per-scale JS / TV / signed entropy-gap (rollout vs its OWN gt_indices)
and logs a wandb run (project gust2-drift-<size>, group=<sc>,
name=<size>-<sc>-nsp-<arch>-T<tp>) plus a master CSV row. Pull/plot locally with
plot_scale_drift.py.

  python measure_drift_sweep.py --sweep_root .../experiments/scaling-tempopt-n128 \
      --sizes large medium small --csv .../drift_sweep.csv --wandb_dir .../wandb
  python measure_drift_sweep.py --sweep_root ./experiments/rollouts --glob '*/rollout_tokens.npz' --no_wandb
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

from scale_distribution_drift import compute_drift

CELL_RE = re.compile(r"(small|medium|large)-(sc\d+)-nsp-(s\d+)")
TEMP_RE = re.compile(r"T(\d+p\d+)")


def parse_path(p):
    """(size, sc, arch, T_float, T_str) from a rollout_tokens.npz path, or None."""
    cell = TEMP = None
    for part in p.split(os.sep):
        m = CELL_RE.search(part)
        if m:
            cell = m.groups()
        t = TEMP_RE.fullmatch(part)
        if t:
            TEMP = t.group(1)
    if not cell:
        return None
    T = float(TEMP.replace("p", ".")) if TEMP else None
    return cell[0], cell[1], cell[2], T, TEMP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", required=True)
    ap.add_argument("--glob", default="*/T*/rollout/rollout_tokens.npz")
    ap.add_argument("--sizes", nargs="+", default=["small", "medium", "large"])
    ap.add_argument("--scs", nargs="+", default=None, help="filter sc341/sc917/sc1941")
    ap.add_argument("--csv", default="drift_sweep.csv")
    ap.add_argument("--wandb_project_prefix", default="gust2-drift")
    ap.add_argument("--wandb_dir", default=None)
    ap.add_argument("--no_wandb", action="store_true")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.sweep_root, a.glob)))
    print(f"found {len(files)} rollout_tokens.npz under {a.sweep_root}")

    csv_rows = []
    for i, f in enumerate(files):
        meta = parse_path(f)
        if meta is None:
            print(f"  [skip:unparsed] {f}"); continue
        size, sc, arch, T, Tstr = meta
        if size not in a.sizes or (a.scs and sc not in a.scs):
            continue
        try:
            rows = compute_drift(f)            # ref = own gt_indices
        except Exception as e:
            print(f"  [skip:err] {f}: {e}"); continue
        # token-weighted aggregates over scales
        npos = np.array([r["n_pos"] for r in rows], float)
        js = np.array([r["js"] for r in rows])
        tv = np.array([r["tv"] for r in rows])
        js_w = float((js * npos).sum() / npos.sum())
        worst = max(rows, key=lambda r: r["js"])
        name = f"{size}-{sc}-nsp-{arch}" + (f"-T{Tstr}" if Tstr else "")
        print(f"[{i+1}/{len(files)}] {name}: js_w={js_w:.3f} "
              f"worst=s{worst['scale']}({worst['js']:.2f})")

        summary = {"drift/js_weighted": js_w, "drift/js_max": float(js.max()),
                   "drift/tv_weighted": float((tv * npos).sum() / npos.sum()),
                   "drift/worst_scale": worst["scale"]}
        for r in rows:
            s = r["scale"]
            summary[f"drift/js/s{s}"] = r["js"]
            summary[f"drift/tv/s{s}"] = r["tv"]
            summary[f"drift/dH/s{s}"] = r["dH"]
            csv_rows.append({"size": size, "sc": sc, "arch": arch, "T": T,
                             "scale": s, "n_pos": r["n_pos"], "js": r["js"],
                             "tv": r["tv"], "dH": r["dH"], "H_gt": r["H_ref"]})

        if not a.no_wandb:
            import wandb
            run = wandb.init(project=f"{a.wandb_project_prefix}-{size}",
                             name=name, group=sc, dir=a.wandb_dir,
                             config={"size": size, "sc": sc, "arch": arch, "T": T,
                                     "scales": [r["scale"] for r in rows]},
                             reinit=True)
            run.summary.update(summary)
            run.finish()

    if csv_rows:
        keys = list(csv_rows[0].keys())
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader(); w.writerows(csv_rows)
        print(f"\nwrote {len(csv_rows)} rows -> {a.csv}")


if __name__ == "__main__":
    main()
