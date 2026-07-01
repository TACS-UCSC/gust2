"""Pull snapshot / spectrum / histogram PNGs from
`gust2-analysis-bridges-scaling` and assemble per-(sc, VQ size, kind) grids
for the Beamer scaling-sweep report.

For each (sc-config, VQ size, kind):
  snapshots_{sc}_{size}_{kind}.png    5 NSP rows × 5 timestep cols
  spectra_{sc}_{size}_{kind}.png      5 NSP rows × 3 metric cols  (TKE / Ens / Hist)

Also writes per-(sc, kind) metrics CSVs and booktabs `.tex` snippets to be
\\input{}'ed by the report.

Pattern cribbed from plot_sc341_histograms.py:33-91. Cache-aware: re-runs
skip already-downloaded PNGs.

Usage:
    ~/llm/bin/python plot_scaling_report_grids.py                  # full sweep
    ~/llm/bin/python plot_scaling_report_grids.py \\
        --sc sc341 --size small --kind multistep                   # smoke test
"""

import argparse
import csv
import os
import re

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import wandb

ENTITY  = "bigpseud-ucsc"
PROJECT = "gust2-analysis-bridges-scaling"

SCS   = ["sc341", "sc917"]
SIZES = ["small", "medium", "large"]
ARCHS = {
    "sc341": ["s06", "s09", "s13", "s18", "s24"],
    "sc917": ["s13", "s22", "s34", "s50", "s74"],
}
KINDS = ["multistep", "singlestep"]   # base / -eval runs
SNAPSHOT_TS = {
    "multistep":  [1, 100, 500, 1000, 1500],
    "singlestep": [1, 100, 500, 1000, 1999],
}
SPECTRA_KEYS = [
    ("tke_spectrum",       "TKE spectrum"),
    ("enstrophy_spectrum", "Enstrophy spectrum"),
    ("pixel_histogram",    "Pixel histogram"),
]

# Real trainable-param counts (M) per Test 9/10 of the report.
# sX label = projected ~X M; tabulated values are authoritative.
PARAMS_M = {
    "sc341":  {"s06":  6.04, "s09":  9.11, "s13": 12.65,
               "s18": 17.96, "s24": 24.04},
    "sc917":  {"s13": 13.32, "s22": 22.18, "s34": 34.14,
               "s50": 50.07, "s74": 73.68},
    "sc1941": {"s31": 31.17, "s48": 47.64, "s73": 73.11,
               "s113": 113.43, "s139": 138.74},
}


# ----------------------------------------------------------------------
# Wandb fetch
# ----------------------------------------------------------------------

def run_name(sc, size, arch, kind):
    base = f"{size}-{sc}-nsp-{arch}"
    return f"{base}-eval" if kind == "singlestep" else base


def get_run(api, name):
    """Return the wandb Run with exact display name, or None."""
    runs = list(api.runs(f"{ENTITY}/{PROJECT}",
                         filters={"display_name": name}))
    runs = [r for r in runs if r.name == name]
    return runs[0] if runs else None


def download_match(run, substr, cache_dir):
    """Download the first file in `run` whose name contains `substr` and
    ends in .png. Returns local path, or None if no match."""
    os.makedirs(cache_dir, exist_ok=True)
    matches = [f for f in run.files()
               if substr in f.name and f.name.endswith(".png")]
    if not matches:
        return None
    f = matches[0]
    out = os.path.join(cache_dir, f.name)
    if not os.path.exists(out):
        f.download(root=cache_dir, replace=True)
    return out


def fetch_snapshot(run, t, cache_dir):
    # Trailing "_" anchors so t1 doesn't match t10/t100/t1000/t1500/t1999.
    return download_match(run, f"snapshot/t{t}_", cache_dir)


def fetch_spectrum(run, key, cache_dir):
    return download_match(run, f"{key}_0_", cache_dir)


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------

def _missing_cell(ax, msg="(missing)"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, color="0.55", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def render_snapshot_grid(sc, size, kind, paths, out_path):
    """paths[(arch, t)] -> local PNG path or None."""
    archs = ARCHS[sc]
    ts = SNAPSHOT_TS[kind]
    nrows, ncols = len(archs), len(ts)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 1.7 * nrows),
                             squeeze=False)
    for i, arch in enumerate(archs):
        for j, t in enumerate(ts):
            ax = axes[i, j]
            p = paths.get((arch, t))
            if p is None or not os.path.exists(p):
                _missing_cell(ax)
            else:
                ax.imshow(mpimg.imread(p))
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            if i == 0:
                ax.set_title(f"t={t}", fontsize=10)
            if j == 0:
                lbl = f"{arch}\n({PARAMS_M[sc][arch]:.1f} M)"
                ax.set_ylabel(lbl, fontsize=9, rotation=0,
                              ha="right", va="center", labelpad=22)

    kind_long = "multistep rollout" if kind == "multistep" else "single-step (teacher-forced)"
    fig.suptitle(f"{size}-{sc} — {kind_long}: rows = NSP arch, cols = t",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def render_spectra_grid(sc, size, kind, paths, out_path):
    """paths[(arch, key)] -> local PNG path or None.

    Layout: rows = metric (TKE / Ens / Hist), cols = NSP arch. Landscape
    aspect (~2.4) so the whole grid fits one 16:9 slide without cropping.
    """
    archs = ARCHS[sc]
    rows = SPECTRA_KEYS
    nrows, ncols = len(rows), len(archs)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.0 * nrows),
                             squeeze=False)
    for i, (key, title) in enumerate(rows):
        for j, arch in enumerate(archs):
            ax = axes[i, j]
            p = paths.get((arch, key))
            if p is None or not os.path.exists(p):
                _missing_cell(ax)
            else:
                ax.imshow(mpimg.imread(p))
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            if i == 0:
                ax.set_title(f"{arch}\n({PARAMS_M[sc][arch]:.1f} M)",
                             fontsize=10)
            if j == 0:
                ax.set_ylabel(title, fontsize=10, rotation=90,
                              ha="center", va="center", labelpad=10)

    kind_long = "multistep rollout" if kind == "multistep" else "single-step (teacher-forced)"
    fig.suptitle(f"{size}-{sc} — {kind_long}: spectra & pixel histograms",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# Metrics → CSV + LaTeX
# ----------------------------------------------------------------------

# Columns differ between kinds. Order = (csv key, summary key, fmt).
COLS_MS = [
    ("tke_rse_nsp",   "tke_rse/nsp",        "{:.4f}"),
    ("tke_rse_vqvae", "tke_rse/vqvae",      "{:.4f}"),
    ("ens_rse_nsp",   "enstrophy_rse/nsp",  "{:.4f}"),
    ("ens_rse_vqvae", "enstrophy_rse/vqvae","{:.4f}"),
    ("emd_nsp",       "emd/nsp",            "{:.4f}"),
    ("emd_vqvae",     "emd/vqvae",          "{:.4f}"),
    ("collapse_rate", "collapse_rate",      "{:.3f}"),
]
COLS_SS = [
    ("cross_entropy", "cross_entropy",      "{:.4f}"),
    ("pixel_rmse",    "pixel_rmse",         "{:.4f}"),
    ("tke_rse_nsp",   "tke_rse/nsp",        "{:.4f}"),
    ("ens_rse_nsp",   "enstrophy_rse/nsp",  "{:.4f}"),
    ("emd_nsp",       "emd/nsp",            "{:.4f}"),
    ("ood_rate_mean", "ood_rate_mean",      "{:.3f}"),
]


def collect_metrics(api, sc, kind):
    """Return list of dicts, one per (size, arch) that exists in wandb."""
    cols = COLS_MS if kind == "multistep" else COLS_SS
    rows = []
    for size in SIZES:
        for arch in ARCHS[sc]:
            run = get_run(api, run_name(sc, size, arch, kind))
            if run is None:
                continue
            row = {"size": size, "arch": arch,
                   "params_M": PARAMS_M[sc][arch]}
            s = run.summary
            for csv_key, summ_key, _ in cols:
                v = s.get(summ_key)
                row[csv_key] = float(v) if v is not None else None
            rows.append(row)
    return rows, cols


def write_csv(rows, cols, out_path):
    fieldnames = ["size", "arch", "params_M"] + [c[0] for c in cols]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k])
                        for k in fieldnames})
    print(f"  saved {out_path}")


def _fmt(v, spec):
    return "--" if v is None else spec.format(v)


def write_latex_table(rows, cols, sc, kind, out_path):
    """booktabs tabular block. Bolds the row with min EMD (ms) / min CE (ss)
    within each VQ size group."""
    # Column header labels (LaTeX-safe).
    headers_ms = [r"TKE RSE", r"TKE VQ", r"Ens RSE", r"Ens VQ",
                  r"EMD", r"EMD VQ", r"collapse"]
    headers_ss = [r"CE", r"px RMSE", r"TKE RSE", r"Ens RSE",
                  r"EMD", r"OOD"]
    headers = headers_ms if kind == "multistep" else headers_ss
    bold_key = "emd_nsp" if kind == "multistep" else "cross_entropy"

    # Determine best (lowest) per VQ size for bolding.
    best = {}
    for size in SIZES:
        in_grp = [r for r in rows
                  if r["size"] == size and r.get(bold_key) is not None]
        if in_grp:
            best[size] = min(in_grp, key=lambda r: r[bold_key])["arch"]

    col_spec = "l l r " + " ".join("r" * len(cols))
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    lines.append("size & arch & params (M) & " +
                 " & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    prev_size = None
    for r in rows:
        if prev_size is not None and r["size"] != prev_size:
            lines.append(r"\midrule")
        prev_size = r["size"]
        cells = [
            r["size"], r["arch"], f"{r['params_M']:.2f}",
            *[_fmt(r.get(c[0]), c[2]) for c in cols],
        ]
        if best.get(r["size"]) == r["arch"]:
            cells = [r"\textbf{" + c + "}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default="plots/scaling_report")
    p.add_argument("--cache_dir",  default="plots/scaling_report/cache")
    p.add_argument("--sc",   default="all",
                   choices=["all"] + SCS)
    p.add_argument("--size", default="all",
                   choices=["all"] + SIZES)
    p.add_argument("--kind", default="all",
                   choices=["all"] + KINDS)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir,  exist_ok=True)

    sel_scs   = SCS   if args.sc   == "all" else [args.sc]
    sel_sizes = SIZES if args.size == "all" else [args.size]
    sel_kinds = KINDS if args.kind == "all" else [args.kind]

    api = wandb.Api()

    # ---- Grids ----
    for sc in sel_scs:
        for size in sel_sizes:
            for kind in sel_kinds:
                snap_paths = {}
                spec_paths = {}
                for arch in ARCHS[sc]:
                    name = run_name(sc, size, arch, kind)
                    run = get_run(api, name)
                    if run is None:
                        print(f"  [missing] {name}")
                        continue
                    run_cache = os.path.join(args.cache_dir, name)
                    for t in SNAPSHOT_TS[kind]:
                        snap_paths[(arch, t)] = fetch_snapshot(
                            run, t, run_cache)
                    for key, _ in SPECTRA_KEYS:
                        spec_paths[(arch, key)] = fetch_spectrum(
                            run, key, run_cache)

                snap_out = os.path.join(
                    args.output_dir, f"snapshots_{sc}_{size}_{kind}.png")
                render_snapshot_grid(sc, size, kind, snap_paths, snap_out)

                spec_out = os.path.join(
                    args.output_dir, f"spectra_{sc}_{size}_{kind}.png")
                render_spectra_grid(sc, size, kind, spec_paths, spec_out)

    # ---- Metrics CSV + LaTeX (only if no size filter — these are per-sc) ----
    if args.size == "all":
        for sc in sel_scs:
            for kind in sel_kinds:
                rows, cols = collect_metrics(api, sc, kind)
                csv_out = os.path.join(
                    args.output_dir, f"metrics_{sc}_{kind}.csv")
                write_csv(rows, cols, csv_out)
                tex_out = os.path.join(
                    args.output_dir, f"table_{sc}_{kind}.tex")
                write_latex_table(rows, cols, sc, kind, tex_out)

    print(f"\nDone. Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
