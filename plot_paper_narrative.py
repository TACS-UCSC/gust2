"""Assemble the greedy-collapse / temperature-recovery narrative figures for
the paper, straight from the wandb-logged per-run PNGs.

Two projects feed the story:
  * gust2-analysis-bridges-scaling  -- GREEDY rollouts, full 5-arch grid per
    config (the collapse baseline). sc341 holds; sc917/sc1941 collapse to a
    mean flow.
  * gust2-sampling-{sc341,sc917,sc1941} -- temperature sweep on the anchor of
    each config (recovery: cranking T un-collapses the field, then over-cooks).

The spectra/histograms are only logged to wandb as *rendered* PNGs (each one
already overlays GT / VQ-VAE-floor / NSP), so we tile those PNGs rather than
re-plot from arrays. The one genuinely re-plotted figure is the scalar
recovery curve (emd/tke gap vs T), built from summary metrics.

Outputs -> plots/paper_narrative/
  greedy_spectra_<sc>.png      rows=NSP arch, cols=[TKE, Enstrophy, Pixel hist]
  greedy_snapshots_<sc>.png    rows=NSP arch, cols=[t1,t100,t500,t1000,t1500]
  temp_spectra_<sc>.png        rows=temperature, cols=[TKE, Pixel hist]
  temp_snapshots_<sc>.png      rows=temperature, cols=[t1,t500,t1000,t1500]
  recovery_curves.png          emd/tke gap vs T, greedy marked, floor=1.0

Usage:
    ~/llm/bin/python plot_paper_narrative.py
    ~/llm/bin/python plot_paper_narrative.py --only sc917      # one config
    ~/llm/bin/python plot_paper_narrative.py --skip-download   # tile from cache
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import wandb

ENTITY = "bigpseud-ucsc"
GREEDY_PROJECT = "gust2-analysis-bridges-scaling"

# ----------------------------------------------------------------------
# What to pull
# ----------------------------------------------------------------------

# Greedy collapse: full NSP-arch sweep per config, at the anchor's VQ size.
GREEDY = {
    "sc341":  {"size": "medium", "archs": ["s06", "s09", "s13", "s18", "s24"]},
    "sc917":  {"size": "large",  "archs": ["s13", "s22", "s34", "s50", "s74"]},
    "sc1941": {"size": "large",  "archs": ["s31", "s48", "s73", "s113", "s139"]},
}

# Temperature recovery: anchor model per config + swept temperatures.
TEMP = {
    "sc341":  {"proj": "gust2-sampling-sc341",  "run": "medium-sc341-nsp-s18",
               "temps": [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], "seed": 0},
    "sc917":  {"proj": "gust2-sampling-sc917",  "run": "large-sc917-nsp-s34",
               "temps": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0], "seed": 0},
    "sc1941": {"proj": "gust2-sampling-sc1941", "run": "large-sc1941-nsp-s73",
               "temps": [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0], "seed": 0},
}

# Anchor arch per config (the model the temperature sweep uses).
ANCHOR = {"sc341": "s18", "sc917": "s34", "sc1941": "s73"}

PARAMS_M = {
    "sc341":  {"s06": 6.04, "s09": 9.11, "s13": 12.65, "s18": 17.96, "s24": 24.04},
    "sc917":  {"s13": 13.32, "s22": 22.18, "s34": 34.14, "s50": 50.07, "s74": 73.68},
    "sc1941": {"s31": 31.17, "s48": 47.64, "s73": 73.11, "s113": 113.43, "s139": 138.74},
}

SPECTRA = [("tke_spectrum", "TKE spectrum"),
           ("enstrophy_spectrum", "Enstrophy spectrum"),
           ("pixel_histogram", "Pixel histogram")]
SPECTRA_TEMP = [("tke_spectrum", "TKE spectrum"),
                ("pixel_histogram", "Pixel histogram")]

SNAP_GREEDY = [1, 100, 500, 1000, 1500]
SNAP_TEMP = [1, 500, 1000, 1500]

OUT_DIR = "plots/paper_narrative"
CACHE_DIR = os.path.join(OUT_DIR, "cache")

# ----------------------------------------------------------------------
# Download (cache-aware, parallel over runs)
# ----------------------------------------------------------------------

def tfmt(t):
    """Match the run-name temperature token: 1.0 -> '1.0', 2.5 -> '2.5'."""
    return f"{t}"


def needed_substrings(kind):
    """List of (substr) to match in a run's file names."""
    subs = []
    if kind == "greedy":
        for key, _ in SPECTRA:
            subs.append(f"{key}_0_")
        for t in SNAP_GREEDY:
            subs.append(f"snapshot/t{t}_")
    else:  # temp
        for key, _ in SPECTRA_TEMP:
            subs.append(f"{key}_0_")
        for t in SNAP_TEMP:
            subs.append(f"snapshot/t{t}_")
    return subs


def download_run(api, project, run_name, substrings, cache_dir):
    """Download every .png in `run_name` matching any substring. Returns
    {substr: local_path}. Cache-aware. Resilient to missing runs/files."""
    out = {}
    run_cache = os.path.join(cache_dir, run_name)
    os.makedirs(run_cache, exist_ok=True)
    try:
        runs = [r for r in api.runs(f"{ENTITY}/{project}",
                                    filters={"display_name": run_name})
                if r.name == run_name]
        if not runs:
            return run_name, out
        run = runs[0]
        files = list(run.files())
    except Exception as e:
        print(f"  [warn] {run_name}: list failed ({e})")
        return run_name, out
    for substr in substrings:
        match = next((f for f in files
                      if substr in f.name and f.name.endswith(".png")), None)
        if match is None:
            continue
        local = os.path.join(run_cache, match.name)
        if not os.path.exists(local):
            try:
                match.download(root=run_cache, replace=True)
            except Exception as e:
                print(f"  [warn] {run_name}:{substr} download failed ({e})")
                continue
        out[substr] = local
    return run_name, out


def gather_downloads(selected, skip_download):
    """Build the full task list, download in parallel, return
    paths[run_name][substr] = local_path."""
    tasks = []  # (project, run_name, substrings)
    for sc in selected:
        g = GREEDY[sc]
        for arch in g["archs"]:
            rn = f"{g['size']}-{sc}-nsp-{arch}"
            tasks.append((GREEDY_PROJECT, rn, needed_substrings("greedy")))
        t = TEMP[sc]
        for temp in t["temps"]:
            rn = f"{t['run']}-T{tfmt(temp)}-s{t['seed']}"
            tasks.append((t["proj"], rn, needed_substrings("temp")))

    paths = {}
    if skip_download:
        # Reconstruct from cache only.
        for _, rn, subs in tasks:
            run_cache = os.path.join(CACHE_DIR, rn)
            d = {}
            if os.path.isdir(run_cache):
                for root, _, files in os.walk(run_cache):
                    for f in files:
                        for substr in subs:
                            if substr in os.path.join(root, f).replace(run_cache + "/", "") \
                               and f.endswith(".png"):
                                d[substr] = os.path.join(root, f)
            paths[rn] = d
        return paths

    api = wandb.Api()
    print(f"Downloading PNGs for {len(tasks)} runs ...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(download_run, api, proj, rn, subs, CACHE_DIR): rn
                for proj, rn, subs in tasks}
        for fut in as_completed(futs):
            rn, d = fut.result()
            paths[rn] = d
            print(f"  {rn}: {len(d)} files")
    return paths


# ----------------------------------------------------------------------
# Tiling
# ----------------------------------------------------------------------

def _blank(ax, msg="(missing)"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            color="0.55", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _imcell(ax, path):
    if path and os.path.exists(path):
        ax.imshow(mpimg.imread(path))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        _blank(ax)


def render_grid(row_labels, col_titles, cell_paths, suptitle, out_path,
                cell_w=2.7, cell_h=2.0):
    """cell_paths[(r, c)] -> local PNG path or None."""
    nr, nc = len(row_labels), len(col_titles)
    top_frac = 0.97 - min(0.04, 0.4 / max(nr, 1))   # more headroom for short grids
    fig, axes = plt.subplots(nr, nc, figsize=(cell_w * nc, cell_h * nr),
                             squeeze=False)
    for i, rlab in enumerate(row_labels):
        for j, ctitle in enumerate(col_titles):
            ax = axes[i, j]
            _imcell(ax, cell_paths.get((i, j)))
            if i == 0:
                ax.set_title(ctitle, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(rlab, fontsize=10, rotation=0, ha="right",
                              va="center", labelpad=8)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.997)
    fig.tight_layout(rect=[0, 0, 1, top_frac])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def greedy_figs(sc, paths):
    g = GREEDY[sc]
    size = g["size"]
    archs = g["archs"]
    rlabels = []
    for a in archs:
        star = "  ★" if a == ANCHOR[sc] else ""
        rlabels.append(f"{a}\n{PARAMS_M[sc][a]:.0f}M{star}")

    # spectra
    cells = {}
    for i, a in enumerate(archs):
        rn = f"{size}-{sc}-nsp-{a}"
        d = paths.get(rn, {})
        for j, (key, _) in enumerate(SPECTRA):
            cells[(i, j)] = d.get(f"{key}_0_")
    render_grid(rlabels, [t for _, t in SPECTRA], cells,
                f"{size}-{sc} — GREEDY rollout (collapse baseline): rows = NSP arch  (★ = anchor)",
                os.path.join(OUT_DIR, f"greedy_spectra_{sc}.png"),
                cell_w=3.2, cell_h=2.4)

    # snapshots
    cells = {}
    for i, a in enumerate(archs):
        rn = f"{size}-{sc}-nsp-{a}"
        d = paths.get(rn, {})
        for j, t in enumerate(SNAP_GREEDY):
            cells[(i, j)] = d.get(f"snapshot/t{t}_")
    render_grid(rlabels, [f"t={t}" for t in SNAP_GREEDY], cells,
                f"{size}-{sc} — GREEDY rollout snapshots (each cell: GT | prediction): rows = NSP arch  (★ = anchor)",
                os.path.join(OUT_DIR, f"greedy_snapshots_{sc}.png"),
                cell_w=2.7, cell_h=1.45)


def temp_figs(sc, paths):
    t = TEMP[sc]
    temps = t["temps"]
    rlabels = [f"T={tt}" for tt in temps]

    # spectra (TKE + histogram)
    cells = {}
    for i, tt in enumerate(temps):
        rn = f"{t['run']}-T{tfmt(tt)}-s{t['seed']}"
        d = paths.get(rn, {})
        for j, (key, _) in enumerate(SPECTRA_TEMP):
            cells[(i, j)] = d.get(f"{key}_0_")
    render_grid(rlabels, [tt for _, tt in SPECTRA_TEMP], cells,
                f"{t['run']} — temperature recovery: rows = sampling T",
                os.path.join(OUT_DIR, f"temp_spectra_{sc}.png"),
                cell_w=3.4, cell_h=2.2)

    # snapshots
    cells = {}
    for i, tt in enumerate(temps):
        rn = f"{t['run']}-T{tfmt(tt)}-s{t['seed']}"
        d = paths.get(rn, {})
        for j, ts in enumerate(SNAP_TEMP):
            cells[(i, j)] = d.get(f"snapshot/t{ts}_")
    render_grid(rlabels, [f"t={ts}" for ts in SNAP_TEMP], cells,
                f"{t['run']} — temperature recovery snapshots (each cell: GT | prediction): rows = T",
                os.path.join(OUT_DIR, f"temp_snapshots_{sc}.png"),
                cell_w=2.9, cell_h=1.5)


# ----------------------------------------------------------------------
# Scalar recovery-curve summary (re-plotted from metrics)
# ----------------------------------------------------------------------

def fetch_scalars(selected):
    api = wandb.Api()
    out = {}
    for sc in selected:
        rec = {"temps": [], "emd_gap": [], "tke_gap": [],
               "greedy_emd_gap": None, "greedy_tke_gap": None}
        # greedy anchor from bridges-scaling
        g = GREEDY[sc]
        rn = f"{g['size']}-{sc}-nsp-{ANCHOR[sc]}"
        runs = [r for r in api.runs(f"{ENTITY}/{GREEDY_PROJECT}",
                                    filters={"display_name": rn}) if r.name == rn]
        if runs:
            s = runs[0].summary
            if s.get("emd/vqvae"):
                rec["greedy_emd_gap"] = s["emd/nsp"] / s["emd/vqvae"]
            if s.get("tke_rse/vqvae"):
                rec["greedy_tke_gap"] = s["tke_rse/nsp"] / s["tke_rse/vqvae"]
        # temperature sweep (seed-avg)
        t = TEMP[sc]
        byT = defaultdict(lambda: defaultdict(list))
        fe, ft = [], []
        for r in api.runs(f"{ENTITY}/{t['proj']}"):
            T = None
            for tok in r.name.split("-"):
                if tok.startswith("T"):
                    try:
                        T = float(tok[1:])
                    except ValueError:
                        pass
            if T is None or r.summary.get("emd/nsp") is None:
                continue
            byT[T]["e"].append(r.summary["emd/nsp"])
            byT[T]["t"].append(r.summary["tke_rse/nsp"])
            if r.summary.get("emd/vqvae"):
                fe.append(r.summary["emd/vqvae"])
            if r.summary.get("tke_rse/vqvae"):
                ft.append(r.summary["tke_rse/vqvae"])
        fe = float(np.median(fe)); ft = float(np.median(ft))
        for T in sorted(byT):
            rec["temps"].append(T)
            rec["emd_gap"].append(float(np.mean(byT[T]["e"])) / fe)
            rec["tke_gap"].append(float(np.mean(byT[T]["t"])) / ft)
        out[sc] = rec
    return out


def recovery_curves(selected, scalars):
    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.4), squeeze=False,
                             sharey=True)
    for k, sc in enumerate(selected):
        ax = axes[0, k]
        rec = scalars[sc]
        Ts = rec["temps"]
        ax.plot(Ts, rec["emd_gap"], "o-", color="#C0392B", lw=1.9, ms=7,
                label="pixel-EMD gap")
        ax.plot(Ts, rec["tke_gap"], "s-", color="#1F4E79", lw=1.9, ms=7,
                label="TKE-RSE gap")
        # greedy markers, placed just left of the swept range
        x0 = min(Ts) - 0.35
        if rec["greedy_emd_gap"] is not None:
            ax.plot([x0], [rec["greedy_emd_gap"]], "*", color="#C0392B", ms=16,
                    markeredgecolor="k", markeredgewidth=0.6, zorder=5)
        if rec["greedy_tke_gap"] is not None:
            ax.plot([x0], [rec["greedy_tke_gap"]], "*", color="#1F4E79", ms=16,
                    markeredgecolor="k", markeredgewidth=0.6, zorder=5)
        ax.axvline(x0 + 0.18, color="0.7", ls=":", lw=1)
        ax.text(x0, ax.get_ylim()[0], "greedy", fontsize=8, color="0.3",
                ha="center", va="bottom")
        ax.axhline(1.0, color="0.35", ls="--", lw=1.3,
                   label="VQ-VAE floor (gap=1.0)")
        ax.set_yscale("log")
        ax.set_ylim(0.4, 13)
        ax.set_xlabel("sampling temperature T", fontsize=11)
        if k == 0:
            ax.set_ylabel("rollout metric / VQ-VAE floor", fontsize=11)
        anchor_p = PARAMS_M[sc][ANCHOR[sc]]
        ax.set_title(f"{sc}  ({TEMP[sc]['run']}, {anchor_p:.0f}M)",
                     fontsize=12, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Greedy collapses → temperature recovers  "
                 "(rollout metric / VQ-VAE floor; <1 beats tokenizer)",
                 fontsize=12.5, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = os.path.join(OUT_DIR, "recovery_curves.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=list(GREEDY))
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args()

    selected = [args.only] if args.only else list(GREEDY)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    paths = gather_downloads(selected, args.skip_download)

    print("\nTiling figures ...")
    for sc in selected:
        greedy_figs(sc, paths)
        temp_figs(sc, paths)

    print("\nRecovery curves ...")
    scalars = fetch_scalars(selected)
    recovery_curves(selected, scalars)

    print(f"\nDone. Figures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
