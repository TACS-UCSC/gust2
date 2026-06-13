"""visualize_diagnostics.py — continuous, threshold-free visual analysis of
one model's posmask-temp diagnostics sweep.

The EMD-threshold collapse flag in multitraj_survival.py is an artifact of
the per-SCALE-mask era, where going OOD was a phase-transition-like event a
hard cutoff could detect. Under the per-position mask the failure mode is
*diffusive drift* — continuous, gradual, sometimes recovering — so this
script shows raw traces and decoded fields with NO collapsed/survived
classification. Reference levels (VQ floor, 2x floor) are drawn as faint
lines only.

Figures (written to --output_dir, default <sweep_root>/visual):
  1. emd_traces.png        — windowed-EMD medians+IQR vs t, all temps on one
                             axes + per-temp small multiples (CPU)
  2. logit_traces.png      — entropy / top-1 / finest-scale entropy medians
                             vs t, temperature-colored (CPU)
  3. emd_vs_entropy.png    — per-temp small multiples: EMD (left axis) with
                             frame entropy (right axis) — do logit shifts
                             move with the drift? (CPU)
  4. snapshots.png         — filmstrip: rows = GT, VQ recon, one row per
                             temp (representative trajectory = median final
                             EMD); cols = rollout times (GPU decode)
  5. highk_energy.png      — time-resolved high-k TKE band energy vs t per
                             temp, with GT and VQ-recon reference bands
                             (decode; this is the diffuse-attractor
                             mechanism made visible)
  6. temperature_selection.png — quantitative "which T" selector: late-window
                             pixel-EMD vs T (primary, no decode) + time-avg
                             E(k) overlay, high-k/GT ratio, pixel-PDF
                             Wasserstein, spectral RSE. Best T flagged; high-k
                             ratio ~1 separates genuine restoration from
                             high-T noise. Per-T metrics also logged as wandb
                             scalars (select/<cfg>/...).

Figures 1-3 and the EMD-only form of figure 6 need NO decode — they run on a
login node with --skip_decode. The decode-backed figures (4, 5, full 6) run
on GPU *or* CPU; JAX uses whatever device is present (slower on CPU but a
single decode pass feeds figures 5 and 6). For a CPU-only node set
JAX_PLATFORMS=cpu and keep the decode budget modest (--spectra_stride 100
--n_traj_spectra 2); see scripts/bridges/submit_visualize.sh --cpu.

Batch submission (one 1-GPU job per model, figures also logged to wandb):
  ./scripts/bridges/submit_visualize.sh [--model sc1941] [--dry-run]

Interactive-node usage on Bridges2:
  srun -p GPU-shared --gres=gpu:h100-80:1 -N 1 -A mth260004p \\
       -t 2:00:00 --pty bash
  source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
  module load cuda/12.6.1
  cd /ocean/projects/mth260004p/sambamur/gust
  OCEAN=/ocean/projects/mth260004p/sambamur
  python visualize_diagnostics.py \\
      --sweep_root $OCEAN/experiments/diagnostics/posmask-temp/small-sc1941-nsp-s73 \\
      --vqvae_dir  $OCEAN/experiments/vqvae/small-sc1941 \\
      --data_path  $OCEAN/data_lowres/output.h5 \\
      --no_wandb
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from diagnostics_common import (
    add_wandb_args,
    assign_cfg_styles,
    band_plot,
    describe_decode,
    discover_cfgs,
    ema,
    init_wandb,
    load_cfg_meta,
    load_rollout_tokens,
    outside_legend,
    safe_median,
    set_diag_style,
    wandb_log_figs_and_scalars,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Threshold-free visual analysis of a diagnostics sweep")
    p.add_argument("--sweep_root", required=True,
                   help="model sweep dir containing <cfg>/rollout/ + survival/")
    p.add_argument("--vqvae_dir", default=None,
                   help="VQ-VAE checkpoint (needed unless --skip_decode)")
    p.add_argument("--data_path", default=None,
                   help="HDF5 data file (needed unless --skip_decode)")
    p.add_argument("--output_dir", default=None,
                   help="default: <sweep_root>/visual")
    p.add_argument("--skip_decode", action="store_true",
                   help="only the CPU figures (1-3); no GPU/JAX needed")
    p.add_argument("--field", default="omega")
    p.add_argument("--sample_start", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--snap_times", default="0,250,500,1000,1500,2000",
                   help="comma-separated rollout times for the filmstrip")
    p.add_argument("--spectra_stride", type=int, default=50,
                   help="decode every Nth frame for time-resolved spectra")
    p.add_argument("--n_traj_spectra", type=int, default=4,
                   help="trajectories per temp for time-resolved spectra")
    p.add_argument("--highk_frac", type=float, default=0.33,
                   help="high-k band = top this fraction of k bins")
    p.add_argument("--ema", type=int, default=20,
                   help="EMA window for logit traces (0 disables)")
    add_wandb_args(p)
    return p.parse_args()


def median_final_emd_traj(emd):
    """Index of the trajectory whose final windowed EMD is the median —
    a representative trajectory, not best- or worst-case."""
    final = emd[:, -1]
    return int(np.argsort(final)[len(final) // 2])


# =============================================================================
# CPU figures
# =============================================================================


def fig_emd_traces(cfgs, styles, metas, surv_npz, output_dir):
    probe_times = surv_npz["probe_times"]
    vq_floor = None  # drawn only if survival.json metadata is present

    n_cfg = len(cfgs)
    ncols = 4
    nrows = (n_cfg + ncols - 1) // ncols
    fig = plt.figure(figsize=(4.2 * ncols, 3.0 * (nrows + 1.4)),
                     constrained_layout=True)
    sub = fig.subfigures(2, 1, height_ratios=[1.4, nrows])

    # --- top: all temps overlaid, medians only ---
    ax = sub[0].subplots(1, 1)
    for cfg in cfgs:
        emd = surv_npz[f"emd_{cfg}"]
        med = safe_median(emd, axis=0)
        ax.plot(probe_times, med, lw=2.0, color=styles[cfg]["color"],
                ls=styles[cfg]["ls"],
                label=f"{cfg} [{describe_decode(metas[cfg])}]")
    ax.set_xlabel("rollout step t")
    ax.set_ylabel("window EMD vs GT pool")
    ax.set_title("Windowed pixel-EMD drift (medians; no collapse flags)")
    outside_legend(ax)

    # --- bottom: per-temp small multiples with IQR + individuals ---
    axes = sub[1].subplots(nrows, ncols, sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax, cfg in zip(axes.ravel(), cfgs):
        emd = surv_npz[f"emd_{cfg}"]
        band_plot(ax, probe_times, emd, color=styles[cfg]["color"],
                  individual_max=30)
        ax.set_title(f"{cfg}  (N={emd.shape[0]})", fontsize=10)
    for ax in axes.ravel()[n_cfg:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    for ax in axes[:, 0]:
        ax.set_ylabel("window EMD")

    out = os.path.join(output_dir, "emd_traces.png")
    fig.savefig(out)
    print(f"saved {out}")
    return fig


def fig_logit_traces(cfgs, styles, metas, diag, output_dir, smooth):
    """diag: {cfg: diagnostics.npz dict}."""
    any_d = next(iter(diag.values()))
    scales = np.asarray(any_d["scales"]).tolist()
    rows = [
        ("frame_entropy", "top-K entropy (nats)"),
        ("frame_top1_prob", "top-1 prob"),
        ("per_scale_entropy", f"H finest ({scales[-1]}×{scales[-1]})"),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 3.0 * len(rows)),
                             sharex=True, constrained_layout=True)
    for r, (key, ylabel) in enumerate(rows):
        ax = axes[r]
        for cfg in cfgs:
            if cfg not in diag:
                continue
            arr = diag[cfg][key]
            if key == "per_scale_entropy":
                arr = arr[..., -1]
            med = safe_median(arr, axis=0)
            if smooth > 1:
                med = ema(med, smooth)
            ax.plot(np.arange(med.shape[0]), med, lw=1.8,
                    color=styles[cfg]["color"], ls=styles[cfg]["ls"],
                    label=f"{cfg} [{describe_decode(metas[cfg])}]")
        ax.set_ylabel(ylabel)
        if r == 0:
            outside_legend(ax)
    axes[-1].set_xlabel("rollout step t")
    fig.suptitle("Logit traces vs absolute time (medians, no alignment)")
    out = os.path.join(output_dir, "logit_traces.png")
    fig.savefig(out)
    print(f"saved {out}")
    return fig


def fig_emd_vs_entropy(cfgs, styles, surv_npz, diag, output_dir, smooth):
    probe_times = surv_npz["probe_times"]
    n_cfg = len(cfgs)
    ncols = 4
    nrows = (n_cfg + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.4 * ncols, 3.0 * nrows),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for ax, cfg in zip(axes.ravel(), cfgs):
        emd_med = safe_median(surv_npz[f"emd_{cfg}"], axis=0)
        ax.plot(probe_times, emd_med, lw=1.8, color=styles[cfg]["color"])
        ax.set_title(cfg, fontsize=10)
        ax.tick_params(axis="y", labelcolor=styles[cfg]["color"])
        if cfg in diag:
            ent_med = safe_median(diag[cfg]["frame_entropy"], axis=0)
            if smooth > 1:
                ent_med = ema(ent_med, smooth)
            ax2 = ax.twinx()
            ax2.plot(np.arange(ent_med.shape[0]), ent_med, lw=1.2,
                     color="0.25", ls="--", alpha=0.8)
            ax2.tick_params(axis="y", labelcolor="0.25", labelsize=8)
            ax2.grid(False)
    for ax in axes.ravel()[n_cfg:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("t")
    fig.suptitle("Windowed EMD (color, left) vs frame entropy "
                 "(gray dashed, right) — per temperature")
    out = os.path.join(output_dir, "emd_vs_entropy.png")
    fig.savefig(out)
    print(f"saved {out}")
    return fig


# =============================================================================
# GPU figures
# =============================================================================


def fig_snapshots(args, cfgs, metas, surv_npz, decode_fn, gt, output_dir):
    """Filmstrip: rows = GT, VQ recon, one per temp; cols = snap times."""
    snap_times = [int(t) for t in args.snap_times.split(",")]
    n_frames = gt.shape[0]
    snap_times = [t for t in snap_times if t < n_frames]

    # Representative trajectory per cfg = median final EMD.
    rep = {}
    for cfg in cfgs:
        emd = surv_npz[f"emd_{cfg}"]
        rep[cfg] = median_final_emd_traj(emd)

    # Decode VQ recon of GT tokens (codebook ceiling reference) and each
    # cfg's representative trajectory at the snap times.
    first = load_rollout_tokens(
        os.path.join(args.sweep_root, cfgs[0], "rollout"))
    gt_idx = first["gt_indices"][0]                       # (T+1, P)
    print("decoding VQ recon reference...")
    vq_fields = decode_fn(gt_idx[snap_times])
    rows = [("GT", gt[snap_times, 0]),
            ("VQ recon", vq_fields[:, 0])]
    for cfg in cfgs:
        d = load_rollout_tokens(os.path.join(args.sweep_root, cfg, "rollout"))
        j = rep[cfg]
        print(f"decoding {cfg} traj {j}...")
        fields = decode_fn(d["rollout_indices"][j][snap_times])
        rows.append((f"{cfg}\n[{describe_decode(metas[cfg])}] traj{j}",
                     fields[:, 0]))

    vmax = float(np.percentile(np.abs(gt[snap_times, 0]), 99.5))
    n_rows, n_cols = len(rows), len(snap_times)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.1 * n_cols, 2.1 * n_rows),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    im = None
    for r, (label, fields) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            im = ax.imshow(fields[c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if r == 0:
                ax.set_title(f"t={snap_times[c]}", fontsize=10)
        axes[r, 0].set_ylabel(label, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=30)
    # Shared colorbar; constrained_layout handles spacing — per project
    # convention, never tight_layout / bbox_inches='tight' here.
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    out = os.path.join(output_dir, "snapshots.png")
    fig.savefig(out)
    print(f"saved {out}")
    return fig


def _subsample(arr, n, seed=0):
    arr = np.asarray(arr).ravel()
    if arr.size <= n:
        return arr
    rng = np.random.default_rng(seed)
    return rng.choice(arr, n, replace=False)


def compute_decoded_stats(args, cfgs, decode_fn, gt, n_pix=200_000):
    """Single decode pass per cfg -> spectra + pixel pools, shared by the
    high-k and temperature-selection figures (one decode pass total — the
    only GPU/CPU-heavy step). Decodes args.n_traj_spectra trajectories every
    args.spectra_stride frames; references decode GT-token VQ recon + raw GT.
    """
    from analyze_rollout import (compute_tke_spectrum,
                                 setup_spectral_analysis)
    H, W = gt.shape[-2:]
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    n_bins = len(k_centers)
    band = slice(int(n_bins * (1.0 - args.highk_frac)), n_bins)

    def spectra(fields):                       # (T,H,W) -> (T, n_bins)
        return np.stack([compute_tke_spectrum(f, Kx, Ky, Ksq, bin_masks)
                         for f in fields])

    probes = np.arange(0, gt.shape[0], args.spectra_stride)

    gt_spec_t = spectra(gt[probes, 0])
    first = load_rollout_tokens(
        os.path.join(args.sweep_root, cfgs[0], "rollout"))
    print("decoding VQ recon reference...")
    vq_fields = decode_fn(first["gt_indices"][0][probes])[:, 0]
    vq_spec_t = spectra(vq_fields)

    stats = {}
    for cfg in cfgs:
        idx = load_rollout_tokens(
            os.path.join(args.sweep_root, cfg, "rollout"))["rollout_indices"]
        n_traj = min(args.n_traj_spectra, idx.shape[0])
        print(f"decoding {cfg} ({n_traj} traj x {len(probes)} probes)...")
        sp, pix = [], []
        for j in range(n_traj):
            f = decode_fn(idx[j][probes])[:, 0]
            sp.append(spectra(f))
            pix.append(f.ravel())
        sp = np.stack(sp)                       # (n_traj, T, n_bins)
        stats[cfg] = {
            "highk_t": np.median(sp[..., band].sum(-1), axis=0),  # (T,)
            "avg_spec": sp.mean(axis=(0, 1)),                     # (n_bins,)
            "pix": _subsample(np.concatenate(pix), n_pix),
        }
    return {
        "probes": probes, "k_centers": k_centers, "band": band,
        "gt_avg_spec": gt_spec_t.mean(0), "vq_avg_spec": vq_spec_t.mean(0),
        "gt_highk_t": gt_spec_t[:, band].sum(1),
        "vq_highk_t": vq_spec_t[:, band].sum(1),
        "gt_pix": _subsample(gt[probes, 0], n_pix),
        "stats": stats,
    }


def fig_highk_energy(decoded, cfgs, styles, metas, highk_frac, output_dir):
    """Time-resolved high-k TKE band energy per temp — the diffuse-attractor
    mechanism: spectral collapse shows up as this trace decaying."""
    probes = decoded["probes"]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.plot(probes, decoded["gt_highk_t"], color="k", lw=1.5, label="GT")
    ax.plot(probes, decoded["vq_highk_t"], color="0.5", lw=1.5, ls=":",
            label="VQ recon (codebook ceiling)")
    for cfg in cfgs:
        ax.plot(probes, decoded["stats"][cfg]["highk_t"], lw=1.8,
                color=styles[cfg]["color"], ls=styles[cfg]["ls"],
                label=f"{cfg} [{describe_decode(metas[cfg])}]")
    ax.set_yscale("log")
    ax.set_xlabel("rollout step t")
    ax.set_ylabel(f"high-k TKE (top {highk_frac:.0%} of k bins)")
    ax.set_title("Time-resolved high-k energy — diffusive collapse = decay "
                 "toward the smooth attractor")
    outside_legend(ax)
    out = os.path.join(output_dir, "highk_energy.png")
    fig.savefig(out)
    print(f"saved {out}")
    return fig


def late_emd(surv_npz, cfg, frac=0.5):
    """Median over trajectories of the late-window (t > frac*T) mean
    windowed pixel-EMD vs GT — the threshold-free drift magnitude."""
    emd = surv_npz[f"emd_{cfg}"]
    pt = np.asarray(surv_npz["probe_times"])
    late = pt > pt.max() * frac
    return float(np.median(emd[:, late].mean(axis=1)))


def fig_temperature_selection(cfgs, styles, metas, surv_npz, diag, decoded,
                              output_dir):
    """Quantitative 'which temperature' selector — replaces eyeballing.

    Primary metric = late-window pixel-EMD vs GT (already in survival_data,
    needs NO decode, so this figure renders on a login node with
    --skip_decode). With decoded stats it adds the time-averaged E(k) overlay
    plus high-k/GT ratio, pixel-PDF Wasserstein, and spectral RSE vs T, so the
    optimum is read off curves. Best T is flagged by min pixel-EMD; the
    high-k ratio (~1 = GT) marks the genuine-restoration vs noise-overshoot
    boundary that distinguishes a real optimum from high-T noise gaming.

    Returns (fig, scalars).
    """
    from scipy.stats import wasserstein_distance
    temps = np.array([metas[c].get("temperature") or np.nan for c in cfgs])

    emd_vals = np.array([late_emd(surv_npz, c) for c in cfgs])
    best_i = int(np.nanargmin(emd_vals))
    best_T = temps[best_i]

    scalars = {"best_T_pixel_emd": float(best_T)}
    for c, e in zip(cfgs, emd_vals):
        scalars[f"select/{c}/late_pixel_emd"] = float(e)

    if decoded is None:
        # EMD-only ranking (+ final entropy if logits present) — zero decode.
        cols = 2 if diag else 1
        fig, axes = plt.subplots(1, cols, figsize=(6.5 * cols, 4.5),
                                 constrained_layout=True, squeeze=False)
        ax = axes[0, 0]
        ax.plot(temps, emd_vals, "-o", color="0.3")
        ax.plot(best_T, emd_vals[best_i], "*", ms=18, color="C3",
                label=f"best T={best_T:g}")
        ax.set_xlabel("temperature")
        ax.set_ylabel("late pixel-EMD vs GT")
        ax.set_title("Pixel-EMD vs temperature (lower = better)")
        ax.legend()
        if diag:
            ax = axes[0, 1]
            ent = np.array([
                np.nanmedian(diag[c]["frame_entropy"][:, -200:])
                if c in diag else np.nan for c in cfgs])
            ax.plot(temps, ent, "-o", color="0.3")
            ax.set_xlabel("temperature")
            ax.set_ylabel("late-window entropy (nats)")
            ax.set_title("Final entropy vs temperature")
        fig.suptitle(f"Temperature selection (EMD-only) — best T={best_T:g}")
        out = os.path.join(output_dir, "temperature_selection.png")
        fig.savefig(out)
        print(f"saved {out}")
        return fig, scalars

    # Full version with decoded spectra + PDF.
    kc = decoded["k_centers"]
    band = decoded["band"]
    gt_spec = decoded["gt_avg_spec"]
    vq_spec = decoded["vq_avg_spec"]
    gt_hk = gt_spec[band].sum()
    gt_pix = decoded["gt_pix"]

    hk_ratio, pdf_w, spec_rse = [], [], []
    for c in cfgs:
        s = decoded["stats"][c]
        hk_ratio.append(s["avg_spec"][band].sum() / gt_hk)
        pdf_w.append(wasserstein_distance(s["pix"], gt_pix))
        spec_rse.append(np.linalg.norm(s["avg_spec"] - gt_spec)
                        / (np.linalg.norm(gt_spec) + 1e-12))
    hk_ratio = np.array(hk_ratio)
    pdf_w = np.array(pdf_w)
    spec_rse = np.array(spec_rse)
    for c, r, w, rse in zip(cfgs, hk_ratio, pdf_w, spec_rse):
        scalars[f"select/{c}/highk_ratio_to_gt"] = float(r)
        scalars[f"select/{c}/pdf_wasserstein"] = float(w)
        scalars[f"select/{c}/spectral_rse"] = float(rse)

    fig = plt.figure(figsize=(17, 7), constrained_layout=True)
    sub = fig.subfigures(1, 2, width_ratios=[1.15, 1])
    axL = sub[0].subplots(1, 1)
    axL.loglog(kc, gt_spec, color="k", lw=2.0, label="GT")
    axL.loglog(kc, vq_spec, color="0.5", lw=1.6, ls=":",
               label="VQ recon (ceiling)")
    for c in cfgs:
        axL.loglog(kc, decoded["stats"][c]["avg_spec"], lw=1.4,
                   color=styles[c]["color"], ls=styles[c]["ls"],
                   label=f"{c} [{describe_decode(metas[c])}]")
    axL.axvspan(kc[band][0], kc[-1], color="0.85", alpha=0.5, zorder=0)
    axL.set_xlabel("wavenumber k")
    axL.set_ylabel("time-averaged E(k)")
    axL.set_title("Time-averaged TKE spectrum vs GT (shaded = high-k band)")
    outside_legend(axL)

    axes = sub[1].subplots(2, 2)

    def metric_vs_t(ax, vals, ylabel, title, hline=None, best="min"):
        ax.plot(temps, vals, "-o", color="0.3")
        if best == "min":
            bi = int(np.nanargmin(vals))
        else:                                  # closest to hline
            bi = int(np.nanargmin(np.abs(vals - hline)))
        ax.plot(temps[bi], vals[bi], "*", ms=15, color="C3")
        if hline is not None:
            ax.axhline(hline, color="C0", ls="--", lw=1, label="GT match")
            ax.legend(fontsize=8)
        ax.set_xlabel("temperature")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)

    metric_vs_t(axes[0, 0], emd_vals, "late pixel-EMD",
                "Pixel-EMD (primary, lower=better)")
    metric_vs_t(axes[0, 1], pdf_w, "Wasserstein", "Pixel-PDF distance")
    metric_vs_t(axes[1, 0], hk_ratio, "E_T(hi-k)/E_GT",
                "High-k ratio (1=GT, >1=noise)", hline=1.0, best="near")
    metric_vs_t(axes[1, 1], spec_rse, "spectral RSE", "Spectral RSE")

    hk_at_best = float(hk_ratio[best_i])
    note = ("high-k drained" if hk_at_best < 0.8 else
            "noise overshoot" if hk_at_best > 1.5 else "GT-matched")
    scalars["best_T_highk_ratio"] = hk_at_best
    fig.suptitle(f"Temperature selection — best by pixel-EMD: T={best_T:g} "
                 f"(high-k ratio {hk_at_best:.2f}, {note})")
    out = os.path.join(output_dir, "temperature_selection.png")
    fig.savefig(out)
    print(f"saved {out}")
    return fig, scalars


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.sweep_root, "visual")
    os.makedirs(output_dir, exist_ok=True)
    set_diag_style()

    cfgs = discover_cfgs(args.sweep_root)
    if not cfgs:
        raise SystemExit(f"No cfgs with rollouts under {args.sweep_root}")
    metas = {c: load_cfg_meta(os.path.join(args.sweep_root, c))
             for c in cfgs}
    # Order panels by temperature.
    cfgs = sorted(cfgs, key=lambda c: (metas[c].get("temperature") or 0))
    styles = assign_cfg_styles(metas)
    print(f"{len(cfgs)} cfgs: {cfgs}")

    surv_path = os.path.join(args.sweep_root, "survival", "survival_data.npz")
    surv_npz = dict(np.load(surv_path)) if os.path.isfile(surv_path) else None
    if surv_npz is None:
        print(f"[warn] no {surv_path} — EMD figures skipped "
              f"(run multitraj_survival.py first)")

    diag = {}
    for cfg in cfgs:
        p = os.path.join(args.sweep_root, cfg, "logits", "diagnostics.npz")
        if os.path.isfile(p):
            d = np.load(p)
            diag[cfg] = {k: d[k] for k in
                         ("frame_entropy", "frame_top1_prob",
                          "per_scale_entropy", "scales")}

    figs = {}
    scalars = {}

    # ---- CPU figures ----
    if surv_npz is not None:
        figs["emd_traces"] = fig_emd_traces(
            cfgs, styles, metas, surv_npz, output_dir)
        if diag:
            figs["emd_vs_entropy"] = fig_emd_vs_entropy(
                cfgs, styles, surv_npz, diag, output_dir, args.ema)
    if diag:
        figs["logit_traces"] = fig_logit_traces(
            cfgs, styles, metas, diag, output_dir, args.ema)

    # ---- decode-backed figures (run on GPU or CPU; CPU just slower) ----
    if not args.skip_decode:
        if not (args.vqvae_dir and args.data_path):
            raise SystemExit(
                "--vqvae_dir and --data_path required unless --skip_decode")
        import jax
        from analyze_rollout import decode_all_tokens, load_raw_gt
        from tokenizer import load_checkpoint
        import jax.numpy as jnp

        print(f"JAX devices: {jax.devices()}")
        print("Loading VQ-VAE...")
        key = jax.random.PRNGKey(0)
        _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
        codebook = ema_state.codebook

        first = load_rollout_tokens(
            os.path.join(args.sweep_root, cfgs[0], "rollout"))
        scales = tuple(int(s) for s in first["scales"])
        new_to_old = jnp.array(first["new_to_old"])
        n_frames = int(first["n_steps"]) + 1
        start_frame = int(first["start_frame"])

        def decode_fn(flat_indices):
            return decode_all_tokens(np.asarray(flat_indices), decoder, vq,
                                     codebook, new_to_old, scales,
                                     args.batch_size)

        print("Loading GT fields...")
        gt = load_raw_gt(args.data_path, args.field, args.sample_start,
                         start_frame, n_frames)

        if surv_npz is not None:
            figs["snapshots"] = fig_snapshots(
                args, cfgs, metas, surv_npz, decode_fn, gt, output_dir)

        # One decode pass feeds both high-k and the temperature selector.
        decoded = compute_decoded_stats(args, cfgs, decode_fn, gt)
        figs["highk_energy"] = fig_highk_energy(
            decoded, cfgs, styles, metas, args.highk_frac, output_dir)
        if surv_npz is not None:
            figs["temperature_selection"], scalars = fig_temperature_selection(
                cfgs, styles, metas, surv_npz, diag, decoded, output_dir)
    elif surv_npz is not None:
        # CPU/login-node ranking: pixel-EMD vs T, no decode.
        figs["temperature_selection"], scalars = fig_temperature_selection(
            cfgs, styles, metas, surv_npz, diag, None, output_dir)

    if "best_T_pixel_emd" in scalars:
        print(f"\n>>> best temperature by pixel-EMD: "
              f"T={scalars['best_T_pixel_emd']:g}"
              + (f"  (high-k ratio {scalars['best_T_highk_ratio']:.2f})"
                 if "best_T_highk_ratio" in scalars else ""))

    # ---- wandb ----
    run = init_wandb(args, job_type="visual", config={
        "sweep_root": args.sweep_root,
        "n_cfgs": len(cfgs),
        "skip_decode": args.skip_decode,
        "spectra_stride": args.spectra_stride,
        "n_traj_spectra": args.n_traj_spectra,
        "highk_frac": args.highk_frac,
    })
    wandb_log_figs_and_scalars(run, scalars=scalars, figs=figs)
    print(f"\nDone — {len(figs)} figures in {output_dir}")


if __name__ == "__main__":
    main()
