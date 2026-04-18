"""Detailed analysis of the best-performing NSP rollout.

Produces:
  - Single-step statistics for the first 10 predicted frames (snapshot,
    TKE spectrum, enstrophy spectrum, pixel histogram, per-frame metrics).
  - Long-term (time-averaged) spectra + histogram + metrics over the first
    ``n_rollout_steps`` frames.
  - Snapshots every 100 steps from 100 through ``n_rollout_steps``.

All plots and metrics are written locally and logged to a new wandb project.

Usage:
    python analyze_best.py \
        --rollout_dir .../rollouts-sampling/small-sc341-nsp-large-T0.7-s0 \
        --vqvae_dir   .../vqvae/small-sc341 \
        --data_path   .../data_lowres/output.h5 \
        --output_dir  .../analysis-best/small-sc341-nsp-large-T0.7-s0 \
        --wandb_project gust2-best \
        --wandb_name  small-sc341-nsp-large-T0.7-s0
"""

import argparse
import json
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy.stats import wasserstein_distance

from tokenizer import load_checkpoint
from analyze_rollout import (
    compute_enstrophy_spectrum,
    compute_histograms,
    compute_tke_spectrum,
    decode_all_tokens,
    load_raw_gt,
    load_rollout_data,
    relative_spectral_error,
    setup_spectral_analysis,
)


# =============================================================================
# Plotting
# =============================================================================


def plot_snapshot(gt, vqvae, nsp, t):
    """Three-panel snapshot with a properly-placed colorbar.

    Uses constrained_layout so the figure-level colorbar does not overlap
    the NSP panel (which is what happens when tight_layout() is mixed with a
    figure-level colorbar).
    """
    vmin = float(gt.min())
    vmax = float(gt.max())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, field, label in [
        (axes[0], gt, "Ground Truth"),
        (axes[1], vqvae, "VQ-VAE"),
        (axes[2], nsp, "NSP"),
    ]:
        im = ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       origin="lower")
        ax.set_title(f"{label} (t={t})", fontsize=12)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Vorticity")
    return fig


def plot_triple_spectrum(k, gt, vq, nsp, title, ylabel):
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = k > 0
    for spec, label, color, ls in [
        (gt, "Ground Truth", "blue", "-"),
        (vq, "VQ-VAE", "green", "--"),
        (nsp, "NSP", "red", ":"),
    ]:
        mask = valid & (spec > 0)
        if np.any(mask):
            ax.loglog(k[mask], spec[mask], label=label, color=color,
                      linestyle=ls, alpha=0.8, linewidth=2)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Wavenumber |k|", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    return fig


def plot_triple_histogram(hist_data, title):
    bc = hist_data["bin_centers"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for h, label, color, ls in [
        (hist_data["gt_hist"], "Ground Truth", "blue", "-"),
        (hist_data["vqvae_hist"], "VQ-VAE", "green", "--"),
        (hist_data["nsp_hist"], "NSP", "red", ":"),
    ]:
        ax.step(bc, h, where="mid", label=label, color=color, linestyle=ls,
                linewidth=2, alpha=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Pixel Value (Vorticity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =============================================================================
# Main
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--vqvae_dir", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--field", default="omega")
    p.add_argument("--sample_start", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--n_rollout_steps", type=int, default=1000,
                   help="Number of rollout steps to analyse (uses frames "
                        "0..n_rollout_steps).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", default="gust2-best")
    p.add_argument("--wandb_name", default=None)
    p.add_argument("--wandb_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    first10_dir = os.path.join(args.output_dir, "first10")
    every100_dir = os.path.join(args.output_dir, "every100")
    os.makedirs(first10_dir, exist_ok=True)
    os.makedirs(every100_dir, exist_ok=True)

    # --- Rollout ---
    print(f"Loading rollout from {args.rollout_dir}...")
    rollout = load_rollout_data(args.rollout_dir)
    scales = rollout["scales"]
    new_to_old = rollout["new_to_old"]
    available = rollout["n_steps"] + 1
    n_frames = min(args.n_rollout_steps + 1, available)
    print(f"  scales={list(scales)}  start_frame={rollout['start_frame']}  "
          f"using {n_frames}/{available} frames")

    # --- VQ-VAE ---
    print(f"Loading VQ-VAE from {args.vqvae_dir}...")
    key = jax.random.PRNGKey(args.seed)
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook
    print(f"  Codebook: {codebook.shape}")

    # --- Raw GT ---
    print("Loading raw GT...")
    raw_gt = load_raw_gt(args.data_path, args.field,
                         args.sample_start, rollout["start_frame"], n_frames)
    print(f"  Loaded {raw_gt.shape[0]} frames")

    # --- Decode ---
    print("Decoding VQ-VAE reconstructions...")
    vqvae_fields = decode_all_tokens(
        rollout["gt_indices"][:n_frames],
        decoder, vq, codebook, new_to_old, scales, args.batch_size)

    print("Decoding NSP rollout predictions...")
    nsp_fields = decode_all_tokens(
        rollout["rollout_indices"][:n_frames],
        decoder, vq, codebook, new_to_old, scales, args.batch_size)

    # --- Per-frame spectra (feed both long-term averages and per-step plots) ---
    print("Computing per-frame spectra...")
    H, W = 256, 256
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    n_bins = len(k_centers)

    gt_tke  = np.zeros((n_frames, n_bins))
    gt_ens  = np.zeros((n_frames, n_bins))
    vq_tke  = np.zeros((n_frames, n_bins))
    vq_ens  = np.zeros((n_frames, n_bins))
    nsp_tke = np.zeros((n_frames, n_bins))
    nsp_ens = np.zeros((n_frames, n_bins))
    for i in range(n_frames):
        gt_tke[i]  = compute_tke_spectrum(raw_gt[i, 0], Kx, Ky, Ksq, bin_masks)
        gt_ens[i]  = compute_enstrophy_spectrum(raw_gt[i, 0], bin_masks)
        vq_tke[i]  = compute_tke_spectrum(vqvae_fields[i, 0], Kx, Ky, Ksq, bin_masks)
        vq_ens[i]  = compute_enstrophy_spectrum(vqvae_fields[i, 0], bin_masks)
        nsp_tke[i] = compute_tke_spectrum(nsp_fields[i, 0], Kx, Ky, Ksq, bin_masks)
        nsp_ens[i] = compute_enstrophy_spectrum(nsp_fields[i, 0], bin_masks)
        if (i + 1) % 200 == 0 or i == n_frames - 1:
            print(f"  {i + 1}/{n_frames}")

    gt_tke_avg  = gt_tke.mean(axis=0)
    gt_ens_avg  = gt_ens.mean(axis=0)
    vq_tke_avg  = vq_tke.mean(axis=0)
    vq_ens_avg  = vq_ens.mean(axis=0)
    nsp_tke_avg = nsp_tke.mean(axis=0)
    nsp_ens_avg = nsp_ens.mean(axis=0)

    # --- Long-term histograms + EMD ---
    print("Computing long-term histograms + EMD...")
    gt_pix  = raw_gt[:, 0].ravel()
    vq_pix  = vqvae_fields[:, 0].ravel()
    nsp_pix = nsp_fields[:, 0].ravel()
    long_hist = compute_histograms(gt_pix, vq_pix, nsp_pix)
    rng = np.random.default_rng(args.seed)
    n_sub = min(1_000_000, len(gt_pix))
    gt_sub  = rng.choice(gt_pix,  n_sub, replace=False)
    vq_sub  = rng.choice(vq_pix,  n_sub, replace=False)
    nsp_sub = rng.choice(nsp_pix, n_sub, replace=False)

    longterm = {
        "n_frames":            int(n_frames),
        "tke_rse_vqvae":       relative_spectral_error(vq_tke_avg,  gt_tke_avg),
        "tke_rse_nsp":         relative_spectral_error(nsp_tke_avg, gt_tke_avg),
        "enstrophy_rse_vqvae": relative_spectral_error(vq_ens_avg,  gt_ens_avg),
        "enstrophy_rse_nsp":   relative_spectral_error(nsp_ens_avg, gt_ens_avg),
        "emd_vqvae":           float(wasserstein_distance(vq_sub,  gt_sub)),
        "emd_nsp":             float(wasserstein_distance(nsp_sub, gt_sub)),
    }

    # --- Per-step stats for first 10 predicted frames ---
    print("Computing per-step stats for first 10 frames...")
    first10_times = list(range(1, min(11, n_frames)))
    per_step = []
    for t in first10_times:
        per_step.append({
            "t":                   t,
            "tke_rse_nsp":         relative_spectral_error(nsp_tke[t], gt_tke[t]),
            "enstrophy_rse_nsp":   relative_spectral_error(nsp_ens[t], gt_ens[t]),
            "emd_nsp":             float(wasserstein_distance(
                nsp_fields[t, 0].ravel(), raw_gt[t, 0].ravel())),
            "tke_rse_vqvae":       relative_spectral_error(vq_tke[t], gt_tke[t]),
            "enstrophy_rse_vqvae": relative_spectral_error(vq_ens[t], gt_ens[t]),
            "emd_vqvae":           float(wasserstein_distance(
                vqvae_fields[t, 0].ravel(), raw_gt[t, 0].ravel())),
        })

    # --- Save metrics + arrays ---
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({"longterm": longterm, "per_step_first10": per_step},
                  f, indent=2)

    np.savez_compressed(
        os.path.join(args.output_dir, "analysis_data.npz"),
        k_centers=k_centers,
        gt_tke_per=gt_tke,   gt_ens_per=gt_ens,
        vq_tke_per=vq_tke,   vq_ens_per=vq_ens,
        nsp_tke_per=nsp_tke, nsp_ens_per=nsp_ens,
        gt_tke_avg=gt_tke_avg,   gt_ens_avg=gt_ens_avg,
        vq_tke_avg=vq_tke_avg,   vq_ens_avg=vq_ens_avg,
        nsp_tke_avg=nsp_tke_avg, nsp_ens_avg=nsp_ens_avg,
        hist_bin_centers=long_hist["bin_centers"],
        hist_gt=long_hist["gt_hist"],
        hist_vqvae=long_hist["vqvae_hist"],
        hist_nsp=long_hist["nsp_hist"],
    )

    # --- Long-term plots ---
    print("Plotting long-term averages...")
    tke_avg_fig = plot_triple_spectrum(
        k_centers, gt_tke_avg, vq_tke_avg, nsp_tke_avg,
        f"Time-averaged TKE (t=0..{n_frames - 1})", "E(k)")
    ens_avg_fig = plot_triple_spectrum(
        k_centers, gt_ens_avg, vq_ens_avg, nsp_ens_avg,
        f"Time-averaged Enstrophy (t=0..{n_frames - 1})", "Z(k)")
    hist_avg_fig = plot_triple_histogram(
        long_hist, f"Pixel histogram (t=0..{n_frames - 1})")
    tke_avg_fig.savefig(os.path.join(args.output_dir, "tke_spectrum.png"), dpi=150)
    ens_avg_fig.savefig(os.path.join(args.output_dir, "enstrophy_spectrum.png"), dpi=150)
    hist_avg_fig.savefig(os.path.join(args.output_dir, "pixel_histogram.png"), dpi=150)

    # --- Per-step plots (first 10) ---
    print("Plotting first-10 per-step figures...")
    first10_figs = {}
    for t in first10_times:
        snap = plot_snapshot(raw_gt[t, 0], vqvae_fields[t, 0],
                             nsp_fields[t, 0], t)
        tke  = plot_triple_spectrum(k_centers, gt_tke[t], vq_tke[t], nsp_tke[t],
                                    f"TKE spectrum (t={t})", "E(k)")
        ens  = plot_triple_spectrum(k_centers, gt_ens[t], vq_ens[t], nsp_ens[t],
                                    f"Enstrophy spectrum (t={t})", "Z(k)")
        hd   = compute_histograms(raw_gt[t, 0].ravel(),
                                  vqvae_fields[t, 0].ravel(),
                                  nsp_fields[t, 0].ravel())
        hist = plot_triple_histogram(hd, f"Pixel histogram (t={t})")
        snap.savefig(os.path.join(first10_dir, f"snapshot_t{t:02d}.png"), dpi=150)
        tke .savefig(os.path.join(first10_dir, f"tke_t{t:02d}.png"),      dpi=150)
        ens .savefig(os.path.join(first10_dir, f"ens_t{t:02d}.png"),      dpi=150)
        hist.savefig(os.path.join(first10_dir, f"hist_t{t:02d}.png"),     dpi=150)
        first10_figs[t] = (snap, tke, ens, hist)

    # --- Long-term snapshots every 100 ---
    print("Plotting every-100 snapshots...")
    every100_times = list(range(100, n_frames, 100))
    every100_figs = {}
    for t in every100_times:
        snap = plot_snapshot(raw_gt[t, 0], vqvae_fields[t, 0],
                             nsp_fields[t, 0], t)
        snap.savefig(os.path.join(every100_dir, f"snapshot_t{t:04d}.png"), dpi=150)
        every100_figs[t] = snap

    # --- Wandb ---
    print(f"Logging to wandb project {args.wandb_project}...")
    if args.wandb_dir is not None:
        os.makedirs(args.wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = args.wandb_dir
    run_name = args.wandb_name or os.path.basename(
        os.path.normpath(args.rollout_dir))
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "rollout_dir": args.rollout_dir,
            "vqvae_dir":   args.vqvae_dir,
            "n_frames":    int(n_frames),
            "scales":      list(scales),
            "first10_times":  first10_times,
            "every100_times": every100_times,
        },
    )

    # Per-step scalars + images at step=t (creates a scrubbable timeline).
    for s in per_step:
        t = s["t"]
        snap, tke, ens, hist = first10_figs[t]
        wandb.log({
            "step/tke_rse/nsp":         s["tke_rse_nsp"],
            "step/enstrophy_rse/nsp":   s["enstrophy_rse_nsp"],
            "step/emd/nsp":             s["emd_nsp"],
            "step/tke_rse/vqvae":       s["tke_rse_vqvae"],
            "step/enstrophy_rse/vqvae": s["enstrophy_rse_vqvae"],
            "step/emd/vqvae":           s["emd_vqvae"],
            "snapshot":                 wandb.Image(snap),
            "spectrum/tke":             wandb.Image(tke),
            "spectrum/enstrophy":       wandb.Image(ens),
            "histogram":                wandb.Image(hist),
        }, step=t)

    for t, snap in every100_figs.items():
        wandb.log({"snapshot": wandb.Image(snap)}, step=t)

    wandb.log({
        "longterm/tke_spectrum":       wandb.Image(tke_avg_fig),
        "longterm/enstrophy_spectrum": wandb.Image(ens_avg_fig),
        "longterm/pixel_histogram":    wandb.Image(hist_avg_fig),
    }, step=n_frames - 1)

    for k, v in longterm.items():
        wandb.run.summary[f"longterm/{k}"] = v

    wandb.finish()
    plt.close("all")

    # --- Console summary ---
    print(f"\n=== Long-term averages ({n_frames} frames) ===")
    for k, v in longterm.items():
        print(f"  {k:<22}{v:.4f}" if isinstance(v, float) else f"  {k:<22}{v}")

    print("\n=== Per-step (first 10 predicted frames) ===")
    print(f"  {'t':>3} {'tke_rse':>9} {'ens_rse':>9} {'emd':>9}")
    for s in per_step:
        print(f"  {s['t']:>3} {s['tke_rse_nsp']:>9.4f} "
              f"{s['enstrophy_rse_nsp']:>9.4f} {s['emd_nsp']:>9.4f}")


if __name__ == "__main__":
    main()
