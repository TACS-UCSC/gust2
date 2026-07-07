"""Tokenizer recon floor on the val window — the B3 gate's test-time metrics.

Runs the full quantized reconstruction path (encoder -> MultiScaleVQ argmax
-> decoder) over the val window for any VQ-VAE checkpoint — the flat B3
tokenizers are degenerate single-scale configs, so one script covers the
whole Pareto. Computes the SAME metrics as the family floors (analyze_rollout
machinery: pixel EMD = the `emd/vqvae` floor, TKE/enstrophy RSE) plus the
spectrum-honesty extras the train-loss gate read can't see: recon spectra
overlays, high-k retention, pixel histograms, and val-window codebook usage.

Motivation (2026-07-06): flat-sc1024 BEATS sc917 on train recon MSE while
using only 275/4096 codes — if its recon spectrum tail is much worse (mode-
averaged/low-pass recons), MSE flatters it and the M6 gate verdict flips
back. Judge by EMD + spectra, not MSE.

Outputs (mirroring analyze_rollout conventions):
  recon_floor_data.npz   spectra + histograms for replotting
  metrics.json           scalar metrics
  tke_spectrum.png, enstrophy_spectrum.png, pixel_histogram.png,
  snapshot_t0.png
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import json
import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from tokenizer import load_checkpoint
from analyze_rollout import (
    setup_spectral_analysis,
    setup_radial_bincount,
    _spectra_from_fields_batched,
    relative_spectral_error,
    pixel_emd,
)
from analyze_continuous import band_energies


def parse_args():
    parser = argparse.ArgumentParser(
        description="Val-window quantized-recon floor for a VQ-VAE tokenizer")
    parser.add_argument("--vqvae_dir", type=str, required=True,
                        help="VQ-VAE checkpoint dir (multi-scale or flat)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, default="omega")
    parser.add_argument("--sample_start", type=int, default=20000,
                        help="Val window start (project convention)")
    parser.add_argument("--sample_stop", type=int, default=22000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="gust2-analysis")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default="recon-floor")
    parser.add_argument("--wandb_dir", type=str, default=None)
    return parser.parse_args()


def plot_spectrum_pair(k_centers, gt_spec, recon_spec, spectrum_type, ylabel,
                       output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    valid = k_centers > 0
    ax.loglog(k_centers[valid], gt_spec[valid], "b-", lw=2, label="Ground Truth")
    ax.loglog(k_centers[valid], recon_spec[valid], "g--", lw=2,
              label="VQ-VAE recon")
    ax.set_xlabel("Wavenumber |k|")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Time-Averaged {spectrum_type} (val recon floor)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"  Saved {spectrum_type} to {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading VQ-VAE from {args.vqvae_dir}...")
    encoder, decoder, vq, ema_state, arch_config = load_checkpoint(
        args.vqvae_dir, jax.random.PRNGKey(0))
    encoder = eqx.nn.inference_mode(encoder)
    decoder = eqx.nn.inference_mode(decoder)
    vq = eqx.nn.inference_mode(vq)
    codebook = ema_state.codebook
    scales = tuple(arch_config["scales"])
    tokens_per_frame = int(sum(s * s for s in scales))
    print(f"  scales={scales} ({tokens_per_frame} tokens/frame), "
          f"K={codebook.shape[0]}")

    @eqx.filter_jit
    def recon_batch(xb):
        """(B, 1, H, W) f32 -> recon (B, 1, H, W) f32 + per-scale indices."""
        def single(x):
            z_e = encoder(x)
            z_q, all_indices, _partials, _commit, _all_z = vq(z_e, codebook)
            return decoder(z_q), all_indices
        return jax.vmap(single)(xb)

    print(f"Loading GT frames {args.sample_start}-{args.sample_stop}...")
    with h5py.File(args.data_path, "r") as f:
        gt = np.asarray(f[f"fields/{args.field}"]
                        [args.sample_start:args.sample_stop], dtype=np.float32)
    M, H, W = gt.shape

    print(f"Reconstructing {M} frames (batch {args.batch_size})...")
    recon = np.zeros_like(gt)
    seen_codes = set()
    for i in range(0, M, args.batch_size):
        xb = jnp.asarray(gt[i:i + args.batch_size][:, None], dtype=jnp.float32)
        rb, idx = recon_batch(xb)
        recon[i:i + args.batch_size] = np.asarray(rb)[:, 0]
        for scale_idx in idx:
            seen_codes.update(np.unique(np.asarray(scale_idx)).tolist())
        if (i // args.batch_size + 1) % 10 == 0:
            print(f"  {min(i + args.batch_size, M)}/{M} frames")
    unique_codes_val = len(seen_codes)

    # --- Spectra ---
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    bin_index, bin_counts, n_bins = setup_radial_bincount(bin_masks)

    print("Computing spectra...")
    gt_tke_f, gt_ens_f = _spectra_from_fields_batched(
        gt, Kx, Ky, Ksq, bin_index, bin_counts, n_bins)
    rc_tke_f, rc_ens_f = _spectra_from_fields_batched(
        recon, Kx, Ky, Ksq, bin_index, bin_counts, n_bins)
    gt_tke, gt_ens = gt_tke_f.mean(axis=0), gt_ens_f.mean(axis=0)
    rc_tke, rc_ens = rc_tke_f.mean(axis=0), rc_ens_f.mean(axis=0)
    gt_bands = band_energies(gt_tke, n_bins)
    rc_bands = band_energies(rc_tke, n_bins)

    # --- Pixel distributions ---
    gt_pixels, rc_pixels = gt.ravel(), recon.ravel()
    bin_min, bin_max = gt_pixels.min(), gt_pixels.max()
    margin = (bin_max - bin_min) * 0.01
    hbins = np.linspace(bin_min - margin, bin_max + margin, 101)
    hist_bin_centers = 0.5 * (hbins[:-1] + hbins[1:])
    gt_hist = np.histogram(gt_pixels, bins=hbins, density=True)[0]
    rc_hist = np.histogram(rc_pixels, bins=hbins, density=True)[0]

    # --- Metrics ---
    metrics = {
        "mse/vqvae": float(np.mean((recon - gt) ** 2)),
        "emd/vqvae": pixel_emd(rc_pixels, gt_pixels),
        "tke_rse/vqvae": relative_spectral_error(rc_tke, gt_tke),
        "enstrophy_rse/vqvae": relative_spectral_error(rc_ens, gt_ens),
        "highk_retention/vqvae": float(rc_bands[2] / max(gt_bands[2], 1e-30)),
        "midk_retention/vqvae": float(rc_bands[1] / max(gt_bands[1], 1e-30)),
        "unique_codes_val": unique_codes_val,
        "tokens_per_frame": tokens_per_frame,
        "n_scales": len(scales),
        "n_frames": int(M),
    }

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(args.output_dir, "recon_floor_data.npz"),
             k_centers=k_centers, gt_tke=gt_tke, recon_tke=rc_tke,
             gt_enstrophy=gt_ens, recon_enstrophy=rc_ens,
             hist_bin_centers=hist_bin_centers, gt_hist=gt_hist,
             recon_hist=rc_hist, scales=np.array(scales),
             sample_start=args.sample_start, sample_stop=args.sample_stop)

    # --- Plots ---
    plot_spectrum_pair(k_centers, gt_tke, rc_tke, "TKE Spectrum", "E(k)",
                       os.path.join(args.output_dir, "tke_spectrum.png"))
    plot_spectrum_pair(k_centers, gt_ens, rc_ens, "Enstrophy Spectrum", "Z(k)",
                       os.path.join(args.output_dir, "enstrophy_spectrum.png"))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(hist_bin_centers, gt_hist, "b-", lw=2, label="Ground Truth")
    ax.semilogy(hist_bin_centers, rc_hist, "g--", lw=2, label="VQ-VAE recon")
    ax.set_xlabel("Vorticity")
    ax.set_ylabel("Density")
    ax.set_title("Pixel distribution (val recon floor)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_dir, "pixel_histogram.png"), dpi=100)
    plt.close(fig)

    # GT / recon side-by-side (constrained_layout; shared colorbar — do NOT
    # pass bbox_inches='tight' when saving)
    vmin, vmax = gt[0].min(), gt[0].max()
    snap, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, field, label in [(axes[0], gt[0], "Ground Truth"),
                             (axes[1], recon[0], "VQ-VAE recon")]:
        im = ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       origin="lower")
        ax.set_title(f"{label} (t={args.sample_start})", fontsize=12)
        ax.axis("off")
    snap.colorbar(im, ax=axes, shrink=0.8, label="Vorticity")
    snap.savefig(os.path.join(args.output_dir, "snapshot_t0.png"), dpi=100)
    plt.close(snap)

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6g}" if isinstance(v, float) else f"  {k}: {v}")

    if WANDB_AVAILABLE:
        if args.wandb_dir is not None:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = args.wandb_dir
        run = wandb.init(project=args.wandb_project, name=args.wandb_name,
                         group=args.wandb_group,
                         config={**vars(args), "scales": list(scales),
                                 "tokens_per_frame": tokens_per_frame})
        wandb.log({
            **metrics,
            "tke_spectrum": wandb.Image(
                os.path.join(args.output_dir, "tke_spectrum.png")),
            "enstrophy_spectrum": wandb.Image(
                os.path.join(args.output_dir, "enstrophy_spectrum.png")),
            "pixel_histogram": wandb.Image(
                os.path.join(args.output_dir, "pixel_histogram.png")),
            "snapshot": wandb.Image(
                os.path.join(args.output_dir, "snapshot_t0.png")),
        })
        run.finish()
        print("Logged to wandb")

    print("Done.")


if __name__ == "__main__":
    main()
