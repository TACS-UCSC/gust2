"""Spectral and histogram analysis of NSP rollout quality.

Compares three sources in pixel space:
  1. Ground truth: raw validation vorticity fields from HDF5
  2. VQ-VAE reconstruction: ground truth tokens decoded through VQ-VAE
  3. NSP predictions: autoregressive rollout tokens decoded through VQ-VAE

Produces time-averaged TKE and enstrophy spectra, pixel value histograms,
relative spectral error, and earth mover's distance.

Usage:
    python analyze_rollout.py \
        --rollout_dir experiments/rollouts/small-sc341-nsp-medium \
        --vqvae_dir experiments/vqvae/small-sc341 \
        --data_path data_lowres/output.h5 \
        --output_dir experiments/analysis/small-sc341-nsp-medium
"""

import argparse
import json
import os

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from tokenizer import load_checkpoint, reconstruct_from_indices, unflatten_to_scales

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectral analysis of NSP rollout vs ground truth")
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="Directory with rollout_tokens.npz")
    parser.add_argument("--vqvae_dir", type=str, required=True,
                        help="VQ-VAE checkpoint directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="HDF5 data file")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name under /fields/")
    parser.add_argument("--sample_start", type=int, default=20000,
                        help="Where validation data starts in HDF5")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="gust2-analysis")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================


def load_rollout_data(rollout_dir):
    """Load rollout_tokens.npz."""
    path = os.path.join(rollout_dir, "rollout_tokens.npz")
    data = dict(np.load(path, allow_pickle=True))
    return {
        "rollout_indices": data["rollout_indices"],
        "gt_indices": data["gt_indices"],
        "scales": tuple(int(s) for s in data["scales"]),
        "start_frame": int(data["start_frame"]),
        "n_steps": int(data["n_steps"]),
        "new_to_old": jnp.array(data["new_to_old"]),
    }


def load_raw_gt(data_path, field, sample_start, start_frame, n_frames):
    """Load raw vorticity fields from HDF5.

    Returns: (n_frames, 1, 256, 256) float32 array.
    """
    offset = sample_start + start_frame
    with h5py.File(data_path, "r") as f:
        raw = f[f"fields/{field}"][offset:offset + n_frames].astype(np.float32)
    return raw[:, None, :, :]


# =============================================================================
# VQ-VAE decoding
# =============================================================================


@eqx.filter_jit
def _vmap_decode(decoder, vq, codebook, new_to_old, flat_indices_batch,
                 scales):
    """Batched decode: compact flat indices -> (B, 1, 256, 256) fields."""
    decoder = eqx.nn.inference_mode(decoder)
    vq = eqx.nn.inference_mode(vq)

    def decode_single(flat_indices):
        indices_list = unflatten_to_scales(flat_indices, scales)
        original_indices = [new_to_old[idx] for idx in indices_list]
        z_q = reconstruct_from_indices(original_indices, codebook, vq)
        return decoder(z_q)

    return jax.vmap(decode_single)(flat_indices_batch)


def decode_all_tokens(indices_array, decoder, vq, codebook, new_to_old,
                      scales, batch_size):
    """Decode all frames from compact token indices.

    Args:
        indices_array: (N, tokens_per_frame) compact indices
    Returns:
        fields: (N, 1, 256, 256) numpy array
    """
    N = indices_array.shape[0]
    all_fields = []

    for i in range(0, N, batch_size):
        batch = jnp.array(indices_array[i:i + batch_size])
        decoded = _vmap_decode(decoder, vq, codebook, new_to_old, batch,
                               scales)
        all_fields.append(np.array(decoded))
        done = min(i + batch_size, N)
        if (i // batch_size + 1) % 5 == 0 or done == N:
            print(f"    {done}/{N} frames")

    return np.concatenate(all_fields, axis=0)


# =============================================================================
# Spectral analysis
# =============================================================================


def setup_spectral_analysis(H, W):
    """Set up wavenumber grids and precompute radial bin masks."""
    kx = np.fft.fftfreq(W, d=1.0) * 2 * np.pi
    ky = np.fft.fftfreq(H, d=1.0) * 2 * np.pi
    Kx, Ky = np.meshgrid(kx, ky)
    Ksq = Kx**2 + Ky**2

    k_mag = np.sqrt(Ksq)
    k_max = np.max(k_mag)
    n_bins = min(H // 2, W // 2)
    k_bins = np.linspace(0, k_max, n_bins)
    k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])

    bin_masks = []
    for i in range(len(k_centers)):
        if i == 0:
            bin_masks.append(k_mag <= k_bins[1])
        else:
            bin_masks.append((k_mag > k_bins[i]) & (k_mag <= k_bins[i + 1]))

    return Kx, Ky, Ksq, k_centers, bin_masks


def radial_average(density_2d, bin_masks):
    """Radially average a 2D spectral density field."""
    spectrum = np.zeros(len(bin_masks))
    for i, mask in enumerate(bin_masks):
        if np.any(mask):
            spectrum[i] = np.mean(density_2d[mask])
    return spectrum


def compute_tke_spectrum(omega, Kx, Ky, Ksq, bin_masks):
    """Compute radially-averaged TKE spectrum E(k) from vorticity."""
    omega_hat = np.fft.fft2(omega)
    psi_hat = np.zeros_like(omega_hat, dtype=complex)
    nonzero = Ksq > 0
    psi_hat[nonzero] = omega_hat[nonzero] / Ksq[nonzero]

    u_hat = 1j * Ky * psi_hat
    v_hat = 1j * Kx * psi_hat
    KE_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)

    return radial_average(KE_density, bin_masks)


def compute_enstrophy_spectrum(omega, bin_masks):
    """Compute radially-averaged enstrophy spectrum Z(k) from vorticity."""
    omega_hat = np.fft.fft2(omega)
    enstrophy_density = 0.5 * np.abs(omega_hat)**2
    return radial_average(enstrophy_density, bin_masks)


# =============================================================================
# Metrics
# =============================================================================


def relative_spectral_error(pred_spectrum, gt_spectrum, eps=1e-10):
    """Mean relative error across wavenumber bins."""
    valid = gt_spectrum > eps
    if not np.any(valid):
        return float('nan')
    rel_err = np.abs(pred_spectrum[valid] - gt_spectrum[valid]) / gt_spectrum[valid]
    return float(np.mean(rel_err))


def pixel_emd(pred_pixels, gt_pixels, n_subsample=1_000_000, seed=42):
    """Wasserstein-1 distance between pixel value distributions."""
    rng = np.random.default_rng(seed)
    if len(gt_pixels) > n_subsample:
        gt_sub = rng.choice(gt_pixels, n_subsample, replace=False)
        pred_sub = rng.choice(pred_pixels, n_subsample, replace=False)
    else:
        gt_sub, pred_sub = gt_pixels, pred_pixels
    return float(wasserstein_distance(gt_sub, pred_sub))


# =============================================================================
# Plotting
# =============================================================================


def plot_spectrum(k_centers, gt_spec, vqvae_spec, nsp_spec,
                  spectrum_type, ylabel, output_path):
    """Plot time-averaged spectral comparison (log-log, 3 curves)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_k = k_centers > 0

    for spec, label, color, ls in [
        (gt_spec, "Ground Truth", "blue", "-"),
        (vqvae_spec, "VQ-VAE", "green", "--"),
        (nsp_spec, "NSP", "red", ":"),
    ]:
        mask = valid_k & (spec > 0)
        if np.any(mask):
            ax.loglog(k_centers[mask], spec[mask],
                      label=label, color=color, linestyle=ls,
                      alpha=0.8, linewidth=2)

    ax.set_title(f"Time-Averaged {spectrum_type} Spectrum", fontsize=14)
    ax.set_xlabel("Wavenumber |k|", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved {spectrum_type} spectrum to {output_path}")
    return fig


def plot_histogram(gt_pixels, vqvae_pixels, nsp_pixels, output_path,
                   n_bins=100):
    """Plot density-normalized pixel value histograms (3 curves)."""
    bin_min = np.min(gt_pixels)
    bin_max = np.max(gt_pixels)
    margin = (bin_max - bin_min) * 0.01
    bins = np.linspace(bin_min - margin, bin_max + margin, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(10, 6))

    for pixels, label, color, ls in [
        (gt_pixels, "Ground Truth", "blue", "-"),
        (vqvae_pixels, "VQ-VAE", "green", "--"),
        (nsp_pixels, "NSP", "red", ":"),
    ]:
        hist, _ = np.histogram(pixels, bins=bins, density=True)
        ax.step(bin_centers, hist, where="mid",
                label=label, color=color, linestyle=ls,
                linewidth=2, alpha=0.8)

    ax.set_xlabel("Pixel Value (Vorticity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pixel Value Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved pixel histogram to {output_path}")
    return fig


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load rollout data ---
    print(f"Loading rollout from {args.rollout_dir}...")
    rollout = load_rollout_data(args.rollout_dir)
    n_frames = rollout["n_steps"] + 1
    scales = rollout["scales"]
    new_to_old = rollout["new_to_old"]
    print(f"  {n_frames} frames, scales={list(scales)}, "
          f"start_frame={rollout['start_frame']}")

    # --- Load VQ-VAE ---
    print(f"Loading VQ-VAE from {args.vqvae_dir}...")
    key = jax.random.PRNGKey(args.seed)
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook
    print(f"  Codebook: {codebook.shape}")

    # --- Load raw ground truth from HDF5 ---
    print(f"Loading raw GT from {args.data_path} "
          f"(frames {args.sample_start + rollout['start_frame']}"
          f"-{args.sample_start + rollout['start_frame'] + n_frames})...")
    raw_gt = load_raw_gt(args.data_path, args.field,
                         args.sample_start, rollout["start_frame"], n_frames)
    print(f"  Loaded {raw_gt.shape[0]} frames")

    # --- Decode VQ-VAE reconstruction (gt tokens -> pixels) ---
    print("Decoding VQ-VAE reconstructions...")
    vqvae_fields = decode_all_tokens(
        rollout["gt_indices"], decoder, vq, codebook,
        new_to_old, scales, args.batch_size)

    # --- Decode NSP predictions (rollout tokens -> pixels) ---
    print("Decoding NSP predictions...")
    nsp_fields = decode_all_tokens(
        rollout["rollout_indices"], decoder, vq, codebook,
        new_to_old, scales, args.batch_size)

    # --- Spectral analysis ---
    print("Computing spectra...")
    H, W = 256, 256
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    n_bins = len(k_centers)

    gt_tke = np.zeros(n_bins)
    gt_ens = np.zeros(n_bins)
    vqvae_tke = np.zeros(n_bins)
    vqvae_ens = np.zeros(n_bins)
    nsp_tke = np.zeros(n_bins)
    nsp_ens = np.zeros(n_bins)

    for i in range(n_frames):
        gt_field = raw_gt[i, 0]
        vq_field = vqvae_fields[i, 0]
        ns_field = nsp_fields[i, 0]

        gt_tke += compute_tke_spectrum(gt_field, Kx, Ky, Ksq, bin_masks)
        gt_ens += compute_enstrophy_spectrum(gt_field, bin_masks)
        vqvae_tke += compute_tke_spectrum(vq_field, Kx, Ky, Ksq, bin_masks)
        vqvae_ens += compute_enstrophy_spectrum(vq_field, bin_masks)
        nsp_tke += compute_tke_spectrum(ns_field, Kx, Ky, Ksq, bin_masks)
        nsp_ens += compute_enstrophy_spectrum(ns_field, bin_masks)

        if (i + 1) % 500 == 0 or i == n_frames - 1:
            print(f"  {i + 1}/{n_frames} frames")

    gt_tke /= n_frames
    gt_ens /= n_frames
    vqvae_tke /= n_frames
    vqvae_ens /= n_frames
    nsp_tke /= n_frames
    nsp_ens /= n_frames

    # --- Pixel distributions ---
    print("Computing pixel distributions...")
    gt_pixels = raw_gt[:, 0].ravel()
    vqvae_pixels = vqvae_fields[:, 0].ravel()
    nsp_pixels = nsp_fields[:, 0].ravel()

    # --- Metrics ---
    print("Computing metrics...")
    metrics = {
        "n_frames": n_frames,
        "scales": list(scales),
        "start_frame": rollout["start_frame"],
        "tke_rse_vqvae": relative_spectral_error(vqvae_tke, gt_tke),
        "tke_rse_nsp": relative_spectral_error(nsp_tke, gt_tke),
        "enstrophy_rse_vqvae": relative_spectral_error(vqvae_ens, gt_ens),
        "enstrophy_rse_nsp": relative_spectral_error(nsp_ens, gt_ens),
        "emd_vqvae": pixel_emd(vqvae_pixels, gt_pixels),
        "emd_nsp": pixel_emd(nsp_pixels, gt_pixels),
        "k_centers": k_centers.tolist(),
        "tke_gt": gt_tke.tolist(),
        "tke_vqvae": vqvae_tke.tolist(),
        "tke_nsp": nsp_tke.tolist(),
        "enstrophy_gt": gt_ens.tolist(),
        "enstrophy_vqvae": vqvae_ens.tolist(),
        "enstrophy_nsp": nsp_ens.tolist(),
    }

    # --- Plots ---
    print("Saving plots...")
    tke_fig = plot_spectrum(
        k_centers, gt_tke, vqvae_tke, nsp_tke,
        "TKE", "E(k)", os.path.join(args.output_dir, "tke_spectrum.png"))
    ens_fig = plot_spectrum(
        k_centers, gt_ens, vqvae_ens, nsp_ens,
        "Enstrophy", "Z(k)",
        os.path.join(args.output_dir, "enstrophy_spectrum.png"))
    hist_fig = plot_histogram(
        gt_pixels, vqvae_pixels, nsp_pixels,
        os.path.join(args.output_dir, "pixel_histogram.png"))

    # --- Save metrics ---
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # --- Wandb ---
    if WANDB_AVAILABLE:
        if args.wandb_dir is not None:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = args.wandb_dir
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "rollout_dir": args.rollout_dir,
                "vqvae_dir": args.vqvae_dir,
                "n_frames": n_frames,
                "scales": list(scales),
            },
        )
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)
        wandb.log({
            "tke_rse/vqvae": metrics["tke_rse_vqvae"],
            "tke_rse/nsp": metrics["tke_rse_nsp"],
            "enstrophy_rse/vqvae": metrics["enstrophy_rse_vqvae"],
            "enstrophy_rse/nsp": metrics["enstrophy_rse_nsp"],
            "emd/vqvae": metrics["emd_vqvae"],
            "emd/nsp": metrics["emd_nsp"],
            "tke_spectrum": wandb.Image(tke_fig),
            "enstrophy_spectrum": wandb.Image(ens_fig),
            "pixel_histogram": wandb.Image(hist_fig),
        })
        wandb.finish()
        print("  Logged to wandb")

    plt.close("all")

    # --- Summary ---
    print(f"\nResults ({n_frames} frames):")
    print(f"  TKE RSE:       VQ-VAE={metrics['tke_rse_vqvae']:.4f}  "
          f"NSP={metrics['tke_rse_nsp']:.4f}")
    print(f"  Enstrophy RSE: VQ-VAE={metrics['enstrophy_rse_vqvae']:.4f}  "
          f"NSP={metrics['enstrophy_rse_nsp']:.4f}")
    print(f"  Pixel EMD:     VQ-VAE={metrics['emd_vqvae']:.6f}  "
          f"NSP={metrics['emd_nsp']:.6f}")


if __name__ == "__main__":
    main()
