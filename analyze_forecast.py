"""Short-horizon forecast-skill analysis of NSP rollouts.

Companion to analyze_rollout.py. Where analyze_rollout measures the LONG-RUN
climate (time-averaged spectra/EMD over a 2000-step free rollout from one IC),
this measures SHORT-TERM forecast skill: free-running error at lead times
k in {1, 2, 5, 10}, averaged over an ENSEMBLE OF DISTINCT INITIAL CONDITIONS
spread across the validation set.

Input: a forecast-mode rollout (rollout_nsp.py --n_ics N), whose
rollout_tokens.npz has rank-3 arrays (N_ics, n_steps+1, tokens) where the N
axis indexes DISTINCT start frames (not sampling seeds) and the time axis is
lead time. cfg_meta.json carries forecast_mode/n_ics/start_frames.

For each lead time k we slice the time axis at index k and, using the SAME
metric definitions as analyze_rollout (just pooled/averaged across the IC
ensemble at fixed lead instead of across all time), compute:
  - pixel-EMD: Wasserstein-1 between the pooled lead-k predicted pixels and
    the matched raw-GT pixels.
  - TKE / enstrophy relative-spectral-error: IC-averaged lead-k predicted
    spectrum vs IC-averaged lead-k GT spectrum.
  - a VQ-VAE floor (decode the lead-k GT tokens vs raw GT).

Outputs metrics.json (per_horizon dict + pooled VQ floor), analysis_data.npz
(per-horizon spectra), per-horizon spectrum PNGs, and wandb scalars namespaced
by horizon (emd/nsp/k<k>, tke_rse/nsp/k<k>, ...).

Usage:
    python analyze_forecast.py \
        --rollout_dir experiments/scaling-forecast/small-sc341-nsp-s13/T1p0/rollout \
        --vqvae_dir experiments/vqvae/small-sc341 \
        --data_path data_lowres/output.h5 \
        --horizons 1,2,5,10 \
        --output_dir experiments/scaling-forecast/small-sc341-nsp-s13/T1p0/analysis
"""

import argparse
import json
import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import numpy as np

from tokenizer import load_checkpoint

# Reuse the metric / decode / spectral / plotting helpers verbatim — they are
# pure module-level functions with no global state.
from analyze_rollout import (
    decode_all_tokens,
    setup_spectral_analysis,
    compute_tke_spectrum,
    compute_enstrophy_spectrum,
    relative_spectral_error,
    pixel_emd,
    plot_spectrum,
)

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
        description="Short-horizon forecast-skill analysis of NSP rollouts")
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="Directory with a forecast-mode rollout_tokens.npz")
    parser.add_argument("--vqvae_dir", type=str, required=True,
                        help="VQ-VAE checkpoint directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="HDF5 data file")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name under /fields/")
    parser.add_argument("--sample_start", type=int, default=20000,
                        help="Where validation data starts in HDF5 (must match "
                             "the offset the val tokens were produced with)")
    parser.add_argument("--horizons", type=str, default="1,2,5,10",
                        help="Comma-separated lead times to evaluate")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    # Wandb
    parser.add_argument("--wandb_project", type=str,
                        default="gust2-forecast-analysis")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    return parser.parse_args()


# =============================================================================
# Data loading
# =============================================================================


def load_forecast_rollout(rollout_dir):
    """Load a forecast-mode rollout_tokens.npz (rank-3, distinct ICs).

    Returns a dict with rollout_indices/gt_indices (N_ics, T+1, tokens),
    start_frames (N_ics,), scales tuple, new_to_old, n_steps, n_ics.
    """
    path = os.path.join(rollout_dir, "rollout_tokens.npz")
    data = dict(np.load(path, allow_pickle=True))

    rollout_indices = np.asarray(data["rollout_indices"])
    gt_indices = np.asarray(data["gt_indices"])
    if rollout_indices.ndim != 3:
        raise SystemExit(
            f"analyze_forecast expects a forecast-mode rollout with rank-3 "
            f"rollout_indices (N_ics, T+1, tokens); got shape "
            f"{rollout_indices.shape}. Run rollout_nsp.py with --n_ics > 0.")

    # start_frames is the per-IC source of truth (saved when N>1).
    if "start_frames" in data:
        start_frames = np.asarray(data["start_frames"]).astype(np.int64)
    else:
        raise SystemExit(
            "rollout_tokens.npz has no per-IC 'start_frames' array; this does "
            "not look like a forecast-mode rollout.")

    return {
        "rollout_indices": rollout_indices,
        "gt_indices": gt_indices,
        "start_frames": start_frames,
        "scales": tuple(int(s) for s in data["scales"]),
        "new_to_old": jnp.array(data["new_to_old"]),
        "n_steps": int(data["n_steps"]),
        "n_ics": int(rollout_indices.shape[0]),
    }


def load_raw_gt_gathered(data_path, field, offsets):
    """Read raw vorticity frames at arbitrary (non-contiguous) HDF5 offsets.

    h5py fancy indexing needs strictly-increasing unique indices, so we sort,
    de-dup, read, then scatter back to the requested order.

    Args:
        offsets: (M,) integer HDF5 frame indices.
    Returns:
        (M, 256, 256) float32 in the order of `offsets`.
    """
    offsets = np.asarray(offsets, dtype=np.int64)
    order = np.argsort(offsets, kind="stable")
    sorted_off = offsets[order]
    uniq, inv = np.unique(sorted_off, return_inverse=True)
    with h5py.File(data_path, "r") as f:
        raw_uniq = f[f"fields/{field}"][uniq].astype(np.float32)  # (U, H, W)
    raw_sorted = raw_uniq[inv]                 # (M, H, W) sorted-with-dups
    out = np.empty_like(raw_sorted)
    out[order] = raw_sorted                    # unsort back to offsets order
    return out


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load forecast rollout ---
    print(f"Loading forecast rollout from {args.rollout_dir}...")
    rollout = load_forecast_rollout(args.rollout_dir)
    rollout_indices = rollout["rollout_indices"]   # (N, T+1, tok)
    gt_indices = rollout["gt_indices"]
    start_frames = rollout["start_frames"]         # (N,)
    scales = rollout["scales"]
    new_to_old = rollout["new_to_old"]
    n_steps = rollout["n_steps"]
    N = rollout["n_ics"]
    print(f"  {N} ICs, n_steps={n_steps}, scales={list(scales)}, "
          f"start_frames {int(start_frames[0])}..{int(start_frames[-1])}")

    horizons = sorted({int(x) for x in args.horizons.split(",") if x.strip()})
    horizons = [k for k in horizons if 1 <= k <= n_steps]
    if not horizons:
        raise SystemExit(
            f"No valid horizons in {args.horizons!r} for n_steps={n_steps}.")
    print(f"  Horizons: {horizons}")

    # --- Load VQ-VAE ---
    print(f"Loading VQ-VAE from {args.vqvae_dir}...")
    key = jax.random.PRNGKey(args.seed)
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook
    print(f"  Codebook: {codebook.shape}")

    # --- Spectral setup ---
    H, W = 256, 256
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)

    def avg_tke(fields_2d):   # fields_2d: (M, H, W) -> mean radial TKE spectrum
        acc = np.zeros(len(k_centers))
        for f in fields_2d:
            acc += compute_tke_spectrum(f, Kx, Ky, Ksq, bin_masks)
        return acc / len(fields_2d)

    def avg_ens(fields_2d):
        acc = np.zeros(len(k_centers))
        for f in fields_2d:
            acc += compute_enstrophy_spectrum(f, bin_masks)
        return acc / len(fields_2d)

    # Pooled accumulators for a stable (≈horizon-independent) VQ-VAE floor.
    pool_raw_px, pool_vq_px = [], []
    pool_gt_tke, pool_gt_ens = [], []
    pool_vq_tke, pool_vq_ens = [], []

    per_horizon = {}
    wandb_scalars = {}
    spectrum_figs = {}
    npz_arrays = {"k_centers": k_centers}

    # --- Per-horizon forecast error ---
    for k in horizons:
        print(f"\n=== Lead time k={k} ===")
        pred_k_tokens = rollout_indices[:, k, :]   # (N, tok)
        gt_k_tokens = gt_indices[:, k, :]          # (N, tok)
        offsets_k = args.sample_start + start_frames + k

        print(f"  Reading {N} raw GT frames at lead {k}...")
        raw_gt_k = load_raw_gt_gathered(args.data_path, args.field, offsets_k)

        print(f"  Decoding {N} predicted lead-{k} frames...")
        pred_k = decode_all_tokens(pred_k_tokens, decoder, vq, codebook,
                                   new_to_old, scales, args.batch_size)
        print(f"  Decoding {N} VQ-recon lead-{k} frames (floor)...")
        vqvae_k = decode_all_tokens(gt_k_tokens, decoder, vq, codebook,
                                    new_to_old, scales, args.batch_size)

        pred_2d = pred_k[:, 0]      # (N, H, W)
        vqvae_2d = vqvae_k[:, 0]

        # Spectra: IC-average at fixed lead.
        gt_tke_k = avg_tke(raw_gt_k);  gt_ens_k = avg_ens(raw_gt_k)
        nsp_tke_k = avg_tke(pred_2d);  nsp_ens_k = avg_ens(pred_2d)
        vq_tke_k = avg_tke(vqvae_2d);  vq_ens_k = avg_ens(vqvae_2d)

        # Pixel-EMD: pooled lead-k pred pixels vs matched raw-GT pixels.
        emd_nsp_k = pixel_emd(pred_2d.ravel(), raw_gt_k.ravel(), seed=args.seed)
        emd_vqvae_k = pixel_emd(vqvae_2d.ravel(), raw_gt_k.ravel(),
                                seed=args.seed)

        tke_rse_nsp_k = relative_spectral_error(nsp_tke_k, gt_tke_k)
        ens_rse_nsp_k = relative_spectral_error(nsp_ens_k, gt_ens_k)
        tke_rse_vqvae_k = relative_spectral_error(vq_tke_k, gt_tke_k)
        ens_rse_vqvae_k = relative_spectral_error(vq_ens_k, gt_ens_k)

        per_horizon[str(k)] = {
            "emd_nsp": emd_nsp_k,
            "tke_rse_nsp": tke_rse_nsp_k,
            "enstrophy_rse_nsp": ens_rse_nsp_k,
            "emd_vqvae": emd_vqvae_k,
            "tke_rse_vqvae": tke_rse_vqvae_k,
            "enstrophy_rse_vqvae": ens_rse_vqvae_k,
        }
        wandb_scalars.update({
            f"emd/nsp/k{k}": emd_nsp_k,
            f"tke_rse/nsp/k{k}": tke_rse_nsp_k,
            f"enstrophy_rse/nsp/k{k}": ens_rse_nsp_k,
            f"emd/vqvae/k{k}": emd_vqvae_k,
            f"tke_rse/vqvae/k{k}": tke_rse_vqvae_k,
            f"enstrophy_rse/vqvae/k{k}": ens_rse_vqvae_k,
        })
        print(f"  EMD:  NSP={emd_nsp_k:.6f}   VQ-VAE floor={emd_vqvae_k:.6f}")
        print(f"  TKE RSE: NSP={tke_rse_nsp_k:.4f}   VQ-VAE={tke_rse_vqvae_k:.4f}")

        # Per-horizon spectrum figures (GT / VQ-VAE / NSP).
        tke_fig = plot_spectrum(
            k_centers, gt_tke_k, vq_tke_k, nsp_tke_k,
            f"TKE (lead k={k})", "E(k)",
            os.path.join(args.output_dir, f"tke_spectrum_k{k}.png"))
        ens_fig = plot_spectrum(
            k_centers, gt_ens_k, vq_ens_k, nsp_ens_k,
            f"Enstrophy (lead k={k})", "Z(k)",
            os.path.join(args.output_dir, f"enstrophy_spectrum_k{k}.png"))
        spectrum_figs[f"tke_spectrum/k{k}"] = tke_fig
        spectrum_figs[f"enstrophy_spectrum/k{k}"] = ens_fig

        npz_arrays.update({
            f"tke_gt_k{k}": gt_tke_k, f"tke_nsp_k{k}": nsp_tke_k,
            f"tke_vqvae_k{k}": vq_tke_k,
            f"enstrophy_gt_k{k}": gt_ens_k, f"enstrophy_nsp_k{k}": nsp_ens_k,
            f"enstrophy_vqvae_k{k}": vq_ens_k,
        })

        pool_raw_px.append(raw_gt_k.ravel())
        pool_vq_px.append(vqvae_2d.ravel())
        pool_gt_tke.append(gt_tke_k);  pool_gt_ens.append(gt_ens_k)
        pool_vq_tke.append(vq_tke_k);  pool_vq_ens.append(vq_ens_k)

    # --- Pooled VQ-VAE floor (over all evaluated horizons) ---
    emd_vqvae_pool = pixel_emd(np.concatenate(pool_vq_px),
                               np.concatenate(pool_raw_px), seed=args.seed)
    tke_rse_vqvae_pool = relative_spectral_error(
        np.mean(pool_vq_tke, axis=0), np.mean(pool_gt_tke, axis=0))
    ens_rse_vqvae_pool = relative_spectral_error(
        np.mean(pool_vq_ens, axis=0), np.mean(pool_gt_ens, axis=0))

    metrics = {
        "n_ics": N,
        "horizons": horizons,
        "scales": list(scales),
        "start_frames": start_frames.tolist(),
        # Pooled VQ-VAE floor (≈horizon-independent dashed line for plots).
        "emd_vqvae": emd_vqvae_pool,
        "tke_rse_vqvae": tke_rse_vqvae_pool,
        "enstrophy_rse_vqvae": ens_rse_vqvae_pool,
        "per_horizon": per_horizon,
    }

    # --- Save plot data + metrics ---
    npz_path = os.path.join(args.output_dir, "analysis_data.npz")
    np.savez_compressed(npz_path, horizons=np.array(horizons), **npz_arrays)
    print(f"\n  Saved plot data to {npz_path}")

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
                "n_ics": N,
                "horizons": horizons,
                "scales": list(scales),
            },
        )
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)
        log_dict = dict(wandb_scalars)
        log_dict.update({
            "emd/vqvae": emd_vqvae_pool,
            "tke_rse/vqvae": tke_rse_vqvae_pool,
            "enstrophy_rse/vqvae": ens_rse_vqvae_pool,
            "n_ics": N,
        })
        for name, fig in spectrum_figs.items():
            log_dict[name] = wandb.Image(fig)
        wandb.log(log_dict)
        wandb.finish()
        print("  Logged to wandb")

    import matplotlib.pyplot as plt
    plt.close("all")

    # --- Summary ---
    print(f"\nForecast skill ({N} ICs):")
    print(f"  {'k':>4} {'EMD_nsp':>10} {'EMD_vq':>10} "
          f"{'TKE_nsp':>9} {'Ens_nsp':>9}")
    for k in horizons:
        m = per_horizon[str(k)]
        print(f"  {k:>4} {m['emd_nsp']:>10.6f} {m['emd_vqvae']:>10.6f} "
              f"{m['tke_rse_nsp']:>9.4f} {m['enstrophy_rse_nsp']:>9.4f}")
    print(f"  pooled VQ-VAE floor: EMD={emd_vqvae_pool:.6f}  "
          f"TKE_RSE={tke_rse_vqvae_pool:.4f}")


if __name__ == "__main__":
    main()
