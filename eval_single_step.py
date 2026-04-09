"""Single-step evaluation of NSP models.

For each consecutive validation frame pair (t0, t1), predicts t1 from t0
using greedy decoding (one step, no autoregressive rollout). Reports:

  1. Cross-entropy loss (NSP's own loss, averaged over all pairs)
  2. Pixel RMSE (decode predicted vs GT tokens through VQ-VAE, L2 in pixel space)

Outputs:
  - eval_single_step.json: cross_entropy, pixel_rmse, per-scale breakdowns

Usage:
    python eval_single_step.py \
        --checkpoint_dir experiments/ar/small-sc341-nsp-medium \
        --tokens_path experiments/tokens/small-sc341-val.npz \
        --vqvae_dir experiments/vqvae/small-sc341 \
        --data_path data_lowres/output.h5 \
        --output_dir experiments/eval/small-sc341-nsp-medium
"""

import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np

import matplotlib.pyplot as plt

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from analyze_rollout import (
    setup_spectral_analysis,
    compute_tke_spectrum,
    compute_enstrophy_spectrum,
    compute_histograms,
    relative_spectral_error,
    pixel_emd,
    plot_spectrum,
    plot_histogram,
    plot_snapshot,
)
from nsp_model import (
    NSPConfig, create_nsp_model, forward_teacher_forced,
    build_teacher_forced_mask,
)
from tokenizer import (
    load_tokenized_data, load_checkpoint,
    reconstruct_from_indices, unflatten_to_scales,
)


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-step NSP evaluation (cross-entropy + pixel RMSE)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="NSP checkpoint directory")
    parser.add_argument("--tokens_path", type=str, required=True,
                        help="Validation tokenized .npz")
    parser.add_argument("--vqvae_dir", type=str, required=True,
                        help="VQ-VAE checkpoint directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="HDF5 data file (for raw GT pixels)")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name under /fields/")
    parser.add_argument("--sample_start", type=int, default=20000,
                        help="Where validation data starts in HDF5")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for VQ-VAE decoding")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="Evaluate only first N pairs (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="gust2-eval")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    return parser.parse_args()


# =============================================================================
# Single-step prediction + cross-entropy
# =============================================================================


def make_predict_and_loss(config, scales_t0, padded_len_t0,
                          scales_t1, padded_len_t1,
                          attn_bias, scale_masks, trainable_indices):
    """Build JIT-compiled function that predicts t1 and computes cross-entropy.

    Returns: fn(model, exp_heads, t0_tokens, t1_tokens)
             -> (predicted_tokens, cross_entropy, per_scale_ce)
    """
    tokens_per_frame = config.tokens_per_frame
    tokens_t1_trunc = sum(h * w for h, w in scales_t1)

    boundaries_t0 = [0]
    for h, w in scales_t0:
        boundaries_t0.append(boundaries_t0[-1] + h * w)

    boundaries_t1 = [0]
    for h, w in scales_t1:
        boundaries_t1.append(boundaries_t1[-1] + h * w)

    boundaries_full = config.scale_boundaries

    # Per-scale loss weights (same as training: 1/sqrt(token_count), normalized)
    token_counts = [config.scales[i][0] * config.scales[i][1]
                    for i in trainable_indices]
    raw_weights = [1.0 / c ** 0.5 for c in token_counts]
    mean_w = sum(raw_weights) / len(raw_weights)
    scale_weights = {idx: w / mean_w
                     for idx, w in zip(trainable_indices, raw_weights)}

    @eqx.filter_jit
    def predict_and_loss(model, exp_heads, t0_tokens, t1_tokens):
        """Single-sample prediction + cross-entropy.

        Args:
            model, exp_heads: NSP model (inference mode)
            t0_tokens: (tokens_per_frame,) compact indices
            t1_tokens: (tokens_per_frame,) GT compact indices

        Returns:
            predicted: (tokens_per_frame,) greedy predictions
            weighted_ce: scalar weighted cross-entropy
            per_scale_ce: (n_trainable,) raw CE per scale
        """
        # Build input: [t0_padded, t1_truncated_padded]
        # For eval we teacher-force t1 context so the CE is comparable to
        # training loss: each scale sees GT for all coarser scales.
        t1_trunc = t1_tokens[:tokens_t1_trunc]
        t0_pad = jnp.pad(t0_tokens, (0, padded_len_t0 - tokens_per_frame))
        t1_pad = jnp.pad(t1_trunc, (0, padded_len_t1 - tokens_t1_trunc))
        tokens_in = jnp.concatenate([t0_pad, t1_pad])

        hidden = forward_teacher_forced(
            model, tokens_in, config,
            scales_t0, padded_len_t0,
            scales_t1, padded_len_t1,
            attn_bias,
        )

        h_t0 = hidden[:padded_len_t0, :]
        h_t1 = hidden[padded_len_t0:, :]

        predicted = jnp.zeros(tokens_per_frame, dtype=jnp.int32)
        total_ce = jnp.float32(0.0)
        per_scale_ce_list = []

        for i, scale_idx in enumerate(trainable_indices):
            h_k, w_k = config.scales[scale_idx]
            n_tokens_k = h_k * w_k

            # Source hidden states from scale k-1
            if scale_idx == 0:
                src_start = boundaries_t0[0]
                src_end = boundaries_t0[1]
                h_src, w_src = config.scales[0]
                h_source = h_t0[src_start:src_end, :]
            else:
                src_in_t1 = scale_idx - 1
                src_start = boundaries_t1[src_in_t1]
                src_end = boundaries_t1[src_in_t1 + 1]
                h_src, w_src = scales_t1[src_in_t1]
                h_source = h_t1[src_start:src_end, :]

            h_source_2d = h_source.reshape(h_src, w_src, config.n_embd)

            h_up = jax.image.resize(
                h_source_2d, (h_k, w_k, config.n_embd), method='bilinear')

            rows = jnp.arange(h_k, dtype=jnp.float32) / max(h_k - 1, 1)
            cols = jnp.arange(w_k, dtype=jnp.float32) / max(w_k - 1, 1)
            grid_r, grid_c = jnp.meshgrid(rows, cols, indexing='ij')
            coords = jnp.stack([grid_r, grid_c], axis=-1)
            pos_emb = jax.vmap(jax.vmap(exp_heads.pos_proj))(coords)

            h_flat = (h_up + pos_emb).reshape(n_tokens_k, config.n_embd)
            logits = jax.vmap(exp_heads.heads[i])(h_flat)
            logits = jnp.where(scale_masks[scale_idx][None, :], logits, -1e9)

            # Greedy prediction
            preds = jnp.argmax(logits, axis=-1)
            tgt_start = boundaries_full[scale_idx]
            tgt_end = boundaries_full[scale_idx + 1]
            predicted = predicted.at[tgt_start:tgt_end].set(preds)

            # Cross-entropy against GT
            targets = t1_tokens[tgt_start:tgt_end]
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs, targets[:, None], axis=-1
            ).squeeze(-1)
            raw_ce = -jnp.mean(target_log_probs)
            total_ce = total_ce + raw_ce * scale_weights[scale_idx]
            per_scale_ce_list.append(raw_ce)

        per_scale_ce = jnp.stack(per_scale_ce_list)
        return predicted, total_ce, per_scale_ce

    return predict_and_loss


# =============================================================================
# VQ-VAE decoding
# =============================================================================


@eqx.filter_jit
def _vmap_decode(decoder, vq, codebook, new_to_old, flat_indices_batch, scales):
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
    """Decode all frames from compact token indices -> (N, 1, 256, 256)."""
    N = indices_array.shape[0]
    all_fields = []
    for i in range(0, N, batch_size):
        batch = jnp.array(indices_array[i:i + batch_size])
        decoded = _vmap_decode(decoder, vq, codebook, new_to_old, batch, scales)
        all_fields.append(np.array(decoded))
    return np.concatenate(all_fields, axis=0)


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load tokenized validation data ---
    print(f"Loading tokens from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    scales_int = token_data["scales"]
    scale_masks = jnp.array(token_data["scale_masks"])
    new_to_old = jnp.array(token_data["new_to_old"])
    tokens_per_frame = sum(s * s for s in scales_int)
    print(f"  {len(indices)} frames, {tokens_per_frame} tokens/frame")

    # --- Load NSP model ---
    print(f"Loading NSP from {args.checkpoint_dir}...")
    state_path = os.path.join(args.checkpoint_dir, "training_state.json")
    with open(state_path) as f:
        training_state = json.load(f)
    arch = training_state["arch_config"]

    config = NSPConfig(
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
        n_embd=arch["n_embd"],
        dropout=0.0,
        rope_theta=arch.get("rope_theta", 16.0),
    )

    key = jax.random.PRNGKey(args.seed)
    model, exp_heads = create_nsp_model(token_data, config, key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "model.eqx"), model)
    exp_heads = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "exp_heads.eqx"), exp_heads)
    model = eqx.nn.inference_mode(model)
    exp_heads = eqx.nn.inference_mode(exp_heads)

    # --- Load VQ-VAE ---
    print(f"Loading VQ-VAE from {args.vqvae_dir}...")
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook

    # --- Load raw GT from HDF5 ---
    import h5py
    n_val_frames = len(indices)
    print(f"Loading raw GT from {args.data_path} "
          f"(frames {args.sample_start}-{args.sample_start + n_val_frames})...")
    with h5py.File(args.data_path, "r") as f:
        raw_gt = f[f"fields/{args.field}"][
            args.sample_start:args.sample_start + n_val_frames
        ].astype(np.float32)
    raw_gt = raw_gt[:, None, :, :]
    print(f"  Loaded {raw_gt.shape[0]} frames")

    # --- Build predict + loss function ---
    trainable_indices = config.trainable_scale_indices
    scales_t0 = config.scales
    scales_t1 = config.scales[:-1]
    tokens_t0 = sum(h * w for h, w in scales_t0)
    tokens_t1 = sum(h * w for h, w in scales_t1)
    padded_len_t0 = ((tokens_t0 + 127) // 128) * 128
    padded_len_t1 = ((tokens_t1 + 127) // 128) * 128

    attn_bias = build_teacher_forced_mask(
        scales_t0, padded_len_t0, scales_t1, padded_len_t1)

    predict_and_loss = make_predict_and_loss(
        config, scales_t0, padded_len_t0,
        scales_t1, padded_len_t1,
        attn_bias, scale_masks, trainable_indices,
    )

    # --- Evaluate all consecutive pairs ---
    n_pairs = len(indices) - 1
    if args.max_pairs is not None:
        n_pairs = min(n_pairs, args.max_pairs)
    print(f"\nEvaluating {n_pairs} consecutive pairs...")

    all_ce = []
    all_per_scale_ce = []
    all_pred_tokens = []
    all_gt_tokens = []

    log_every = max(1, n_pairs // 20)
    t_start = time.time()

    for i in range(n_pairs):
        t0 = jnp.array(indices[i])
        t1 = jnp.array(indices[i + 1])

        predicted, ce, per_scale_ce = predict_and_loss(
            model, exp_heads, t0, t1)

        all_ce.append(float(ce))
        all_per_scale_ce.append(np.array(per_scale_ce))
        all_pred_tokens.append(np.array(predicted))
        all_gt_tokens.append(np.array(t1))

        if (i + 1) % log_every == 0 or i == 0 or i == n_pairs - 1:
            elapsed = time.time() - t_start
            avg_ce = np.mean(all_ce)
            print(f"  {i+1}/{n_pairs}: avg_ce={avg_ce:.4f} "
                  f"({elapsed:.0f}s elapsed)")

    pred_tokens_arr = np.stack(all_pred_tokens)
    gt_tokens_arr = np.stack(all_gt_tokens)

    # --- Cross-entropy results ---
    mean_ce = float(np.mean(all_ce))
    per_scale_means = np.mean(np.stack(all_per_scale_ce), axis=0)

    print(f"\nCross-entropy: {mean_ce:.4f}")
    for j, idx in enumerate(trainable_indices):
        h, w = config.scales[idx]
        print(f"  Scale {h}x{w}: {per_scale_means[j]:.4f}")

    # --- Pixel RMSE ---
    print("\nDecoding predicted tokens...")
    pred_fields = decode_all_tokens(
        pred_tokens_arr, decoder, vq, codebook,
        new_to_old, scales_int, args.batch_size)

    print("Decoding GT tokens...")
    gt_decoded = decode_all_tokens(
        gt_tokens_arr, decoder, vq, codebook,
        new_to_old, scales_int, args.batch_size)

    # RMSE against raw GT pixels (frame i+1 for each pair)
    raw_gt_pairs = raw_gt[1:n_pairs + 1]

    per_sample_mse = np.mean(
        (pred_fields[:, 0] - raw_gt_pairs[:, 0]) ** 2, axis=(1, 2))
    per_sample_rmse = np.sqrt(per_sample_mse)
    mean_rmse = float(np.mean(per_sample_rmse))

    # Also compute RMSE of VQ-VAE reconstruction (the decoding floor)
    vqvae_mse = np.mean(
        (gt_decoded[:, 0] - raw_gt_pairs[:, 0]) ** 2, axis=(1, 2))
    vqvae_rmse = float(np.mean(np.sqrt(vqvae_mse)))

    per_sample_vqvae_rmse = np.sqrt(vqvae_mse)

    print(f"\nPixel RMSE (vs raw GT):")
    print(f"  NSP prediction: {mean_rmse:.6f}")
    print(f"  VQ-VAE recon:   {vqvae_rmse:.6f}")

    # --- Save per-timestep metrics ---
    per_scale_ce_arr = np.stack(all_per_scale_ce)  # (n_pairs, n_trainable)
    scale_names = [f"{config.scales[idx][0]}x{config.scales[idx][1]}"
                   for idx in trainable_indices]

    timestep_path = os.path.join(args.output_dir, "eval_per_timestep.npz")
    np.savez_compressed(timestep_path,
        cross_entropy=np.array(all_ce),             # (n_pairs,)
        per_scale_ce=per_scale_ce_arr,               # (n_pairs, n_trainable)
        scale_names=np.array(scale_names),           # (n_trainable,)
        pixel_rmse=per_sample_rmse,                  # (n_pairs,)
        vqvae_pixel_rmse=per_sample_vqvae_rmse,      # (n_pairs,)
    )
    print(f"  Saved per-timestep metrics to {timestep_path}")

    # --- Spectral analysis (single-step predictions) ---
    print("\nComputing spectra...")
    H, W = 256, 256
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    n_bins_spec = len(k_centers)

    gt_tke = np.zeros(n_bins_spec)
    gt_ens = np.zeros(n_bins_spec)
    vqvae_tke = np.zeros(n_bins_spec)
    vqvae_ens = np.zeros(n_bins_spec)
    nsp_tke = np.zeros(n_bins_spec)
    nsp_ens = np.zeros(n_bins_spec)

    log_every_spec = max(1, n_pairs // 10)
    for i in range(n_pairs):
        gt_f = raw_gt_pairs[i, 0]
        vq_f = gt_decoded[i, 0]
        ns_f = pred_fields[i, 0]

        gt_tke += compute_tke_spectrum(gt_f, Kx, Ky, Ksq, bin_masks)
        gt_ens += compute_enstrophy_spectrum(gt_f, bin_masks)
        vqvae_tke += compute_tke_spectrum(vq_f, Kx, Ky, Ksq, bin_masks)
        vqvae_ens += compute_enstrophy_spectrum(vq_f, bin_masks)
        nsp_tke += compute_tke_spectrum(ns_f, Kx, Ky, Ksq, bin_masks)
        nsp_ens += compute_enstrophy_spectrum(ns_f, bin_masks)

        if (i + 1) % log_every_spec == 0 or i == n_pairs - 1:
            print(f"  {i + 1}/{n_pairs} frames")

    gt_tke /= n_pairs
    gt_ens /= n_pairs
    vqvae_tke /= n_pairs
    vqvae_ens /= n_pairs
    nsp_tke /= n_pairs
    nsp_ens /= n_pairs

    # --- Pixel histograms ---
    print("Computing pixel histograms...")
    gt_pixels = raw_gt_pairs[:, 0].ravel()
    vqvae_pixels = gt_decoded[:, 0].ravel()
    nsp_pixels = pred_fields[:, 0].ravel()
    hist_data = compute_histograms(gt_pixels, vqvae_pixels, nsp_pixels)

    # --- Spectral metrics ---
    tke_rse_vqvae = relative_spectral_error(vqvae_tke, gt_tke)
    tke_rse_nsp = relative_spectral_error(nsp_tke, gt_tke)
    enstrophy_rse_vqvae = relative_spectral_error(vqvae_ens, gt_ens)
    enstrophy_rse_nsp = relative_spectral_error(nsp_ens, gt_ens)
    emd_vqvae = pixel_emd(vqvae_pixels, gt_pixels)
    emd_nsp = pixel_emd(nsp_pixels, gt_pixels)

    print(f"\nSpectral metrics:")
    print(f"  TKE RSE:       VQ-VAE={tke_rse_vqvae:.4f}  NSP={tke_rse_nsp:.4f}")
    print(f"  Enstrophy RSE: VQ-VAE={enstrophy_rse_vqvae:.4f}  "
          f"NSP={enstrophy_rse_nsp:.4f}")
    print(f"  Pixel EMD:     VQ-VAE={emd_vqvae:.6f}  NSP={emd_nsp:.6f}")

    # --- Plots ---
    print("Generating plots...")
    tke_fig = plot_spectrum(
        k_centers, gt_tke, vqvae_tke, nsp_tke,
        "TKE", "E(k)", os.path.join(args.output_dir, "tke_spectrum.png"))
    ens_fig = plot_spectrum(
        k_centers, gt_ens, vqvae_ens, nsp_ens,
        "Enstrophy", "Z(k)",
        os.path.join(args.output_dir, "enstrophy_spectrum.png"))
    hist_fig = plot_histogram(
        hist_data, os.path.join(args.output_dir, "pixel_histogram.png"))
    snap_fig = plot_snapshot(
        raw_gt_pairs[0, 0], gt_decoded[0, 0], pred_fields[0, 0], timestep=1)

    # --- Save results ---
    results = {
        "n_pairs": n_pairs,
        "scales": list(scales_int),
        "cross_entropy": mean_ce,
        "pixel_rmse": mean_rmse,
        "vqvae_pixel_rmse": vqvae_rmse,
        "per_scale_ce": {
            f"{config.scales[idx][0]}x{config.scales[idx][1]}": float(per_scale_means[j])
            for j, idx in enumerate(trainable_indices)
        },
        "tke_rse_vqvae": tke_rse_vqvae,
        "tke_rse_nsp": tke_rse_nsp,
        "enstrophy_rse_vqvae": enstrophy_rse_vqvae,
        "enstrophy_rse_nsp": enstrophy_rse_nsp,
        "emd_vqvae": emd_vqvae,
        "emd_nsp": emd_nsp,
    }

    results_path = os.path.join(args.output_dir, "eval_single_step.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_path}")

    # --- Wandb ---
    if WANDB_AVAILABLE:
        if args.wandb_dir is not None:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = args.wandb_dir
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "checkpoint_dir": args.checkpoint_dir,
                "vqvae_dir": args.vqvae_dir,
                "tokens_path": args.tokens_path,
                "n_pairs": n_pairs,
                "scales": list(scales_int),
                "n_layer": arch["n_layer"],
                "n_head": arch["n_head"],
                "n_embd": arch["n_embd"],
            },
        )
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)

        log_dict = {
            "cross_entropy": mean_ce,
            "pixel_rmse": mean_rmse,
            "vqvae_pixel_rmse": vqvae_rmse,
            "tke_rse/vqvae": tke_rse_vqvae,
            "tke_rse/nsp": tke_rse_nsp,
            "enstrophy_rse/vqvae": enstrophy_rse_vqvae,
            "enstrophy_rse/nsp": enstrophy_rse_nsp,
            "emd/vqvae": emd_vqvae,
            "emd/nsp": emd_nsp,
            "tke_spectrum": wandb.Image(tke_fig),
            "enstrophy_spectrum": wandb.Image(ens_fig),
            "pixel_histogram": wandb.Image(hist_fig),
            "snapshot/first_pushforward": wandb.Image(snap_fig),
        }
        for j, idx in enumerate(trainable_indices):
            h, w = config.scales[idx]
            log_dict[f"ce/scale_{h}x{w}"] = float(per_scale_means[j])

        wandb.log(log_dict)
        wandb.finish()
        print("  Logged to wandb")

    plt.close("all")


if __name__ == "__main__":
    main()
