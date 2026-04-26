"""Build a (1 GT + 1 VQ-VAE + N trajectories) x T-snapshot grid PNG from a
multi-trajectory rollout npz produced by rollout_nsp.py --n_trajectories N.

All trajectories share the same start_frame (only the sampling seed varies),
so GT and VQ-VAE decode are computed once and used as the top two rows.
"""
import argparse
import os

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from analyze_rollout import (
    decode_all_tokens,
    load_raw_gt,
    load_rollout_data,
)
from tokenizer import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True)
    p.add_argument("--vqvae_dir", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_path", required=True)
    p.add_argument("--field", default="omega")
    p.add_argument("--sample_start", type=int, default=20000)
    p.add_argument("--snapshot_times", type=int, nargs="+",
                   default=[1, 2, 5, 10, 50, 100, 250, 500, 1000, 1500, 2000])
    p.add_argument("--cell_size", type=float, default=1.2,
                   help="inches per panel")
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    print(f"Loading rollout from {args.rollout_dir}...")
    rollout = load_rollout_data(args.rollout_dir)
    rollout_indices = np.asarray(rollout["rollout_indices"])
    gt_indices = np.asarray(rollout["gt_indices"])
    if rollout_indices.ndim == 2:
        rollout_indices = rollout_indices[None]
    if gt_indices.ndim == 3:
        gt_indices = gt_indices[0]
    N = rollout_indices.shape[0]
    n_frames = rollout["n_steps"] + 1
    scales = rollout["scales"]
    new_to_old = jnp.array(rollout["new_to_old"])
    start_frame = int(rollout["start_frame"])
    seeds = np.asarray(rollout.get("trajectory_seeds",
                                   np.arange(N, dtype=np.int64)))

    times = [t for t in args.snapshot_times if 0 <= t < n_frames]
    T = len(times)
    print(f"  N={N} trajectories, snapshot timesteps ({T}): {times}")
    print(f"  start_frame={start_frame}, sample_start={args.sample_start}")

    print(f"Loading VQ-VAE from {args.vqvae_dir}...")
    key = jax.random.PRNGKey(args.seed)
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook

    print(f"Loading GT from {args.data_path} "
          f"(frames {args.sample_start + start_frame}+{max(times)})...")
    gt = load_raw_gt(args.data_path, args.field,
                     args.sample_start, start_frame, n_frames)
    gt_at_times = gt[times, 0]                                # (T, 256, 256)

    print("Decoding VQ-VAE recons at snapshot times...")
    vq_idx = np.stack([gt_indices[t] for t in times])          # (T, tokens)
    vq_fields = decode_all_tokens(vq_idx, decoder, vq, codebook,
                                  new_to_old, scales, args.batch_size)
    vq_at_times = np.asarray(vq_fields)[:, 0]                  # (T, 256, 256)

    print(f"Decoding NSP for all {N} trajectories at snapshot times...")
    nsp_idx = rollout_indices[:, times, :]                     # (N, T, tok)
    nsp_idx_flat = nsp_idx.reshape(N * T, -1)
    nsp_fields = decode_all_tokens(nsp_idx_flat, decoder, vq, codebook,
                                   new_to_old, scales, args.batch_size)
    nsp_at_times = np.asarray(nsp_fields).reshape(N, T, 256, 256)

    # ----- Build grid -----
    rows = N + 2
    cols = T
    fig_w = args.cell_size * cols + 1.0    # leave room for left labels
    fig_h = args.cell_size * rows + 0.3
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                             constrained_layout=True)
    if rows == 1:
        axes = axes[None, :]
    if cols == 1:
        axes = axes[:, None]

    vmin = float(gt_at_times.min())
    vmax = float(gt_at_times.max())

    for j, t in enumerate(times):
        axes[0, j].set_title(f"t={t}", fontsize=9)

    def _show(ax, field, label=None):
        ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if label is not None:
            ax.set_ylabel(label, fontsize=8, rotation=0,
                          ha="right", va="center", labelpad=4)

    for j in range(cols):
        _show(axes[0, j], gt_at_times[j], "GT" if j == 0 else None)
        _show(axes[1, j], vq_at_times[j], "VQ-VAE" if j == 0 else None)
        for i in range(N):
            label = f"s={int(seeds[i])}" if j == 0 else None
            _show(axes[i + 2, j], nsp_at_times[i, j], label)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)),
                exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi)
    print(f"Saved: {args.output_path} ({fig_w:.1f} x {fig_h:.1f} in @ {args.dpi} dpi)")


if __name__ == "__main__":
    main()
