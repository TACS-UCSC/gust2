"""multitraj_survival.py — time-to-explosion analysis across sampling configs.

Walks <sweep_root>/<cfg>/rollout/rollout_tokens.npz, decodes every frame of
every trajectory, and identifies when each trajectory "explodes" into the
smooth diffuse attractor. Explosion is defined as pixel_std dropping below
alpha * mean(GT_pixel_std) for K consecutive frames.

Outputs:
  survival.json        — per-cfg explosion times and survival summaries
  survival_data.npz    — per-cfg per-trajectory pixel_std traces (for replot)
  survival_curves.png  — overlaid survival curves across cfgs
  pixel_std_traces.png — per-cfg small-multiples of trajectory pixel_std vs t

Designed to consume the directory layout produced by sweep_sampling.sh:
  <sweep_root>/<cfg>/rollout/rollout_tokens.npz
"""
import argparse
import glob
import json
import os

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


def first_sustained_below(series, threshold, K):
    """First index t where series[t:t+K] are all < threshold.
    Returns len(series) if no such window exists."""
    below = series < threshold
    if K <= 1:
        idxs = np.where(below)[0]
        return int(idxs[0]) if idxs.size else len(series)
    if len(series) < K:
        return len(series)
    rolling = np.lib.stride_tricks.sliding_window_view(below, K).all(axis=-1)
    idxs = np.where(rolling)[0]
    return int(idxs[0]) if idxs.size else len(series)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", required=True,
                   help="dir containing <cfg>/rollout/rollout_tokens.npz")
    p.add_argument("--vqvae_dir", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--threshold_alpha", type=float, default=0.5,
                   help="exploded if pixel_std < alpha * mean(GT_pixel_std)")
    p.add_argument("--sustain_frames", type=int, default=10,
                   help="K consecutive frames below threshold to count as exploded")
    p.add_argument("--field", default="omega")
    p.add_argument("--sample_start", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading VQ-VAE...")
    key = jax.random.PRNGKey(args.seed)
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook

    # Discover configs
    pattern = os.path.join(args.sweep_root, "*", "rollout")
    cfg_dirs = sorted([
        d for d in glob.glob(pattern)
        if os.path.isfile(os.path.join(d, "rollout_tokens.npz"))
    ])
    cfgs = [os.path.basename(os.path.dirname(d)) for d in cfg_dirs]
    print(f"Found {len(cfgs)} configs under {args.sweep_root}: {cfgs}")
    if not cfgs:
        raise SystemExit("No rollout_tokens.npz found.")

    # GT reference (assume all configs share the same start_frame / n_steps)
    first = load_rollout_data(cfg_dirs[0])
    n_frames = first["n_steps"] + 1
    start_frame = int(first["start_frame"])
    print(f"Loading GT (frames {args.sample_start + start_frame}+{n_frames})...")
    gt = load_raw_gt(args.data_path, args.field, args.sample_start,
                     start_frame, n_frames)
    gt_std_per_frame = gt[:, 0].reshape(n_frames, -1).std(axis=1)
    gt_mean_std = float(gt_std_per_frame.mean())
    threshold = args.threshold_alpha * gt_mean_std
    print(f"GT pixel_std: mean={gt_mean_std:.4f} "
          f"(min {gt_std_per_frame.min():.4f}, max {gt_std_per_frame.max():.4f})")
    print(f"Explosion threshold: pixel_std < {threshold:.4f} "
          f"({args.threshold_alpha:.2f} * GT mean) "
          f"for {args.sustain_frames} consecutive frames")

    results = {
        "threshold_alpha": args.threshold_alpha,
        "threshold": threshold,
        "gt_mean_std": gt_mean_std,
        "sustain_frames": args.sustain_frames,
        "n_frames": n_frames,
        "configs": {},
    }
    cfg_std_traces = {}

    for cfg, cfg_dir in zip(cfgs, cfg_dirs):
        print(f"\n=== {cfg} ===")
        rollout = load_rollout_data(cfg_dir)
        idx = np.asarray(rollout["rollout_indices"])
        if idx.ndim == 2:
            idx = idx[None]
        N, T, _ = idx.shape
        scales = rollout["scales"]
        new_to_old = jnp.array(rollout["new_to_old"])

        per_frame_std = np.zeros((N, T), dtype=np.float32)
        for j in range(N):
            fields = decode_all_tokens(idx[j], decoder, vq, codebook,
                                       new_to_old, scales, args.batch_size)
            flat = np.asarray(fields)[:, 0].reshape(T, -1)
            per_frame_std[j] = flat.std(axis=1)
            print(f"  traj {j+1:2d}/{N}: "
                  f"std min {per_frame_std[j].min():.3f} "
                  f"end {per_frame_std[j, -1]:.3f}")

        explosion_t = np.array([
            first_sustained_below(per_frame_std[j], threshold,
                                  args.sustain_frames)
            for j in range(N)
        ], dtype=np.int64)
        survived_mask = explosion_t == T
        survived = int(survived_mask.sum())
        med_t = int(np.median(explosion_t))
        results["configs"][cfg] = {
            "n_trajectories": int(N),
            "explosion_t": explosion_t.tolist(),
            "survived": survived,
            "median_t": med_t,
            "survival_at_500":  float((explosion_t > 500).sum() / N),
            "survival_at_1000": float((explosion_t > 1000).sum() / N),
            "survival_at_2000": float((explosion_t >= T).sum() / N),
        }
        cfg_std_traces[cfg] = per_frame_std
        print(f"  -> survived {survived}/{N}  median t-to-explode={med_t}  "
              f"S(500)={results['configs'][cfg]['survival_at_500']:.2f}  "
              f"S(1000)={results['configs'][cfg]['survival_at_1000']:.2f}  "
              f"S(2000)={results['configs'][cfg]['survival_at_2000']:.2f}")

    # ----- Save raw data + JSON -----
    out_npz = os.path.join(args.output_dir, "survival_data.npz")
    np.savez_compressed(
        out_npz,
        gt_std_per_frame=gt_std_per_frame,
        **{f"std_{cfg}": cfg_std_traces[cfg] for cfg in cfgs},
    )
    out_json = os.path.join(args.output_dir, "survival.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")
    print(f"Saved {out_npz}")

    # ----- Survival curves -----
    fig, ax = plt.subplots(figsize=(10, 6))
    ts = np.arange(n_frames + 1)
    cfg_order = sorted(cfgs, key=lambda c: -results["configs"][c]["survival_at_2000"])
    for cfg in cfg_order:
        info = results["configs"][cfg]
        et = np.array(info["explosion_t"])
        survival = np.array([float((et > t).sum()) / info["n_trajectories"]
                             for t in ts])
        ax.step(ts, survival, where="post",
                label=f"{cfg} (S∞={info['survival_at_2000']:.0%})",
                lw=1.6, alpha=0.9)
    ax.set_xlabel("rollout step t")
    ax.set_ylabel("fraction of trajectories surviving")
    ax.set_title(
        f"Survival curves: collapse = pixel_std < "
        f"{args.threshold_alpha:.2f} · GT_std for "
        f"{args.sustain_frames} consec. frames"
    )
    ax.set_xlim(0, n_frames)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=1)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "survival_curves.png")
    fig.savefig(out_png, dpi=130)
    print(f"Saved {out_png}")

    # ----- Per-cfg pixel-std small multiples -----
    n_cfg = len(cfgs)
    ncols = 4
    nrows = (n_cfg + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4, nrows * 3),
                              sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for ax, cfg in zip(axes.ravel(), cfg_order):
        std = cfg_std_traces[cfg]
        et = np.array(results["configs"][cfg]["explosion_t"])
        for j in range(std.shape[0]):
            color = "C3" if et[j] < std.shape[1] else "C2"
            ax.plot(std[j], lw=0.5, alpha=0.55, color=color)
        ax.plot(gt_std_per_frame, lw=1.2, color="black", label="GT")
        ax.axhline(threshold, lw=0.9, ls="--", color="C0",
                   label=f"thr={threshold:.2f}")
        ax.set_title(
            f"{cfg}  S∞={results['configs'][cfg]['survival_at_2000']:.0%}"
        )
        ax.set_xlabel("t")
        ax.set_ylabel("pixel_std")
    for ax in axes.ravel()[n_cfg:]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="lower left", fontsize=8)
    fig2.tight_layout()
    out_traces = os.path.join(args.output_dir, "pixel_std_traces.png")
    fig2.savefig(out_traces, dpi=130)
    print(f"Saved {out_traces}")


if __name__ == "__main__":
    main()
