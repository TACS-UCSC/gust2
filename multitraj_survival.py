"""multitraj_survival.py — time-to-explosion via windowed pixel EMD.

For each cfg under <sweep_root>, decodes every trajectory frame and computes
windowed pixel-EMD vs the GT pixel pool at probe times. A trajectory is
"exploded" at the first probe time where windowed EMD exceeds
threshold_factor * VQ-VAE-EMD baseline (matches analyze_rollout.py's
collapse_rate definition; default threshold_factor = 2.0).

We previously tried per-frame pixel_std as a collapse proxy. That misfires
because the diffuse attractor is a smooth low-frequency field — its pixel
std is comparable to GT, only the spatial structure changes. Window EMD on
the marginal pixel distribution captures both "lost variance" and "clipped
to a narrow range around the modes" — which is what actually happens.

Outputs:
  survival.json        — per-cfg explosion times and S(t) at 500/1000/2000
  survival_data.npz    — per-cfg per-trajectory EMD and pixel_std traces
  survival_curves.png  — overlaid survival curves across cfgs
  emd_traces.png       — per-cfg small multiples of windowed EMD vs t
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from analyze_rollout import (
    decode_all_tokens,
    load_raw_gt,
    load_rollout_data,
)
from diagnostics_common import (
    add_wandb_args,
    assign_cfg_styles,
    band_plot,
    describe_decode,
    discover_cfgs,
    init_wandb,
    load_cfg_meta,
    outside_legend,
    set_diag_style,
    wandb_log_figs_and_scalars,
)
from tokenizer import load_checkpoint


def first_above(arr, threshold):
    """First index where arr > threshold; returns len(arr) if never."""
    idxs = np.where(arr > threshold)[0]
    return int(idxs[0]) if idxs.size else len(arr)


def emd_subsampled(a, b, n, rng):
    """Wasserstein-1 distance between two pixel pools, subsampling each
    side to at most n samples. EMD is O(n log n) so this caps cost per
    call; n = 50k -> ~30 ms per call."""
    a = a.ravel()
    b = b.ravel()
    if a.size > n:
        a = rng.choice(a, n, replace=False)
    if b.size > n:
        b = rng.choice(b, n, replace=False)
    return float(wasserstein_distance(a, b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep_root", required=True,
                   help="dir containing <cfg>/rollout/rollout_tokens.npz")
    p.add_argument("--vqvae_dir", required=True)
    p.add_argument("--data_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--threshold_factor", type=float, default=2.0,
                   help="explosion if window EMD > factor * VQ-VAE_EMD")
    p.add_argument("--probe_step", type=int, default=25,
                   help="compute window EMD every N frames")
    p.add_argument("--window", type=int, default=50,
                   help="frames per EMD window centered on probe time")
    p.add_argument("--emd_samples", type=int, default=50000,
                   help="max samples per side for each EMD call")
    p.add_argument("--field", default="omega")
    p.add_argument("--sample_start", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    add_wandb_args(p)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading VQ-VAE...")
    key = jax.random.PRNGKey(args.seed)
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook

    cfgs = discover_cfgs(args.sweep_root)
    cfg_dirs = [os.path.join(args.sweep_root, c, "rollout") for c in cfgs]
    print(f"Found {len(cfgs)} configs: {cfgs}")
    if not cfgs:
        raise SystemExit("No rollout_tokens.npz found.")

    # Shared metadata from first cfg
    first = load_rollout_data(cfg_dirs[0])
    n_frames = first["n_steps"] + 1
    start_frame = int(first["start_frame"])
    scales = first["scales"]
    new_to_old = jnp.array(first["new_to_old"])
    gt_indices_first = np.asarray(first["gt_indices"])
    if gt_indices_first.ndim == 3:
        gt_indices_first = gt_indices_first[0]

    print(f"Loading raw GT (frames {args.sample_start + start_frame}"
          f"+{n_frames})")
    gt = load_raw_gt(args.data_path, args.field, args.sample_start,
                     start_frame, n_frames)
    gt_pixels = gt[:, 0].ravel()
    print(f"GT pool: {gt_pixels.size} pixels")

    print("Decoding VQ-VAE recons of GT (for baseline EMD)...")
    vqvae_fields = decode_all_tokens(gt_indices_first, decoder, vq, codebook,
                                     new_to_old, scales, args.batch_size)
    vqvae_pixels = np.asarray(vqvae_fields)[:, 0].ravel()
    vqvae_emd = emd_subsampled(vqvae_pixels, gt_pixels,
                               args.emd_samples, rng)
    threshold = args.threshold_factor * vqvae_emd
    print(f"VQ-VAE baseline EMD: {vqvae_emd:.4f}")
    print(f"Explosion threshold: window EMD > "
          f"{args.threshold_factor:.1f} * VQ-VAE_EMD = {threshold:.4f}")

    gt_std_per_frame = gt[:, 0].reshape(n_frames, -1).std(axis=1)
    gt_mean_std = float(gt_std_per_frame.mean())

    probe_times = list(range(0, n_frames, args.probe_step))
    if probe_times[-1] != n_frames - 1:
        probe_times.append(n_frames - 1)
    probe_times = np.array(probe_times, dtype=np.int64)
    print(f"Probe times: {len(probe_times)} (every {args.probe_step} frames)")

    results = {
        "vqvae_emd": vqvae_emd,
        "threshold_factor": args.threshold_factor,
        "threshold_emd": threshold,
        "gt_mean_std": gt_mean_std,
        "probe_step": args.probe_step,
        "window": args.window,
        "n_frames": n_frames,
        "probe_times": probe_times.tolist(),
        "configs": {},
    }
    cfg_emd_traces = {}
    cfg_std_traces = {}

    for cfg, cfg_dir in zip(cfgs, cfg_dirs):
        print(f"\n=== {cfg} ===")
        rollout = load_rollout_data(cfg_dir)
        idx = np.asarray(rollout["rollout_indices"])
        if idx.ndim == 2:
            idx = idx[None]
        N, T, _ = idx.shape

        per_frame_std = np.zeros((N, T), dtype=np.float32)
        emd_at_probe = np.zeros((N, len(probe_times)), dtype=np.float32)

        for j in range(N):
            fields = decode_all_tokens(idx[j], decoder, vq, codebook,
                                       new_to_old, scales, args.batch_size)
            flat = np.asarray(fields)[:, 0].reshape(T, -1)
            per_frame_std[j] = flat.std(axis=1)
            for k, t in enumerate(probe_times):
                lo = max(0, int(t) - args.window // 2)
                hi = min(T, int(t) + args.window // 2)
                win = flat[lo:hi].ravel()
                emd_at_probe[j, k] = emd_subsampled(
                    win, gt_pixels, args.emd_samples, rng)
            print(f"  traj {j+1:2d}/{N}: "
                  f"emd[end]={emd_at_probe[j, -1]:.3f}  "
                  f"max={emd_at_probe[j].max():.3f}")

        explosion_idx = np.array([
            first_above(emd_at_probe[j], threshold) for j in range(N)
        ], dtype=np.int64)
        explosion_t = np.array([
            int(probe_times[i]) if i < len(probe_times) else n_frames
            for i in explosion_idx
        ], dtype=np.int64)

        survived = int((explosion_t >= n_frames).sum())
        med_t = int(np.median(explosion_t))
        results["configs"][cfg] = {
            "n_trajectories": int(N),
            "explosion_t": explosion_t.tolist(),
            "survived": survived,
            "median_t": med_t,
            "survival_at_500":  float((explosion_t > 500).sum() / N),
            "survival_at_1000": float((explosion_t > 1000).sum() / N),
            # survival_at_2000 is legacy-named (early sweeps were always
            # 2000 steps); both it and survival_at_end mean survival at the
            # end of the rollout. Prefer survival_at_end downstream.
            "survival_at_2000": float((explosion_t >= n_frames).sum() / N),
            "survival_at_end":  float((explosion_t >= n_frames).sum() / N),
            "max_emd_per_traj": emd_at_probe.max(axis=1).tolist(),
        }
        cfg_emd_traces[cfg] = emd_at_probe
        cfg_std_traces[cfg] = per_frame_std
        info = results["configs"][cfg]
        print(f"  -> survived {survived}/{N}  median t-to-explode={med_t}  "
              f"S(500)={info['survival_at_500']:.2f}  "
              f"S(1000)={info['survival_at_1000']:.2f}  "
              f"S(2000)={info['survival_at_2000']:.2f}")

    out_npz = os.path.join(args.output_dir, "survival_data.npz")
    np.savez_compressed(
        out_npz,
        gt_std_per_frame=gt_std_per_frame,
        probe_times=probe_times,
        **{f"emd_{cfg}": cfg_emd_traces[cfg] for cfg in cfgs},
        **{f"std_{cfg}": cfg_std_traces[cfg] for cfg in cfgs},
    )
    out_json = os.path.join(args.output_dir, "survival.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_json}")
    print(f"Saved {out_npz}")

    # ---------- plots ----------
    set_diag_style()
    metas = {cfg: load_cfg_meta(os.path.join(args.sweep_root, cfg))
             for cfg in cfgs}
    styles = assign_cfg_styles(metas)
    cfg_order = sorted(
        cfgs,
        key=lambda c: -results["configs"][c]["survival_at_end"],
    )

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    ts = np.arange(n_frames + 1)
    survival_curves = {}
    for cfg in cfg_order:
        info = results["configs"][cfg]
        et = np.array(info["explosion_t"])
        survival = np.array([
            float((et > t).sum()) / info["n_trajectories"] for t in ts
        ])
        survival_curves[cfg] = survival
        desc = describe_decode(metas[cfg])
        label = f"{cfg} (S_end={info['survival_at_end']:.0%})"
        if desc:
            label = f"{cfg} [{desc}] S_end={info['survival_at_end']:.0%}"
        ax.step(ts, survival, where="post", lw=1.8, alpha=0.9,
                color=styles[cfg]["color"], ls=styles[cfg]["ls"], label=label)
    ax.set_xlabel("rollout step t")
    ax.set_ylabel("fraction of trajectories surviving")
    ax.set_title(
        f"Survival: collapse = window-EMD > "
        f"{args.threshold_factor:.1f} · VQ-VAE_EMD = {threshold:.3f}"
    )
    ax.set_xlim(0, n_frames)
    ax.set_ylim(-0.02, 1.02)
    outside_legend(ax)
    out_surv = os.path.join(args.output_dir, "survival_curves.png")
    fig.savefig(out_surv)
    print(f"Saved {out_surv}")

    n_cfg = len(cfgs)
    ncols = min(4, n_cfg)
    nrows = (n_cfg + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.5, nrows * 3.2),
                              sharex=True, sharey=True,
                              constrained_layout=True)
    axes = np.atleast_2d(axes)
    for ax, cfg in zip(axes.ravel(), cfg_order):
        emd = cfg_emd_traces[cfg]
        et = np.array(results["configs"][cfg]["explosion_t"])
        coll = et < n_frames
        band_plot(ax, probe_times, emd[coll], color="C3", label="collapsed")
        band_plot(ax, probe_times, emd[~coll], color="C2", label="survived")
        ax.axhline(threshold, lw=0.9, ls="--", color="C0",
                   label=f"thr={threshold:.2f}")
        ax.axhline(vqvae_emd, lw=0.9, ls=":", color="black",
                   label=f"VQ baseline={vqvae_emd:.2f}")
        ax.set_title(
            f"{cfg}  S_end="
            f"{results['configs'][cfg]['survival_at_end']:.0%}",
            fontsize=11,
        )
        ax.set_xlabel("t")
        ax.set_ylabel("window EMD")
    for ax in axes.ravel()[n_cfg:]:
        ax.axis("off")
    axes.ravel()[0].legend(loc="upper left", fontsize=8)
    out_traces = os.path.join(args.output_dir, "emd_traces.png")
    fig2.savefig(out_traces)
    print(f"Saved {out_traces}")

    # ---------- wandb ----------
    run = init_wandb(args, job_type="survival", config={
        "sweep_root": args.sweep_root,
        "vqvae_dir": args.vqvae_dir,
        "threshold_factor": args.threshold_factor,
        "probe_step": args.probe_step,
        "window": args.window,
        "n_frames": n_frames,
        "n_cfgs": len(cfgs),
    })
    scalars = {
        "vqvae_emd": vqvae_emd,
        "threshold_emd": threshold,
    }
    for cfg in cfgs:
        info = results["configs"][cfg]
        scalars[f"survival/{cfg}/survival_at_end"] = info["survival_at_end"]
        scalars[f"survival/{cfg}/median_t"] = info["median_t"]
        scalars[f"survival/{cfg}/survived"] = info["survived"]
    wandb_log_figs_and_scalars(
        run,
        scalars=scalars,
        figs={"survival_curves": fig, "emd_traces": fig2},
        line_series={
            "survival_vs_t": (ts, survival_curves, "rollout step t",
                              "Survival S(t) per cfg"),
        },
    )


if __name__ == "__main__":
    main()
