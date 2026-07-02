"""Analyze continuous-baseline rollouts (pixel-field input, no token decode).

The pixel-input sibling of analyze_rollout.py for B1/B2 rollouts produced by
rollout_continuous.py. Reuses the exact spectral/metric machinery (imported
from analyze_rollout) so numbers are comparable across the discrete and
continuous pipelines. Three-way comparison:

  GT        time-averaged stats over the full val window (climate reference)
  one-step  model applied to each GT frame once (teacher-forced) — the
            per-step operator's spectral signature BEFORE compounding
  rollout   closed-loop trajectories — the same operator AFTER compounding

The one-step vs rollout gap is the pillar-2 exhibit (F7.1): a continuous
regressor re-pays its spectral bias every step; the discrete pipeline pays
the tokenizer floor once at readout.

Also computes per-step spectral band-energy traces (low/mid/high k thirds)
— the raw material for the high-k-energy-vs-step figure.

Outputs (mirroring analyze_rollout conventions):
  analysis_data.npz   spectra + histograms + traces for replotting
  metrics.json        scalar metrics
  tke_spectrum.png, enstrophy_spectrum.png, band_traces.png,
  pixel_histogram.png, snapshots/t*.png
"""

import jax
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

from analyze_rollout import (
    setup_spectral_analysis,
    setup_radial_bincount,
    _spectra_from_fields_batched,
    relative_spectral_error,
    pixel_emd,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectral analysis of continuous-baseline pixel rollouts")
    parser.add_argument("--rollout_dir", type=str, required=True,
                        help="Directory with rollout_fields.npz")
    parser.add_argument("--checkpoint_dir", type=str, default=None,
                        help="Model checkpoint for the one-step column "
                             "(omit to skip)")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--field", type=str, default="omega")
    parser.add_argument("--sample_start", type=int, default=20000,
                        help="GT climate window start (must match rollout)")
    parser.add_argument("--sample_stop", type=int, default=22000)
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for one-step predictions")
    parser.add_argument("--spectra_stride", type=int, default=10,
                        help="Save full per-step spectra every this many steps")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, default="gust2-analysis")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    return parser.parse_args()


def compute_onestep_predictions(checkpoint_dir, gt, batch_size):
    """Apply the model once to every GT frame: (M, H, W) -> (M-1, H, W)."""
    import equinox as eqx
    from train_next_vit import build_model

    with open(os.path.join(checkpoint_dir, "training_state.json")) as f:
        arch_config = json.load(f)["arch_config"]
    if arch_config.get("model_type", "next_vit") != "next_vit":
        raise SystemExit(f"one-step column: unsupported model_type "
                         f"{arch_config.get('model_type')!r}")

    encoder, decoder = build_model(arch_config, jax.random.PRNGKey(0))
    encoder = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "encoder.eqx"), encoder)
    decoder = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "decoder.eqx"), decoder)
    encoder = eqx.nn.inference_mode(encoder)
    decoder = eqx.nn.inference_mode(decoder)

    @eqx.filter_jit
    def step(xb):
        return jax.vmap(lambda s: decoder(encoder(s)))(xb)

    inputs = gt[:-1]                                     # predict frames 1..M-1
    preds = np.zeros_like(inputs)
    import jax.numpy as jnp
    for i in range(0, len(inputs), batch_size):
        batch = jnp.asarray(inputs[i:i + batch_size][:, None], dtype=jnp.float32)
        preds[i:i + batch_size] = np.asarray(step(batch))[:, 0]
        if (i // batch_size + 1) % 10 == 0:
            print(f"    {min(i + batch_size, len(inputs))}/{len(inputs)} frames")
    return preds


def band_energies(spectra, n_bins):
    """Split k bins into thirds by index; sum E(k) per band.

    spectra: (..., n_bins) -> (..., 3) [low, mid, high].
    """
    e1, e2 = n_bins // 3, 2 * n_bins // 3
    return np.stack([spectra[..., :e1].sum(axis=-1),
                     spectra[..., e1:e2].sum(axis=-1),
                     spectra[..., e2:].sum(axis=-1)], axis=-1)


def plot_spectrum_3way(k_centers, gt, onestep, rollout, rollout_per_traj,
                       spectrum_type, ylabel, output_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    valid_k = k_centers > 0

    for j in range(rollout_per_traj.shape[0]):
        spec = rollout_per_traj[j]
        mask = valid_k & (spec > 0)
        if np.any(mask):
            ax.loglog(k_centers[mask], spec[mask], color="red",
                      alpha=0.15, linewidth=0.8)

    curves = [(gt, "Ground Truth", "blue", "-")]
    if onestep is not None:
        curves.append((onestep, "One-step", "green", "--"))
    curves.append((rollout, "Rollout (ensemble)", "red", ":"))
    for spec, label, color, ls in curves:
        mask = valid_k & (spec > 0)
        if np.any(mask):
            ax.loglog(k_centers[mask], spec[mask], label=label,
                      color=color, linestyle=ls, alpha=0.8, linewidth=2)

    ax.set_title(f"Time-Averaged {spectrum_type} Spectrum", fontsize=14)
    ax.set_xlabel("Wavenumber |k|", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved {spectrum_type} spectrum to {output_path}")
    return fig


def plot_band_traces(band_traces, gt_bands, output_path):
    """Per-trajectory band-energy ratio vs rollout step (F7.1 raw material)."""
    labels = ["low k", "mid k", "high k"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    N, T1, _ = band_traces.shape
    steps = np.arange(T1)
    for b, ax in enumerate(axes):
        for j in range(N):
            ratio = band_traces[j, :, b] / max(gt_bands[b], 1e-30)
            ax.plot(steps, ratio, alpha=0.6, linewidth=1.0)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="GT level")
        ax.set_yscale("log")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Band energy / GT band energy")
        ax.set_title(labels[b])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved band traces to {output_path}")
    return fig


def plot_histogram_3way(bin_centers, hists, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {"gt": ("Ground Truth", "blue", "-"),
              "onestep": ("One-step", "green", "--"),
              "rollout": ("Rollout", "red", ":")}
    for key, hist in hists.items():
        label, color, ls = styles[key]
        ax.step(bin_centers, hist, where="mid", label=label, color=color,
                linestyle=ls, linewidth=2, alpha=0.8)
    ax.set_xlabel("Pixel Value (Vorticity)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pixel Value Distribution", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"  Saved pixel histogram to {output_path}")
    return fig


def plot_snapshot_pair(gt_field, rollout_field, timestep):
    """GT / rollout side-by-side (constrained_layout; shared colorbar —
    do NOT pass bbox_inches='tight' when saving)."""
    vmin, vmax = gt_field.min(), gt_field.max()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for ax, field, label in [(axes[0], gt_field, "Ground Truth"),
                             (axes[1], rollout_field, "Rollout")]:
        im = ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                       origin="lower")
        ax.set_title(f"{label} (t={timestep})", fontsize=12)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Vorticity")
    return fig


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load rollout fields ---
    rollout_path = os.path.join(args.rollout_dir, "rollout_fields.npz")
    print(f"Loading rollout from {rollout_path}...")
    data = np.load(rollout_path)
    fields = np.asarray(data["fields"], dtype=np.float32)   # (N, T+1, H, W)
    start_frames = np.asarray(data["start_frames"])
    N, T1, H, W = fields.shape
    print(f"  {N} trajectories x {T1} frames ({H}x{W}), "
          f"start_frames={start_frames.tolist()}")

    # --- GT climate window ---
    print(f"Loading GT frames {args.sample_start}-{args.sample_stop}...")
    with h5py.File(args.data_path, "r") as f:
        gt = np.asarray(f[f"fields/{args.field}"]
                        [args.sample_start:args.sample_stop], dtype=np.float32)
    M = gt.shape[0]

    # --- Spectral setup ---
    Kx, Ky, Ksq, k_centers, bin_masks = setup_spectral_analysis(H, W)
    bin_index, bin_counts, n_bins = setup_radial_bincount(bin_masks)

    print("Computing GT spectra...")
    gt_tke_frames, gt_ens_frames = _spectra_from_fields_batched(
        gt, Kx, Ky, Ksq, bin_index, bin_counts, n_bins)
    gt_tke = gt_tke_frames.mean(axis=0)
    gt_ens = gt_ens_frames.mean(axis=0)
    gt_bands = band_energies(gt_tke, n_bins)

    # --- One-step column (optional) ---
    onestep_tke = onestep_ens = None
    onestep_metrics = {}
    onestep_pixels = None
    if args.checkpoint_dir is not None:
        print("Computing one-step predictions...")
        preds = compute_onestep_predictions(args.checkpoint_dir, gt,
                                            args.batch_size)
        onestep_mse = float(np.mean((preds - gt[1:]) ** 2))
        os_tke_frames, os_ens_frames = _spectra_from_fields_batched(
            preds, Kx, Ky, Ksq, bin_index, bin_counts, n_bins)
        onestep_tke = os_tke_frames.mean(axis=0)
        onestep_ens = os_ens_frames.mean(axis=0)
        onestep_pixels = preds.ravel()
        onestep_metrics = {
            "onestep_mse": onestep_mse,
            "tke_rse_onestep": relative_spectral_error(onestep_tke, gt_tke),
            "enstrophy_rse_onestep": relative_spectral_error(onestep_ens, gt_ens),
            "emd_onestep": pixel_emd(onestep_pixels, gt.ravel()),
        }
        print(f"  one-step MSE={onestep_mse:.5f}")

    # --- Rollout spectra: per-frame per trajectory ---
    print("Computing rollout spectra...")
    tke_per_traj = np.zeros((N, n_bins))
    ens_per_traj = np.zeros((N, n_bins))
    band_traces = np.zeros((N, T1, 3))
    strided_idx = np.arange(0, T1, args.spectra_stride)
    tke_traces_strided = np.zeros((N, len(strided_idx), n_bins))
    for j in range(N):
        tke_frames, ens_frames = _spectra_from_fields_batched(
            fields[j], Kx, Ky, Ksq, bin_index, bin_counts, n_bins)
        tke_per_traj[j] = tke_frames.mean(axis=0)
        ens_per_traj[j] = ens_frames.mean(axis=0)
        band_traces[j] = band_energies(tke_frames, n_bins)
        tke_traces_strided[j] = tke_frames[strided_idx]
        print(f"  traj {j + 1}/{N} done")
    rollout_tke = tke_per_traj.mean(axis=0)
    rollout_ens = ens_per_traj.mean(axis=0)

    # --- Pixel distributions ---
    print("Computing pixel distributions...")
    gt_pixels = gt.ravel()
    rollout_pixels_per_traj = [fields[j].ravel() for j in range(N)]

    bin_min, bin_max = gt_pixels.min(), gt_pixels.max()
    margin = (bin_max - bin_min) * 0.01
    hbins = np.linspace(bin_min - margin, bin_max + margin, 101)
    hist_bin_centers = 0.5 * (hbins[:-1] + hbins[1:])
    hists = {"gt": np.histogram(gt_pixels, bins=hbins, density=True)[0]}
    if onestep_pixels is not None:
        hists["onestep"] = np.histogram(onestep_pixels, bins=hbins,
                                        density=True)[0]
    hists["rollout"] = np.histogram(np.concatenate(rollout_pixels_per_traj),
                                    bins=hbins, density=True)[0]

    # --- Per-trajectory metrics ---
    print("Computing metrics...")
    tke_rse_per_traj = np.array([
        relative_spectral_error(tke_per_traj[j], gt_tke) for j in range(N)])
    ens_rse_per_traj = np.array([
        relative_spectral_error(ens_per_traj[j], gt_ens) for j in range(N)])
    emd_per_traj = np.array([
        pixel_emd(rollout_pixels_per_traj[j], gt_pixels) for j in range(N)])

    # High-k retention: tail-mean high-band energy / GT high-band energy.
    # The scalar version of F7.1 — 1.0 = holds the GT high-k level, ->0 =
    # compounding attenuation, >>1 = blow-up.
    tail = max(1, T1 // 10)
    highk_retention_per_traj = (band_traces[:, -tail:, 2].mean(axis=1)
                                / max(gt_bands[2], 1e-30))

    def agg(a):
        return float(np.mean(a)), float(np.std(a)), float(np.max(a))

    tke_rse_mean, tke_rse_std, tke_rse_max = agg(tke_rse_per_traj)
    ens_rse_mean, ens_rse_std, ens_rse_max = agg(ens_rse_per_traj)
    emd_mean, emd_std, emd_max = agg(emd_per_traj)

    metrics = {
        "n_trajectories": N,
        "n_steps": T1 - 1,
        "start_frames": start_frames.tolist(),
        **onestep_metrics,
        "tke_rse_rollout_mean": tke_rse_mean,
        "tke_rse_rollout_std": tke_rse_std,
        "tke_rse_rollout_max": tke_rse_max,
        "enstrophy_rse_rollout_mean": ens_rse_mean,
        "enstrophy_rse_rollout_std": ens_rse_std,
        "enstrophy_rse_rollout_max": ens_rse_max,
        "emd_rollout_mean": emd_mean,
        "emd_rollout_std": emd_std,
        "emd_rollout_max": emd_max,
        "tke_rse_rollout_per_traj": tke_rse_per_traj.tolist(),
        "enstrophy_rse_rollout_per_traj": ens_rse_per_traj.tolist(),
        "emd_rollout_per_traj": emd_per_traj.tolist(),
        "highk_retention_mean": float(np.mean(highk_retention_per_traj)),
        "highk_retention_std": float(np.std(highk_retention_per_traj)),
        "highk_retention_per_traj": highk_retention_per_traj.tolist(),
    }

    # --- Save plot data ---
    data_path = os.path.join(args.output_dir, "analysis_data.npz")
    save_dict = dict(
        k_centers=k_centers,
        tke_gt=gt_tke, enstrophy_gt=gt_ens,
        tke_rollout=rollout_tke, enstrophy_rollout=rollout_ens,
        tke_rollout_per_traj=tke_per_traj,
        enstrophy_rollout_per_traj=ens_per_traj,
        band_traces_tke=band_traces,
        gt_bands_tke=gt_bands,
        tke_traces_strided=tke_traces_strided,
        strided_steps=strided_idx,
        hist_bin_centers=hist_bin_centers,
        hist_gt=hists["gt"], hist_rollout=hists["rollout"],
        emd_rollout_per_traj=emd_per_traj,
        tke_rse_rollout_per_traj=tke_rse_per_traj,
        enstrophy_rse_rollout_per_traj=ens_rse_per_traj,
        highk_retention_per_traj=highk_retention_per_traj,
        start_frames=start_frames,
    )
    if onestep_tke is not None:
        save_dict.update(tke_onestep=onestep_tke, enstrophy_onestep=onestep_ens,
                         hist_onestep=hists["onestep"])
    np.savez_compressed(data_path, **save_dict)
    print(f"  Saved plot data to {data_path}")

    # --- Plots ---
    print("Saving plots...")
    tke_fig = plot_spectrum_3way(
        k_centers, gt_tke, onestep_tke, rollout_tke, tke_per_traj,
        "TKE", "E(k)", os.path.join(args.output_dir, "tke_spectrum.png"))
    ens_fig = plot_spectrum_3way(
        k_centers, gt_ens, onestep_ens, rollout_ens, ens_per_traj,
        "Enstrophy", "Z(k)",
        os.path.join(args.output_dir, "enstrophy_spectrum.png"))
    band_fig = plot_band_traces(
        band_traces, gt_bands, os.path.join(args.output_dir, "band_traces.png"))
    hist_fig = plot_histogram_3way(
        hist_bin_centers, hists,
        os.path.join(args.output_dir, "pixel_histogram.png"))

    # Snapshots: trajectory 0 vs GT at the same absolute time where the val
    # window still covers it; beyond the window only the rollout is shown.
    snapshot_dir = os.path.join(args.output_dir, "snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_figs = {}
    for t in [1, 10, 100, 500, 1000, 2000]:
        if t >= T1:
            continue
        gt_t = int(start_frames[0]) + t
        if gt_t < M:
            fig = plot_snapshot_pair(gt[gt_t], fields[0, t], t)
        else:
            fig, ax = plt.subplots(figsize=(5.5, 5), constrained_layout=True)
            im = ax.imshow(fields[0, t], cmap="RdBu_r", origin="lower")
            ax.set_title(f"Rollout (t={t}, beyond GT window)", fontsize=12)
            ax.axis("off")
            fig.colorbar(im, ax=ax, shrink=0.8, label="Vorticity")
        out = os.path.join(snapshot_dir, f"t{t:04d}.png")
        fig.savefig(out, dpi=150)
        snapshot_figs[t] = fig
        print(f"  Saved snapshot t={t}")

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
            project=args.wandb_project, name=args.wandb_name,
            config={"rollout_dir": args.rollout_dir,
                    "checkpoint_dir": args.checkpoint_dir,
                    "n_trajectories": N, "n_steps": T1 - 1})
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)
        log_dict = {
            **{k: v for k, v in metrics.items()
               if isinstance(v, (int, float))},
            "tke_spectrum": wandb.Image(tke_fig),
            "enstrophy_spectrum": wandb.Image(ens_fig),
            "band_traces": wandb.Image(band_fig),
            "pixel_histogram": wandb.Image(hist_fig),
        }
        for t, fig in snapshot_figs.items():
            log_dict[f"snapshot/t{t}"] = wandb.Image(fig)
        wandb.log(log_dict)
        wandb.finish()
        print("  Logged to wandb")

    plt.close("all")

    # --- Summary ---
    print(f"\nResults ({N} trajectories, {T1 - 1} steps):")
    if onestep_metrics:
        print(f"  One-step:  MSE={onestep_metrics['onestep_mse']:.5f}  "
              f"TKE RSE={onestep_metrics['tke_rse_onestep']:.4f}  "
              f"EMD={onestep_metrics['emd_onestep']:.6f}")
    print(f"  Rollout:   TKE RSE={tke_rse_mean:.4f} +/- {tke_rse_std:.4f}  "
          f"EMD={emd_mean:.6f} +/- {emd_std:.6f}")
    print(f"  High-k retention (tail mean / GT): "
          f"{metrics['highk_retention_mean']:.4f} "
          f"+/- {metrics['highk_retention_std']:.4f}")


if __name__ == "__main__":
    main()
