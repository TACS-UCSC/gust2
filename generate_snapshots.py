"""Generate GT / VQ-VAE / NSP snapshot comparisons from rollout tokens.

Decodes only the requested timesteps (not all frames), so this is much
faster than re-running the full analysis.  Saves PNGs to the output dir
and optionally logs to an existing wandb run.

Usage:
    python generate_snapshots.py \
        --rollout_dir experiments/rollouts/small-sc341-nsp-medium \
        --vqvae_dir experiments/vqvae/small-sc341 \
        --data_path data_lowres/output.h5 \
        --output_dir experiments/analysis/small-sc341-nsp-medium \
        --timesteps 100 500 1000
"""

import argparse
import os

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

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
        description="Generate GT/VQ-VAE/NSP snapshot images from rollout tokens")
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
    parser.add_argument("--timesteps", type=int, nargs="+",
                        default=[100, 500, 1000],
                        help="Timesteps to snapshot (default: 100 500 1000)")
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


def load_raw_gt_frames(data_path, field, sample_start, start_frame, timesteps):
    """Load specific raw vorticity frames from HDF5.

    Returns: dict mapping timestep -> (1, 256, 256) float32 array.
    """
    frames = {}
    with h5py.File(data_path, "r") as f:
        ds = f[f"fields/{field}"]
        for t in timesteps:
            idx = sample_start + start_frame + t
            frames[t] = ds[idx].astype(np.float32)[None, :, :]
    return frames


# =============================================================================
# VQ-VAE decoding (single-frame)
# =============================================================================


@eqx.filter_jit
def _decode_single(decoder, vq, codebook, new_to_old, flat_indices, scales):
    """Decode one frame: compact flat indices -> (1, 256, 256)."""
    decoder = eqx.nn.inference_mode(decoder)
    vq = eqx.nn.inference_mode(vq)
    indices_list = unflatten_to_scales(flat_indices, scales)
    original_indices = [new_to_old[idx] for idx in indices_list]
    z_q = reconstruct_from_indices(original_indices, codebook, vq)
    return decoder(z_q)


# =============================================================================
# Plotting
# =============================================================================


def plot_snapshot(gt_field, vqvae_field, nsp_field, timestep):
    """Plot GT / VQ-VAE / NSP side-by-side for a single timestep."""
    vmin = gt_field.min()
    vmax = gt_field.max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, field, label in [
        (axes[0], gt_field, "Ground Truth"),
        (axes[1], vqvae_field, "VQ-VAE"),
        (axes[2], nsp_field, "NSP"),
    ]:
        im = ax.imshow(field, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                        origin="lower")
        ax.set_title(f"{label} (t={timestep})", fontsize=12)
        ax.axis("off")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Vorticity")
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

    # Always include the first NSP output (t=1), plus user-requested timesteps
    timesteps = sorted(set([1] + args.timesteps))
    timesteps = [t for t in timesteps if t < n_frames]
    skipped = [t for t in args.timesteps if t >= n_frames]
    if skipped:
        print(f"  Skipping timesteps {skipped} (only {n_frames} frames)")
    if not timesteps:
        print("No valid timesteps — nothing to do.")
        return
    print(f"  Generating snapshots for t={timesteps}")

    # --- Load VQ-VAE ---
    print(f"Loading VQ-VAE from {args.vqvae_dir}...")
    key = jax.random.PRNGKey(args.seed)
    _, decoder, vq, ema_state, _ = load_checkpoint(args.vqvae_dir, key)
    codebook = ema_state.codebook
    print(f"  Codebook: {codebook.shape}")

    # --- Load raw GT for requested timesteps ---
    print(f"Loading raw GT frames from {args.data_path}...")
    gt_frames = load_raw_gt_frames(
        args.data_path, args.field, args.sample_start,
        rollout["start_frame"], timesteps)

    # --- Decode & plot each timestep ---
    wandb_images = {}
    for t in timesteps:
        print(f"  t={t}: decoding...")
        gt_field = gt_frames[t][0]

        gt_indices = jnp.array(rollout["gt_indices"][t])
        vqvae_field = np.array(
            _decode_single(decoder, vq, codebook, new_to_old,
                           gt_indices, scales))[0]

        nsp_indices = jnp.array(rollout["rollout_indices"][t])
        nsp_field = np.array(
            _decode_single(decoder, vq, codebook, new_to_old,
                           nsp_indices, scales))[0]

        fig = plot_snapshot(gt_field, vqvae_field, nsp_field, t)
        out_path = os.path.join(args.output_dir, f"snapshot_t{t}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"    Saved {out_path}")
        wandb_images[f"snapshot/t{t}"] = fig
        plt.close(fig)

    # --- Wandb ---
    if WANDB_AVAILABLE and args.wandb_name:
        if args.wandb_dir is not None:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = args.wandb_dir
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "rollout_dir": args.rollout_dir,
                "vqvae_dir": args.vqvae_dir,
                "timesteps": timesteps,
            },
        )
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)
        log_dict = {k: wandb.Image(v) for k, v in wandb_images.items()}
        wandb.log(log_dict)
        wandb.finish()
        print("  Logged to wandb")

    print("Done.")


if __name__ == "__main__":
    main()
