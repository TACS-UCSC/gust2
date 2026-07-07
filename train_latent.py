"""B1 baseline — skip-quantization continuous-latent dynamics (BASELINES_SPEC.md §B1).

The cleanest experiment in the suite: identical frozen enc/dec, identical
data, the only bit flipped is quantize-or-don't. A transformer with the NSP
block recipe (RMSNorm, QK-norm, bias-free, SwiGLU, 2D axial RoPE) regresses
z_{t+1} from z_t in one shot on the full continuous 32x32 latent — no
multi-scale machinery, no quantizer. MSE loss. Parameter count matched to
the flagship NSP (s18-class) via --n_layer/--n_embd/--n_head.

Anchor (locked decision): frozen medium-sc341 VQ-VAE encoder/decoder,
quantizer bypassed. The enc/dec are loaded read-only from the VQ-VAE
checkpoint and are NOT copied into this run's checkpoint — the path is
recorded in arch_config and re-resolved by rollout_continuous.py /
analyze_continuous.py.

Stabilizer variant (each an arm in the baseline grid):
  --noise_sigma S   Gaussian input noise on z_t, per-channel std = S x the
                    train-set per-channel latent std (Stachenfeld-style,
                    latent-space analogue of train_next_vit's flag). The
                    per-channel std is measured by a short encode probe at
                    startup (--probe_frames) and logged to config.
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import optax
import equinox as eqx
import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with 'pip install wandb' for logging.")

from nsp_model import NSPConfig, NSPBlock, build_rope_coords
from tokenizer import load_checkpoint as load_vqvae_checkpoint
from dataloader import FramePairDataset
from train_next_vit import plot_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train continuous-latent next-frame regressor "
                    "(B1 skip-quantization baseline)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .h5 file")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name under /fields/")
    parser.add_argument("--vqvae_checkpoint", type=str, required=True,
                        help="Frozen VQ-VAE checkpoint dir (anchor: "
                             "medium-sc341); quantizer bypassed")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Dynamics architecture (defaults ~18M: the s18-class param match)
    parser.add_argument("--n_layer", type=int, default=10)
    parser.add_argument("--n_embd", type=int, default=384)
    parser.add_argument("--n_head", type=int, default=6)
    parser.add_argument("--rope_theta", type=float, default=32.0)
    # Baseline variant
    parser.add_argument("--noise_sigma", type=float, default=0.0,
                        help="Training input-noise std as a FRACTION of the "
                             "per-channel train-set latent std (0 = plain MSE)")
    parser.add_argument("--probe_frames", type=int, default=2048,
                        help="Frames encoded at startup to measure the "
                             "per-channel latent std for --noise_sigma")
    parser.add_argument("--seed", type=int, default=42)
    # Data slicing (train = [0, 20000), val = [20000, 22000) by project convention)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_stop", type=int, default=20000)
    parser.add_argument("--val_start", type=int, default=20000)
    parser.add_argument("--val_stop", type=int, default=22000,
                        help="Set val_stop <= val_start to disable the val pass")
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true")
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="gust2-baselines")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    return parser.parse_args()


# ---------- Model ----------


class LatentDynamics(eqx.Module):
    """One-shot z_t -> z_{t+1} regressor on the (latent_dim, hw, hw) grid.

    NSP block recipe over the hw*hw token sequence with full (bias-free)
    self-attention and cell-center 2D RoPE; linear in/out projections.
    """
    in_proj: eqx.nn.Linear
    blocks: list
    final_norm: eqx.nn.RMSNorm
    out_proj: eqx.nn.Linear
    latent_hw: int = eqx.field(static=True)

    def __init__(self, config: NSPConfig, latent_dim: int, latent_hw: int,
                 *, key: jax.Array):
        k_in, k_out, k_blocks = jax.random.split(key, 3)
        self.latent_hw = latent_hw
        self.in_proj = eqx.nn.Linear(
            latent_dim, config.n_embd, use_bias=False, key=k_in)
        self.blocks = [NSPBlock(config, key=k)
                       for k in jax.random.split(k_blocks, config.n_layer)]
        self.final_norm = eqx.nn.RMSNorm(config.n_embd)
        self.out_proj = eqx.nn.Linear(
            config.n_embd, latent_dim, use_bias=False, key=k_out)

    def __call__(self, z: jax.Array) -> jax.Array:
        """(latent_dim, hw, hw) -> (latent_dim, hw, hw), single latent step."""
        C = z.shape[0]
        hw = self.latent_hw
        L = hw * hw
        dtype = z.dtype

        # Trace-time constants: full attention, cell-center coords
        rope_coords = build_rope_coords(((hw, hw),), L)
        attn_bias = jnp.zeros((L, L), dtype=dtype)

        x = z.reshape(C, L).T                              # (L, C)
        x = jax.vmap(self.in_proj)(x)
        for block in self.blocks:
            x = block(x, attn_bias, rope_coords)
        x = jax.vmap(self.final_norm)(x.astype(jnp.float32)).astype(dtype)
        x = jax.vmap(self.out_proj)(x)
        return x.T.reshape(C, hw, hw)


def build_model(arch_config, key):
    """Build a LatentDynamics skeleton from an arch_config dict.

    Shared with rollout_continuous.py / analyze_continuous.py so checkpoint
    loading can't drift.
    """
    config = NSPConfig(
        n_layer=arch_config["n_layer"],
        n_head=arch_config["n_head"],
        n_embd=arch_config["n_embd"],
        rope_theta=arch_config["rope_theta"])
    return LatentDynamics(config, arch_config["latent_dim"],
                          arch_config["latent_hw"], key=key)


def load_latent_model(checkpoint_dir, arch_config=None):
    """Load (dynamics, frozen encoder, frozen decoder, arch_config).

    The frozen enc/dec are re-resolved from arch_config["vqvae_checkpoint"];
    only the dynamics weights live in this checkpoint.
    """
    if arch_config is None:
        with open(os.path.join(checkpoint_dir, "training_state.json")) as f:
            arch_config = json.load(f)["arch_config"]
    dynamics = build_model(arch_config, jax.random.PRNGKey(0))
    dynamics = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "dynamics.eqx"), dynamics)
    dynamics = eqx.nn.inference_mode(dynamics)
    encoder, decoder, _, _, _ = load_vqvae_checkpoint(
        arch_config["vqvae_checkpoint"], jax.random.PRNGKey(0))
    encoder = eqx.nn.inference_mode(encoder)
    decoder = eqx.nn.inference_mode(decoder)
    return dynamics, encoder, decoder, arch_config


# ---------- Mixed precision ----------


def _cast_to_half(model):
    """Cast model float arrays to bfloat16. Master weights stay f32 outside."""
    def cast(x):
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.bfloat16)
        return x
    return jax.tree.map(cast, model)


def _mse(pred, target):
    return jnp.mean((pred.astype(jnp.float32) - target.astype(jnp.float32)) ** 2)


# ---------- Frozen-encoder helpers ----------


@eqx.filter_jit
def encode_batch(encoder, xb):
    """(B, 1, H, W) f32 -> (B, latent_dim, hw, hw) f32, frozen encoder."""
    return jax.vmap(encoder)(xb)


@eqx.filter_jit
def decode_batch(decoder, zb):
    """(B, latent_dim, hw, hw) f32 -> (B, 1, H, W) f32, frozen decoder."""
    return jax.vmap(decoder)(zb)


def probe_latent_std(encoder, dataset, n_frames, batch_size):
    """Per-channel latent std over the first n_frames train frames."""
    n_frames = min(n_frames, dataset.data.shape[0])
    sq_sum = None
    mean_sum = None
    count = 0
    for i in range(0, n_frames, batch_size):
        xb = jnp.asarray(dataset.data[i:i + batch_size], dtype=jnp.float32)
        zb = np.asarray(encode_batch(encoder, xb), dtype=np.float64)
        # accumulate over (B, hw, hw) per channel
        b_sum = zb.sum(axis=(0, 2, 3))
        b_sq = (zb ** 2).sum(axis=(0, 2, 3))
        mean_sum = b_sum if mean_sum is None else mean_sum + b_sum
        sq_sum = b_sq if sq_sum is None else sq_sum + b_sq
        count += zb.shape[0] * zb.shape[2] * zb.shape[3]
    mean = mean_sum / count
    var = sq_sum / count - mean ** 2
    return np.sqrt(np.maximum(var, 0.0)).astype(np.float32)   # (latent_dim,)


# ---------- Checkpointing ----------


def save_checkpoint(dynamics, opt_state, epoch, global_step,
                    checkpoint_dir, wandb_dir=None, arch_config=None):
    os.makedirs(checkpoint_dir, exist_ok=True)

    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "dynamics.eqx"), dynamics)
    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "opt_state.eqx"), opt_state)

    state = {"epoch": epoch, "global_step": global_step}
    if arch_config is not None:
        state["arch_config"] = arch_config
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(state, f)

    print(f"Saved checkpoint (epoch {epoch}) to {checkpoint_dir}")

    if wandb_dir is not None:
        os.makedirs(wandb_dir, exist_ok=True)
        eqx.tree_serialise_leaves(os.path.join(wandb_dir, "dynamics.eqx"), dynamics)


# ---------- Training step ----------


def make_train_step(optimizer, noise_std_ch):
    """Build the jitted train step; noise resolves at trace time.

    noise_std_ch: (latent_dim,) absolute per-channel noise std, or None.
    """

    @eqx.filter_value_and_grad
    def loss_and_grad(dynamics, z0, z1, noise_key):
        dyn_half = _cast_to_half(dynamics)
        if noise_std_ch is not None:
            z0 = z0 + noise_std_ch[None, :, None, None] * jax.random.normal(
                noise_key, z0.shape, dtype=z0.dtype)
        pred = jax.vmap(dyn_half)(z0.astype(jnp.bfloat16))
        return _mse(pred, z1)

    @eqx.filter_jit
    def train_step(dynamics, opt_state, z0, z1, noise_key):
        loss, grads = loss_and_grad(dynamics, z0, z1, noise_key)
        updates, opt_state = optimizer.update(grads, opt_state, dynamics)
        dynamics = eqx.apply_updates(dynamics, updates)
        return dynamics, opt_state, loss

    return train_step


@eqx.filter_jit
def val_step(dynamics, decoder, z0, z1, x1):
    """One-step val in f32 (T3 rollout precision policy): latent + pixel MSE."""
    pred_z = jax.vmap(dynamics)(z0)
    pred_x = jax.vmap(decoder)(pred_z)
    return _mse(pred_z, z1), _mse(pred_x, x1)


# ---------- Main ----------


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices,), ("batch",))
    jax.sharding.set_mesh(mesh)
    print(f"Using {num_devices} device(s)")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({args.batch_size}) must be divisible by "
            f"number of devices ({num_devices})")

    # --- Frozen VQ-VAE anchor (quantizer bypassed) ---
    print(f"Loading frozen VQ-VAE from {args.vqvae_checkpoint}")
    key, k_vq = jax.random.split(key)
    encoder, decoder, _, _, vq_arch = load_vqvae_checkpoint(
        args.vqvae_checkpoint, k_vq)
    encoder = eqx.nn.inference_mode(encoder)
    decoder = eqx.nn.inference_mode(decoder)
    latent_dim = vq_arch["codebook_dim"]
    print(f"Anchor: {vq_arch.get('scales')} scales ignored — continuous z, "
          f"latent_dim={latent_dim}")

    print("Loading data...")
    dataset = FramePairDataset(
        path=args.data_path, field=args.field, batch_size=args.batch_size,
        horizon=1, shuffle=True, seed=args.seed, mesh=mesh,
        sample_start=args.sample_start, sample_stop=args.sample_stop)
    print(f"Dataset: {dataset.n_pairs} frame pairs, shape {dataset.sample_shape}")

    val_dataset = None
    if args.val_stop > args.val_start:
        val_dataset = FramePairDataset(
            path=args.data_path, field=args.field, batch_size=args.batch_size,
            horizon=1, shuffle=False, seed=args.seed, mesh=mesh,
            sample_start=args.val_start, sample_stop=args.val_stop)
        print(f"Val: {val_dataset.n_pairs} pairs "
              f"(frames {args.val_start}-{args.val_stop})")

    # Latent grid size from a probe encode of one frame
    z_probe = np.asarray(encode_batch(
        encoder, jnp.asarray(dataset.data[:1], dtype=jnp.float32)))
    latent_hw = z_probe.shape[-1]
    print(f"Latent grid: ({latent_dim}, {latent_hw}, {latent_hw})")

    # Per-channel latent std probe (noise calibration; logged regardless)
    latent_std_ch = probe_latent_std(
        encoder, dataset, args.probe_frames, args.batch_size)
    print(f"Latent std probe ({min(args.probe_frames, dataset.data.shape[0])} "
          f"frames): mean over channels = {latent_std_ch.mean():.5f}")
    noise_std_ch = None
    if args.noise_sigma > 0:
        noise_std_ch = jnp.asarray(args.noise_sigma * latent_std_ch)
        print(f"Input noise: sigma = {args.noise_sigma} x per-channel std "
              f"(mean abs = {float(noise_std_ch.mean()):.5f})")

    arch_config = {
        "model_type": "latent_mse",
        "n_layer": args.n_layer,
        "n_head": args.n_head,
        "n_embd": args.n_embd,
        "rope_theta": args.rope_theta,
        "latent_dim": latent_dim,
        "latent_hw": latent_hw,
        "vqvae_checkpoint": args.vqvae_checkpoint,
    }

    key, k_model = jax.random.split(key)
    dynamics = build_model(arch_config, k_model)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(dynamics, eqx.is_array)))
    print(f"Dynamics: {n_params / 1e6:.1f}M params "
          f"(s18-class target; frozen enc/dec not counted)")

    # Test forward pass
    test_out = jax.vmap(dynamics)(jnp.zeros((1, latent_dim, latent_hw, latent_hw)))
    print(f"Test forward pass OK — output {test_out.shape}")

    steps_per_epoch = len(dataset)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(1000, total_steps // 10)
    print(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=warmup_steps,
        decay_steps=total_steps, end_value=args.lr * 0.01)
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optimizer.init(eqx.filter(dynamics, eqx.is_array))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        state_path = os.path.join(args.checkpoint_dir, "training_state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Cannot resume: no training_state.json in {args.checkpoint_dir}")
        with open(state_path) as f:
            training_state = json.load(f)

        saved_arch = training_state.get("arch_config")
        if saved_arch is not None:
            mismatches = []
            for k, current_val in arch_config.items():
                saved_val = saved_arch.get(k)
                if saved_val is not None and saved_val != current_val:
                    mismatches.append(f"  {k}: checkpoint={saved_val}, current={current_val}")
            if mismatches:
                raise ValueError(
                    "Cannot resume: architecture mismatch:\n" + "\n".join(mismatches))

        start_epoch = training_state["epoch"]
        global_step = training_state["global_step"]

        dynamics = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "dynamics.eqx"), dynamics)
        opt_state = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "opt_state.eqx"), opt_state)

        replicated = NamedSharding(mesh, P())
        dynamics = jax.device_put(dynamics, replicated)
        opt_state = jax.device_put(opt_state, replicated)
        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    config = {
        **arch_config,
        "data_path": args.data_path,
        "field": args.field,
        "n_pairs": dataset.n_pairs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "num_devices": num_devices,
        "n_params": n_params,
        "noise_sigma": args.noise_sigma,
        "latent_std_mean": float(latent_std_ch.mean()),
        "probe_frames": args.probe_frames,
        "sample_start": args.sample_start,
        "sample_stop": args.sample_stop,
    }

    if WANDB_AVAILABLE:
        if args.wandb_dir is not None:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = args.wandb_dir
        wandb_kwargs = dict(project=args.wandb_project, name=args.wandb_name,
                            config=config)
        if args.wandb_id is not None:
            wandb_kwargs["id"] = args.wandb_id
            wandb_kwargs["resume"] = "allow"
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}: {v}\n")
    np.save(os.path.join(args.checkpoint_dir, "latent_std_ch.npy"), latent_std_ch)

    train_step = make_train_step(optimizer, noise_std_ch)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_losses = []

        for batch_idx, frames in enumerate(dataset):
            z0 = encode_batch(encoder, frames[0])
            z1 = encode_batch(encoder, frames[1])
            key, noise_key = jax.random.split(key)

            dynamics, opt_state, loss = train_step(
                dynamics, opt_state, z0, z1, noise_key)

            loss_f = float(loss)
            epoch_losses.append(loss_f)

            if WANDB_AVAILABLE:
                wandb.log({"loss/latent_mse": loss_f, "step": global_step})

            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}: LatentMSE={loss_f:.5f}")

            global_step += 1

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch + 1} Summary: LatentMSE={avg_loss:.5f}")

        val_latent = val_pixel = None
        if val_dataset is not None:
            latent_losses, pixel_losses = [], []
            for vf in val_dataset:
                vz0 = encode_batch(encoder, vf[0])
                vz1 = encode_batch(encoder, vf[1])
                lm, pm = val_step(dynamics, decoder, vz0, vz1, vf[1])
                latent_losses.append(float(lm))
                pixel_losses.append(float(pm))
            if latent_losses:
                val_latent = float(np.mean(latent_losses))
                val_pixel = float(np.mean(pixel_losses))
                print(f"  Val one-step: latent MSE {val_latent:.5f}, "
                      f"pixel MSE {val_pixel:.5f}")

        if WANDB_AVAILABLE:
            x0_last = np.array(frames[0])
            x1_last = np.array(frames[1])
            pred_z = jax.vmap(dynamics)(encode_batch(encoder, frames[0]))
            pred_last = np.array(decode_batch(decoder, pred_z))
            pred_fig = plot_prediction(x0_last, pred_last, x1_last)

            epoch_log = {
                "epoch/latent_mse": avg_loss,
                "epoch": epoch + 1,
                "predictions": wandb.Image(pred_fig),
            }
            if val_latent is not None:
                epoch_log["epoch/val_latent_mse"] = val_latent
                epoch_log["epoch/val_onestep_mse"] = val_pixel
            wandb.log(epoch_log)
            plt.close(pred_fig)

        wandb_dir = wandb.run.dir if (WANDB_AVAILABLE and wandb.run is not None) else None
        save_checkpoint(dynamics, opt_state, epoch + 1, global_step,
                        args.checkpoint_dir, wandb_dir, arch_config)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
