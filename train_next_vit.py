"""B2 baseline — ViT next-frame regressor (BASELINES_SPEC.md §B2).

The tokenizer backbone with the tokenization/quantization machinery stripped
out: the vit_ae Encoder/Decoder pair wired directly together and trained
end-to-end to regress frame t+1 from frame t with MSE. Same blocks, same
8x strided-conv stem, same 32x32 latent grid as the VQ-VAE — the pixel-space
continuous cell of the representation x prediction matrix, architecture-
matched to the discrete pipeline so quantize-or-don't (and AR-or-regress)
are the only changed variables.

Stabilizer variants (each an arm in the baseline grid):
  --noise_sigma S    Gaussian input noise, std = S x train-set pixel std
                     (Stachenfeld-style).
  --pushforward      2-step unroll with the gradient detached through step 1
                     (MP-PDE pushforward trick). Loss = MSE(f(x0), x1)
                     + MSE(f(sg(f(x0))), x2); needs frame triples and ~2x
                     activation memory (use 2 GPUs at batch 64).
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

from vit_ae import Encoder, Decoder
from dataloader import FramePairDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ViT next-frame regressor (B2 continuous-pixel baseline)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .h5 file")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name under /fields/")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Model architecture (defaults = Small tokenizer arch)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--encoder_depth", type=int, default=5)
    parser.add_argument("--decoder_depth", type=int, default=5)
    parser.add_argument("--latent_dim", type=int, default=512,
                        help="Encoder output channels (= codebook_dim slot in "
                             "the VQ-VAE; kept for architecture parity)")
    parser.add_argument("--rope_theta", type=float, default=32.0)
    # Baseline variants
    parser.add_argument("--noise_sigma", type=float, default=0.0,
                        help="Training input-noise std as a FRACTION of the "
                             "train-set pixel std (0 = plain MSE)")
    parser.add_argument("--pushforward", action="store_true",
                        help="Add the MP-PDE pushforward term: 2-step unroll, "
                             "gradient detached through step 1")
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


# ---------- Mixed precision ----------


def _cast_to_half(model):
    """Cast model float arrays to bfloat16. Master weights stay f32 outside."""
    def cast(x):
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.bfloat16)
        return x
    return jax.tree.map(cast, model)


def batched_predict(model, xb):
    """(B, 1, H, W) -> (B, 1, H, W), single next-frame step."""
    encoder, decoder = model
    return jax.vmap(lambda s: decoder(encoder(s)))(xb)


def _mse(pred, target):
    return jnp.mean((pred.astype(jnp.float32) - target.astype(jnp.float32)) ** 2)


# ---------- Checkpointing ----------


def save_checkpoint(encoder, decoder, opt_state, epoch, global_step,
                    checkpoint_dir, wandb_dir=None, arch_config=None):
    os.makedirs(checkpoint_dir, exist_ok=True)

    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "encoder.eqx"), encoder)
    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "decoder.eqx"), decoder)
    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "opt_state.eqx"), opt_state)

    state = {"epoch": epoch, "global_step": global_step}
    if arch_config is not None:
        state["arch_config"] = arch_config
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(state, f)

    print(f"Saved checkpoint (epoch {epoch}) to {checkpoint_dir}")

    if wandb_dir is not None:
        os.makedirs(wandb_dir, exist_ok=True)
        eqx.tree_serialise_leaves(os.path.join(wandb_dir, "encoder.eqx"), encoder)
        eqx.tree_serialise_leaves(os.path.join(wandb_dir, "decoder.eqx"), decoder)


def build_model(arch_config, key):
    """Build an (encoder, decoder) skeleton from an arch_config dict.

    Shared with rollout_continuous.py so checkpoint loading can't drift.
    """
    k_enc, k_dec = jax.random.split(key, 2)
    encoder = Encoder(arch_config["d_model"], arch_config["n_heads"],
                      arch_config["mlp_dim"], arch_config["encoder_depth"],
                      arch_config["latent_dim"], in_channels=1,
                      rope_theta=arch_config["rope_theta"], key=k_enc)
    decoder = Decoder(arch_config["d_model"], arch_config["n_heads"],
                      arch_config["mlp_dim"], arch_config["decoder_depth"],
                      arch_config["latent_dim"], out_channels=1,
                      rope_theta=arch_config["rope_theta"], key=k_dec)
    return encoder, decoder


# ---------- Plotting ----------


def plot_prediction(x0, pred, x1):
    """Input t0 / predicted t1 / target t1 / error rows for up to 4 samples."""
    n = min(4, len(x0))
    fig, axes = plt.subplots(4, n, figsize=(4 * n, 16))
    if n == 1:
        axes = axes[:, None]
    for i in range(n):
        err = pred[i, 0] - x1[i, 0]
        for row, (img, title) in enumerate([
            (x0[i, 0], "Input t"),
            (pred[i, 0], "Pred t+1"),
            (x1[i, 0], "Target t+1"),
            (err, "Error"),
        ]):
            cmap = "RdBu_r"
            axes[row, i].imshow(img, cmap=cmap)
            axes[row, i].set_title(f"{title} ({i})")
            axes[row, i].axis("off")
    plt.tight_layout()
    return fig


# ---------- Training step ----------


def make_train_step(optimizer, noise_std_abs, pushforward):
    """Build the jitted train step; noise/pushforward resolve at trace time."""

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_and_grad(model, frames, noise_key):
        encoder, decoder = model
        model_half = (_cast_to_half(encoder), _cast_to_half(decoder))
        x0 = frames[0]
        if noise_std_abs > 0.0:
            x0 = x0 + noise_std_abs * jax.random.normal(
                noise_key, x0.shape, dtype=x0.dtype)
        pred1 = batched_predict(model_half, x0.astype(jnp.bfloat16))
        onestep = _mse(pred1, frames[1])
        aux = {"onestep_mse": onestep}
        if pushforward:
            pred1_sg = jax.lax.stop_gradient(pred1)
            pred2 = batched_predict(model_half, pred1_sg)
            aux["pushforward_mse"] = _mse(pred2, frames[2])
            total = onestep + aux["pushforward_mse"]
        else:
            total = onestep
        return total, aux

    @eqx.filter_jit
    def train_step(model, opt_state, frames, noise_key):
        (total, aux), grads = loss_and_grad(model, frames, noise_key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, total, aux

    return train_step


@eqx.filter_jit
def val_step(model, x0, x1):
    encoder, decoder = model
    model_half = (_cast_to_half(encoder), _cast_to_half(decoder))
    pred = batched_predict(model_half, x0.astype(jnp.bfloat16))
    return _mse(pred, x1)


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

    horizon = 2 if args.pushforward else 1
    print("Loading data...")
    dataset = FramePairDataset(
        path=args.data_path, field=args.field, batch_size=args.batch_size,
        horizon=horizon, shuffle=True, seed=args.seed, mesh=mesh,
        sample_start=args.sample_start, sample_stop=args.sample_stop)
    train_pixel_std = dataset.pixel_std
    noise_std_abs = args.noise_sigma * train_pixel_std
    print(f"Dataset: {dataset.n_pairs} frame {'triples' if horizon == 2 else 'pairs'}, "
          f"shape {dataset.sample_shape}, pixel std {train_pixel_std:.4f}")
    if args.noise_sigma > 0:
        print(f"Input noise: sigma = {args.noise_sigma} x std = {noise_std_abs:.5f}")

    val_dataset = None
    if args.val_stop > args.val_start:
        val_dataset = FramePairDataset(
            path=args.data_path, field=args.field, batch_size=args.batch_size,
            horizon=1, shuffle=False, seed=args.seed, mesh=mesh,
            sample_start=args.val_start, sample_stop=args.val_stop)
        print(f"Val: {val_dataset.n_pairs} pairs "
              f"(frames {args.val_start}-{args.val_stop})")

    arch_config = {
        "model_type": "next_vit",
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "mlp_dim": args.mlp_dim,
        "encoder_depth": args.encoder_depth,
        "decoder_depth": args.decoder_depth,
        "latent_dim": args.latent_dim,
        "rope_theta": args.rope_theta,
    }

    key, k_model = jax.random.split(key)
    encoder, decoder = build_model(arch_config, k_model)
    model = (encoder, decoder)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {n_params / 1e6:.1f}M params")

    # Test forward pass
    C, H, W = dataset.sample_shape
    test_out = batched_predict(model, jnp.zeros((1, C, H, W)))
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
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

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

        encoder = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "encoder.eqx"), encoder)
        decoder = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "decoder.eqx"), decoder)
        opt_state = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "opt_state.eqx"), opt_state)

        replicated = NamedSharding(mesh, P())
        model = jax.device_put((encoder, decoder), replicated)
        opt_state = jax.device_put(opt_state, replicated)
        print(f"Resumed from epoch {start_epoch}, global step {global_step}")
    else:
        model = (encoder, decoder)

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
        "noise_std_abs": noise_std_abs,
        "pushforward": args.pushforward,
        "sample_start": args.sample_start,
        "sample_stop": args.sample_stop,
        "train_pixel_std": train_pixel_std,
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

    train_step = make_train_step(optimizer, noise_std_abs, args.pushforward)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_losses = []
        epoch_onestep = []
        epoch_push = []

        for batch_idx, frames in enumerate(dataset):
            key, noise_key = jax.random.split(key)

            model, opt_state, total_loss, aux = train_step(
                model, opt_state, frames, noise_key)

            total_loss_f = float(total_loss)
            onestep_f = float(aux["onestep_mse"])
            epoch_losses.append(total_loss_f)
            epoch_onestep.append(onestep_f)
            if args.pushforward:
                epoch_push.append(float(aux["pushforward_mse"]))

            if WANDB_AVAILABLE:
                log_dict = {
                    "loss/total": total_loss_f,
                    "loss/onestep_mse": onestep_f,
                    "step": global_step,
                }
                if args.pushforward:
                    log_dict["loss/pushforward_mse"] = epoch_push[-1]
                wandb.log(log_dict)

            if batch_idx % 50 == 0:
                push_str = (f", Push={epoch_push[-1]:.5f}"
                            if args.pushforward else "")
                print(f"  Batch {batch_idx}: Loss={total_loss_f:.5f}, "
                      f"OneStep={onestep_f:.5f}{push_str}")

            global_step += 1

        avg_loss = np.mean(epoch_losses)
        avg_onestep = np.mean(epoch_onestep)
        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.5f}, "
              f"OneStep={avg_onestep:.5f}")

        val_mse = None
        if val_dataset is not None:
            val_losses = [float(val_step(model, vf[0], vf[1]))
                          for vf in val_dataset]
            val_mse = float(np.mean(val_losses))
            print(f"  Val one-step MSE: {val_mse:.5f}")

        if WANDB_AVAILABLE:
            x0_last = np.array(frames[0])
            x1_last = np.array(frames[1])
            pred_last = np.array(batched_predict(model, frames[0]))
            pred_fig = plot_prediction(x0_last, pred_last, x1_last)

            epoch_log = {
                "epoch/loss": avg_loss,
                "epoch/onestep_mse": avg_onestep,
                "epoch": epoch + 1,
                "predictions": wandb.Image(pred_fig),
            }
            if args.pushforward:
                epoch_log["epoch/pushforward_mse"] = float(np.mean(epoch_push))
            if val_mse is not None:
                epoch_log["epoch/val_onestep_mse"] = val_mse
            wandb.log(epoch_log)
            plt.close(pred_fig)

        encoder, decoder = model
        wandb_dir = wandb.run.dir if (WANDB_AVAILABLE and wandb.run is not None) else None
        save_checkpoint(encoder, decoder, opt_state, epoch + 1, global_step,
                        args.checkpoint_dir, wandb_dir, arch_config)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
