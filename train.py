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
from vq import MultiScaleVQ, EMAState, init_ema_state, ema_update, vqvae_loss_simple


def parse_args():
    parser = argparse.ArgumentParser(description="Train ViT VQ-VAE on 2D field data")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .h5 file")
    parser.add_argument("--field", type=str, default="omega",
                        help="HDF5 field name under /fields/")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Model architecture
    parser.add_argument("--d_model", type=int, default=512, help="Transformer hidden dim")
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--mlp_dim", type=int, default=1024)
    parser.add_argument("--encoder_depth", type=int, default=6)
    parser.add_argument("--decoder_depth", type=int, default=6)
    parser.add_argument("--codebook_dim", type=int, default=64)
    parser.add_argument("--codebook_size", type=int, default=512, help="Number of codebook vectors")
    parser.add_argument("--scales", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated VQ scales")
    parser.add_argument("--rope_theta", type=float, default=32.0)
    # Training
    parser.add_argument("--beta", type=float, default=0.25, help="Commitment loss weight")
    parser.add_argument("--ema_decay", type=float, default=0.99, help="EMA decay for codebook")
    parser.add_argument("--seed", type=int, default=42)
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint in checkpoint_dir")
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="gust2-vqvae")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="Wandb run ID for resuming a run across jobs")
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None,
                        help="Directory for wandb logs (set to avoid home quota)")
    return parser.parse_args()


# ---------- Plotting ----------


def plot_reconstruction(inputs, outputs):
    """Plot input vs reconstruction comparison and return figure."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(min(4, len(inputs))):
        axes[0, i].imshow(inputs[i, 0], cmap='RdBu_r')
        axes[0, i].set_title(f"Input {i}")
        axes[0, i].axis('off')
        axes[1, i].imshow(outputs[i, 0], cmap='RdBu_r')
        axes[1, i].set_title(f"Recon {i}")
        axes[1, i].axis('off')
    plt.tight_layout()
    return fig


def plot_codebook_usage(indices, codebook_size):
    """Plot histogram of codebook usage and return figure."""
    flat_indices = indices.flatten()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(np.array(flat_indices), bins=codebook_size, range=(0, codebook_size), density=True)
    ax.axhline(y=1/codebook_size, color='r', linestyle='--', label='Uniform')
    ax.set_xlabel("Codebook Index")
    ax.set_ylabel("Frequency")
    ax.set_title("Codebook Usage Distribution")
    ax.legend()
    plt.tight_layout()
    return fig


# ---------- Checkpointing ----------


def save_checkpoint(encoder, decoder, vq, ema_state, opt_state,
                    epoch, global_step, checkpoint_dir,
                    wandb_dir=None, arch_config=None):
    os.makedirs(checkpoint_dir, exist_ok=True)

    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "encoder.eqx"), encoder)
    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "decoder.eqx"), decoder)
    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "vq.eqx"), vq)
    eqx.tree_serialise_leaves(os.path.join(checkpoint_dir, "opt_state.eqx"), opt_state)

    # EMA state is plain arrays — save as npz
    np.savez(os.path.join(checkpoint_dir, "ema_state.npz"),
             cluster_sizes=np.array(ema_state.cluster_sizes),
             codebook_sums=np.array(ema_state.codebook_sums),
             codebook=np.array(ema_state.codebook))

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
        eqx.tree_serialise_leaves(os.path.join(wandb_dir, "vq.eqx"), vq)
        np.savez(os.path.join(wandb_dir, "ema_state.npz"),
                 cluster_sizes=np.array(ema_state.cluster_sizes),
                 codebook_sums=np.array(ema_state.codebook_sums),
                 codebook=np.array(ema_state.codebook))


def load_ema_state(path):
    data = np.load(path)
    return EMAState(
        cluster_sizes=jnp.array(data["cluster_sizes"]),
        codebook_sums=jnp.array(data["codebook_sums"]),
        codebook=jnp.array(data["codebook"]),
    )


# ---------- Mixed precision ----------


def _cast_to_half(model):
    """Cast model float arrays to bfloat16. Master weights stay f32 outside."""
    def cast(x):
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(jnp.bfloat16)
        return x
    return jax.tree.map(cast, model)


# ---------- Training step ----------


@eqx.filter_value_and_grad(has_aux=True)
def compute_loss_and_grad(model, codebook, x_batch, beta):
    encoder, decoder, vq = model
    # Encoder/decoder in bf16, VQ stays f32 (quantization needs precision)
    model_bf16 = (_cast_to_half(encoder), _cast_to_half(decoder), vq)
    return vqvae_loss_simple(model_bf16, codebook,
                             x_batch.astype(jnp.bfloat16), beta)


@eqx.filter_jit
def train_step(model, codebook, optimizer, opt_state, x_batch, beta, ema_state,
               ema_decay, ema_key):
    (total_loss, aux), grads = compute_loss_and_grad(model, codebook, x_batch, beta)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    # EMA codebook update (outside gradient boundary)
    # Pass batched outputs directly — ema_update reduces over leading dims
    ema_state = ema_update(ema_state, aux['all_indices'], aux['all_z_flat'],
                           decay=ema_decay, key=ema_key)

    return model, opt_state, ema_state, total_loss, aux


# ---------- Main ----------


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    # Device mesh for automatic SPMD sharding
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices,), ("batch",))
    jax.sharding.set_mesh(mesh)
    print(f"Using {num_devices} device(s)")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({args.batch_size}) must be divisible by "
            f"number of devices ({num_devices})"
        )

    # Load data
    from dataloader import VQVAEDataset
    print("Loading data...")
    dataset = VQVAEDataset(
        path=args.data_path,
        field=args.field,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        mesh=mesh,
    )
    print(f"Dataset: {dataset.n_samples} samples, shape {dataset.sample_shape}")

    # Parse scales
    scales = tuple(int(s) for s in args.scales.split(","))

    arch_config = {
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "mlp_dim": args.mlp_dim,
        "encoder_depth": args.encoder_depth,
        "decoder_depth": args.decoder_depth,
        "codebook_dim": args.codebook_dim,
        "codebook_size": args.codebook_size,
        "scales": list(scales),
        "rope_theta": args.rope_theta,
    }

    # Initialize model components
    key, k_enc, k_dec, k_vq, k_ema = jax.random.split(key, 5)
    encoder = Encoder(args.d_model, args.n_heads, args.mlp_dim, args.encoder_depth,
                      args.codebook_dim, in_channels=1, rope_theta=args.rope_theta, key=k_enc)
    decoder = Decoder(args.d_model, args.n_heads, args.mlp_dim, args.decoder_depth,
                      args.codebook_dim, out_channels=1, rope_theta=args.rope_theta, key=k_dec)
    vq = MultiScaleVQ(args.codebook_dim, scales=scales, key=k_vq)
    ema_state = init_ema_state(args.codebook_dim, args.codebook_size, key=k_ema)

    model = (encoder, decoder, vq)

    # Test forward pass
    test_input = jnp.zeros((1, 1, 256, 256))
    test_loss, test_aux = vqvae_loss_simple(model, ema_state.codebook, test_input, args.beta)
    print(f"Test forward pass OK — loss={float(test_loss):.4f}")

    # Optimizer with warmup + cosine decay
    steps_per_epoch = len(dataset)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(1000, total_steps // 10)
    print(f"LR schedule: {warmup_steps} warmup steps, {total_steps} total steps")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),
        optax.adamw(schedule, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        state_path = os.path.join(args.checkpoint_dir, "training_state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Cannot resume: no training_state.json in {args.checkpoint_dir}")

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
        vq = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "vq.eqx"), vq)
        ema_state = load_ema_state(os.path.join(args.checkpoint_dir, "ema_state.npz"))
        opt_state = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "opt_state.eqx"), opt_state)

        replicated = NamedSharding(mesh, P())
        model = jax.device_put((encoder, decoder, vq), replicated)
        encoder, decoder, vq = model
        opt_state = jax.device_put(opt_state, replicated)
        ema_state = jax.device_put(ema_state, replicated)

        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    model = (encoder, decoder, vq)

    # Config for logging
    config = {
        **arch_config,
        "data_path": args.data_path,
        "field": args.field,
        "n_samples": dataset.n_samples,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "beta": args.beta,
        "ema_decay": args.ema_decay,
        "epochs": args.epochs,
        "seed": args.seed,
        "num_devices": num_devices,
    }

    # Initialize wandb
    if WANDB_AVAILABLE:
        if args.wandb_dir is not None:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = args.wandb_dir
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
        )
        if args.wandb_id is not None:
            wandb_kwargs["id"] = args.wandb_id
            wandb_kwargs["resume"] = "allow"
        if args.wandb_group is not None:
            wandb_kwargs["group"] = args.wandb_group
        wandb.init(**wandb_kwargs)

    # Save config
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "config.txt"), "w") as f:
        for k, v in sorted(config.items()):
            f.write(f"{k}: {v}\n")

    if WANDB_AVAILABLE and wandb.run is not None:
        config_path = os.path.join(wandb.run.dir, "config.txt")
        with open(config_path, "w") as f:
            for k, v in sorted(config.items()):
                f.write(f"{k}: {v}\n")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_losses = []
        epoch_recon_losses = []
        epoch_commit_losses = []

        for batch_idx, x_batch in enumerate(dataset):
            key, step_key, ema_key = jax.random.split(key, 3)

            model, opt_state, ema_state, total_loss, aux = train_step(
                model, ema_state.codebook, optimizer, opt_state, x_batch,
                args.beta, ema_state, args.ema_decay, ema_key,
            )

            total_loss_f = float(total_loss)
            recon_loss_f = float(aux['recon_loss'])
            commit_loss_f = float(aux['commit_loss'])
            epoch_losses.append(total_loss_f)
            epoch_recon_losses.append(recon_loss_f)
            epoch_commit_losses.append(commit_loss_f)

            # Count unique codes per scale
            per_scale_unique = []
            for idx in aux['all_indices']:
                per_scale_unique.append(len(np.unique(np.array(idx).flatten())))

            if WANDB_AVAILABLE:
                log_dict = {
                    "loss/total": total_loss_f,
                    "loss/reconstruction": recon_loss_f,
                    "loss/commitment": commit_loss_f,
                    "codebook/unique_codes": sum(per_scale_unique),
                    "step": global_step,
                }
                for si, s in enumerate(scales):
                    log_dict[f"codebook/unique_codes_scale_{s}"] = per_scale_unique[si]
                    log_dict[f"codebook/utilization_scale_{s}"] = per_scale_unique[si] / args.codebook_size
                wandb.log(log_dict)

            if batch_idx % 50 == 0:
                scale_str = " ".join(f"s{s}:{n}" for s, n in zip(scales, per_scale_unique))
                print(f"  Batch {batch_idx}: Loss={total_loss_f:.4f}, "
                      f"Recon={recon_loss_f:.4f}, Commit={commit_loss_f:.4f} "
                      f"[{scale_str}]")

            global_step += 1

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_recon = np.mean(epoch_recon_losses)
        avg_commit = np.mean(epoch_commit_losses)
        print(f"Epoch {epoch + 1} Summary: Loss={avg_loss:.4f}, "
              f"Recon={avg_recon:.4f}, Commit={avg_commit:.4f}")

        if WANDB_AVAILABLE:
            # Reconstruct last batch for visualization
            encoder, decoder, vq_mod = model
            def fwd_single(x):
                z_e = encoder(x)
                z_q, _, _, _, _ = vq_mod(z_e, ema_state.codebook)
                return decoder(z_q)
            outputs = jax.vmap(fwd_single)(x_batch)

            recon_fig = plot_reconstruction(np.array(x_batch), np.array(outputs))
            all_idx = np.concatenate([np.array(idx).flatten() for idx in aux['all_indices']])
            usage_fig = plot_codebook_usage(all_idx, args.codebook_size)

            wandb.log({
                "epoch/loss": avg_loss,
                "epoch/reconstruction": avg_recon,
                "epoch/commitment": avg_commit,
                "epoch": epoch + 1,
                "reconstructions": wandb.Image(recon_fig),
                "codebook_usage": wandb.Image(usage_fig),
            })
            plt.close(recon_fig)
            plt.close(usage_fig)

        # Save checkpoint
        encoder, decoder, vq_mod = model
        wandb_dir = wandb.run.dir if (WANDB_AVAILABLE and wandb.run is not None) else None
        save_checkpoint(encoder, decoder, vq_mod, ema_state, opt_state,
                        epoch + 1, global_step, args.checkpoint_dir,
                        wandb_dir, arch_config)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
