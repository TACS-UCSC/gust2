"""
Teacher-Forced Next-Scale Prediction (NSP) training.

VAR-style approach: predicts each scale k from scale k-1's hidden states
(bilinear upsampled), with teacher forcing at all scales. Trains all
trainable scales in a single forward pass.

Sequence layout: [full t0, truncated t1] where t1 excludes the last scale.
Prediction: coarser-scale hidden states → bilinear upsample → expansion head.

Usage:
    python train_nsp.py --tokens_path experiments/tokens/small-sc341.npz \
        --n_layer 6 --n_head 8 --n_embd 256 --batch_size 32 --epochs 100
"""

import argparse
import json
import math
import os

import jax
# Required for vmap+random under explicit-axis meshes in JAX >= 0.5.
jax.config.update("jax_threefry_partitionable", False)
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import optax
import equinox as eqx
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed.")

from nsp_model import (
    NSPConfig, NSPModel, ExpansionHeads,
    create_nsp_model, forward_teacher_forced,
    build_teacher_forced_mask, build_rope_coords,
    _local_cell_coords,
)
from tokenizer import load_tokenized_data


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Train NSP (teacher-forced)")
    # Data
    parser.add_argument("--tokens_path", type=str, required=True,
                        help="Path to tokenized .npz file")
    # Model
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--rope_theta", type=float, default=None,
                        help="RoPE base (default: auto = finest grid side)")
    parser.add_argument("--n_refine_layers", type=int, default=2,
                        help="Within-scale refinement blocks in ExpansionHeads")
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=None,
                        help="Warmup steps (default: min(1000, total//10))")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Use only first N frames (default: all)")
    parser.add_argument("--context_drop_rate", type=float, default=0.0,
                        help="Per-token dropout rate for context (t0 + t1 coarser)")
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_nsp")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="gust2-nsp")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_id", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


# =============================================================================
# Paired dataloader
# =============================================================================


def create_paired_dataloader(data, batch_size, sharding=None,
                             shuffle=True, seed=0):
    """Yield batches of (B, 2 * tokens_per_frame) by pairing consecutive frames.

    Args:
        data: (N, tokens_per_frame) array of token indices
        batch_size: batch size
        sharding: optional NamedSharding for SPMD
        shuffle: whether to shuffle frame pairs
        seed: random seed for shuffling
    """
    n_pairs = len(data) - 1
    indices = np.arange(n_pairs)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for i in range(0, n_pairs - batch_size + 1, batch_size):
        batch_idx = indices[i:i + batch_size]
        t0 = data[batch_idx]
        t1 = data[batch_idx + 1]
        batch = jnp.array(np.concatenate([t0, t1], axis=1))
        if sharding is not None:
            batch = jax.device_put(batch, sharding)
        yield batch


# =============================================================================
# Mixed precision
# =============================================================================


def _cast_to_half(x):
    """Cast float arrays to bfloat16. Master weights stay f32 outside."""
    def cast(a):
        if eqx.is_array(a) and jnp.issubdtype(a.dtype, jnp.floating):
            return a.astype(jnp.bfloat16)
        return a
    return jax.tree.map(cast, x)


# =============================================================================
# Loss function
# =============================================================================


def make_compute_loss(config, scales_t0, padded_len_t0,
                      scales_t1, padded_len_t1,
                      attn_bias, scale_weights, trainable_indices,
                      scale_masks, context_drop_rate=0.0):
    """Build the loss function capturing static config.

    Returns: (model, exp_heads, batch_tokens, key) -> (loss, metrics)
    """
    tokens_per_frame = config.tokens_per_frame

    # Boundaries for t0 (full frame)
    boundaries_t0 = [0]
    for h, w in scales_t0:
        boundaries_t0.append(boundaries_t0[-1] + h * w)

    # Boundaries for t1 (truncated — last scale excluded from input)
    boundaries_t1 = [0]
    for h, w in scales_t1:
        boundaries_t1.append(boundaries_t1[-1] + h * w)

    # Boundaries for full frame (for target extraction)
    boundaries_full = config.scale_boundaries

    # Total tokens in truncated t1
    tokens_t1_trunc = sum(h * w for h, w in scales_t1)

    # Total sequence length (for dropout mask)
    seq_len = padded_len_t0 + padded_len_t1

    def compute_loss(model, exp_heads, batch_tokens, key):
        B = batch_tokens.shape[0]

        # Split into t0 and t1 (full, for targets)
        t0_full = batch_tokens[:, :tokens_per_frame]
        t1_full = batch_tokens[:, tokens_per_frame:]

        # Build truncated t1 input (exclude last scale)
        t1_trunc = t1_full[:, :tokens_t1_trunc]

        # Pad to 128-multiples
        t0_pad = jnp.pad(t0_full,
                         ((0, 0), (0, padded_len_t0 - tokens_per_frame)))
        t1_pad = jnp.pad(t1_trunc,
                         ((0, 0), (0, padded_len_t1 - tokens_t1_trunc)))
        tokens_in = jnp.concatenate([t0_pad, t1_pad], axis=1)

        # Forward pass via vmap. Dropout mask is derived per-sample
        # inside the mapped function (via axis_index + fold_in) so the
        # mapped inputs all share the same sharding spec.
        def per_sample_forward(t):
            if context_drop_rate > 0:
                k = jax.random.fold_in(key, jax.lax.axis_index('batch'))
                drop_mask = jax.random.bernoulli(
                    k, 1.0 - context_drop_rate, shape=(seq_len,)
                ).astype(jnp.float32)
            else:
                drop_mask = None
            return forward_teacher_forced(
                model, t, config,
                scales_t0, padded_len_t0,
                scales_t1, padded_len_t1,
                attn_bias, drop_mask=drop_mask,
            )

        hidden = jax.vmap(per_sample_forward, axis_name='batch')(tokens_in)
        # hidden: (B, L0+L1, n_embd)

        h_t0 = hidden[:, :padded_len_t0, :]
        h_t1 = hidden[:, padded_len_t0:, :]

        total_loss = jnp.float32(0.0)
        per_scale_losses = []
        per_scale_accs = []

        for i, scale_idx in enumerate(trainable_indices):
            h_k, w_k = config.scales[scale_idx]
            n_tokens_k = h_k * w_k

            # Source hidden states: scale k-1
            if scale_idx == 0:
                # Scale 0: source from t0's scale 0
                src_start = boundaries_t0[0]
                src_end = boundaries_t0[1]
                h_src, w_src = config.scales[0]
                h_source = h_t0[:, src_start:src_end, :]
            else:
                # Source: scale k-1 from t1 portion
                src_scale_in_t1 = scale_idx - 1
                src_start = boundaries_t1[src_scale_in_t1]
                src_end = boundaries_t1[src_scale_in_t1 + 1]
                h_src, w_src = scales_t1[src_scale_in_t1]
                h_source = h_t1[:, src_start:src_end, :]

            h_source_2d = h_source.reshape(B, h_src, w_src, config.n_embd)

            # Bilinear upsample to target resolution
            h_upsampled = jax.vmap(
                lambda x: jax.image.resize(
                    x, (h_k, w_k, config.n_embd), method='bilinear')
            )(h_source_2d)  # (B, h_k, w_k, n_embd)

            # Target position encoding
            rows = jnp.arange(h_k, dtype=jnp.float32) / max(h_k - 1, 1)
            cols = jnp.arange(w_k, dtype=jnp.float32) / max(w_k - 1, 1)
            grid_r, grid_c = jnp.meshgrid(rows, cols, indexing='ij')
            coords = jnp.stack([grid_r, grid_c], axis=-1)  # (h_k, w_k, 2)

            pos_emb = jax.vmap(jax.vmap(exp_heads.pos_proj))(coords)
            h_positioned = h_upsampled + pos_emb[None]  # (B, h_k, w_k, n_embd)

            # Apply expansion (refinement + head) → logits
            h_flat = h_positioned.reshape(B, n_tokens_k, config.n_embd)
            local_coords = _local_cell_coords(h_k, w_k)
            logits = jax.vmap(
                lambda hh, idx=i, lc=local_coords: exp_heads.expand(hh, idx, lc)
            )(h_flat)
            # (B, n_tokens_k, effective_vocab)

            # Apply scale mask: set invalid tokens to -1e9
            mask_k = scale_masks[scale_idx]  # (effective_vocab,) bool
            logits = jnp.where(mask_k[None, None, :], logits, -1e9)

            # Targets from full t1 (compact indices, no offset needed)
            tgt_start = boundaries_full[scale_idx]
            tgt_end = boundaries_full[scale_idx + 1]
            targets = t1_full[:, tgt_start:tgt_end]

            # Cross-entropy
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_log_probs = jnp.take_along_axis(
                log_probs, targets[:, :, None], axis=-1
            ).squeeze(-1)
            raw_loss = -jnp.mean(target_log_probs)
            weighted_loss = raw_loss * scale_weights[scale_idx]

            total_loss = total_loss + weighted_loss

            preds = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(preds == targets)

            per_scale_losses.append(raw_loss)
            per_scale_accs.append(accuracy)

        metrics = jnp.stack(per_scale_losses + per_scale_accs)
        return total_loss, metrics

    return compute_loss


# =============================================================================
# Train step
# =============================================================================


def make_train_step(compute_loss_fn):
    """Build compiled train step.

    Returns: step(model, exp_heads, opt_state, batch, optimizer, key)
             -> (model, exp_heads, opt_state, loss, metrics)
    """

    @eqx.filter_jit
    def step(model, exp_heads, opt_state, batch_tokens, optimizer, key):
        # Mixed precision: cast to bf16 inside grad boundary
        @eqx.filter_value_and_grad(has_aux=True)
        def loss_and_grad(model_eh):
            m, eh = model_eh
            m_bf16 = _cast_to_half(m)
            eh_bf16 = _cast_to_half(eh)
            return compute_loss_fn(m_bf16, eh_bf16, batch_tokens, key)

        (loss, metrics), grads = loss_and_grad((model, exp_heads))

        params = eqx.filter((model, exp_heads), eqx.is_inexact_array)

        # Replace None grads with zeros
        safe_grads = jax.tree.map(
            lambda g, p: jnp.zeros_like(p) if (g is None and p is not None) else g,
            grads, params, is_leaf=lambda x: x is None,
        )

        updates, opt_state = optimizer.update(safe_grads, opt_state, params)

        # Preserve None structure
        updates = jax.tree.map(
            lambda g, u: None if g is None else u,
            grads, updates, is_leaf=lambda x: x is None,
        )

        model_updates, exp_updates = updates
        model = eqx.apply_updates(model, model_updates)
        exp_heads = eqx.apply_updates(exp_heads, exp_updates)
        return model, exp_heads, opt_state, loss, metrics

    return step


# =============================================================================
# Checkpointing
# =============================================================================


def save_checkpoint(model, exp_heads, opt_state, epoch, global_step,
                    checkpoint_dir, arch_config=None):
    os.makedirs(checkpoint_dir, exist_ok=True)

    eqx.tree_serialise_leaves(
        os.path.join(checkpoint_dir, "model.eqx"), model)
    eqx.tree_serialise_leaves(
        os.path.join(checkpoint_dir, "exp_heads.eqx"), exp_heads)
    eqx.tree_serialise_leaves(
        os.path.join(checkpoint_dir, "opt_state.eqx"), opt_state)

    state = {"epoch": epoch, "global_step": global_step}
    if arch_config is not None:
        state["arch_config"] = arch_config
    with open(os.path.join(checkpoint_dir, "training_state.json"), "w") as f:
        json.dump(state, f)

    print(f"Saved checkpoint (epoch {epoch}) to {checkpoint_dir}")


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    key = jax.random.PRNGKey(args.seed)

    # Device mesh for SPMD
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices,), ("batch",))
    jax.sharding.set_mesh(mesh)
    print(f"Using {num_devices} device(s)")

    if args.batch_size % num_devices != 0:
        raise ValueError(
            f"Batch size ({args.batch_size}) must be divisible by "
            f"number of devices ({num_devices})")

    # Load tokenized data
    print(f"Loading data from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    if args.max_samples is not None:
        indices = indices[:args.max_samples]
    scales_int = token_data["scales"]
    scale_masks = jnp.array(token_data["scale_masks"])  # (n_scales, effective_vocab)
    tokens_per_frame = sum(s * s for s in scales_int)

    print(f"Loaded {len(indices)} frames, {tokens_per_frame} tokens/frame")
    print(f"Scales: {list(scales_int)}")

    # Setup config
    config = NSPConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        rope_theta=args.rope_theta,
        n_refine_layers=args.n_refine_layers,
    )

    # Create model
    key, model_key = jax.random.split(key)
    model, exp_heads = create_nsp_model(token_data, config, model_key)

    trainable_indices = config.trainable_scale_indices
    print(f"Trainable scales: {[config.scales[i] for i in trainable_indices]}")

    # Architecture config for checkpoint validation
    arch_config = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_embd": config.n_embd,
        "scales": [list(s) for s in config.scales],
        "tokens_per_frame": config.tokens_per_frame,
        "effective_vocab_size": config.effective_vocab_size,
        "codebook_dim": config.codebook_dim,
        "first_trainable_scale": config.first_trainable_scale,
        "rope_theta": config.rope_theta,
        "n_refine_layers": config.n_refine_layers,
    }

    # Compute sequence lengths
    scales_t0 = config.scales              # all scales
    scales_t1 = config.scales[:-1]         # last scale excluded from input

    tokens_t0 = sum(h * w for h, w in scales_t0)
    tokens_t1 = sum(h * w for h, w in scales_t1)

    padded_len_t0 = ((tokens_t0 + 127) // 128) * 128
    padded_len_t1 = ((tokens_t1 + 127) // 128) * 128

    print(f"\nSequence layout:")
    print(f"  t0: {tokens_t0} tokens -> padded {padded_len_t0}")
    print(f"  t1 (truncated): {tokens_t1} tokens -> padded {padded_len_t1}")
    print(f"  Total: {padded_len_t0 + padded_len_t1}")

    # Build attention mask
    attn_bias = build_teacher_forced_mask(
        scales_t0, padded_len_t0, scales_t1, padded_len_t1)
    print(f"Attention mask: {attn_bias.shape}")

    # Per-scale loss weights: 1/log(token_count + 1), normalized.
    # Log decays much slower than 1/sqrt(N)=1/K, giving fine scales
    # (where the grid-artifact failure mode lives) more gradient signal.
    token_counts = [config.scales[i][0] * config.scales[i][1]
                    for i in trainable_indices]
    raw_weights = [1.0 / math.log(c + 1.0) for c in token_counts]
    mean_w = sum(raw_weights) / len(raw_weights)
    scale_weights = {idx: w / mean_w
                     for idx, w in zip(trainable_indices, raw_weights)}
    for idx, w in scale_weights.items():
        h, w_s = config.scales[idx]
        print(f"  Scale {h}x{w_s} ({h*w_s} tokens): loss weight = {w:.3f}")

    # Build loss and train step
    compute_loss_fn = make_compute_loss(
        config, scales_t0, padded_len_t0,
        scales_t1, padded_len_t1,
        attn_bias, scale_weights, trainable_indices,
        scale_masks, context_drop_rate=args.context_drop_rate,
    )
    train_step = make_train_step(compute_loss_fn)

    # Optimizer
    n_pairs = len(indices) - 1
    steps_per_epoch = n_pairs // args.batch_size
    total_steps = steps_per_epoch * args.epochs

    warmup_steps = args.warmup_steps
    if warmup_steps is None:
        warmup_steps = min(1000, total_steps // 10)
    print(f"\nLR schedule: {warmup_steps} warmup, {total_steps} total steps")

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=args.lr * 0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),
        optax.adamw(schedule, weight_decay=args.weight_decay),
    )

    params = eqx.filter((model, exp_heads), eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        state_path = os.path.join(args.checkpoint_dir, "training_state.json")
        if not os.path.exists(state_path):
            raise FileNotFoundError(
                f"Cannot resume: no training_state.json in {args.checkpoint_dir}")

        with open(state_path) as f:
            training_state = json.load(f)

        # Validate architecture
        saved_arch = training_state.get("arch_config")
        if saved_arch is not None:
            mismatches = []
            for k, current_val in arch_config.items():
                saved_val = saved_arch.get(k)
                if saved_val is not None and saved_val != current_val:
                    mismatches.append(
                        f"  {k}: checkpoint={saved_val}, current={current_val}")
            if mismatches:
                raise ValueError(
                    "Cannot resume: architecture mismatch:\n"
                    + "\n".join(mismatches))

        start_epoch = training_state["epoch"]
        global_step = training_state["global_step"]

        model = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "model.eqx"), model)
        exp_heads = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "exp_heads.eqx"), exp_heads)
        opt_state = eqx.tree_deserialise_leaves(
            os.path.join(args.checkpoint_dir, "opt_state.eqx"), opt_state)

        replicated = NamedSharding(mesh, P())
        model = jax.device_put(model, replicated)
        exp_heads = jax.device_put(exp_heads, replicated)
        opt_state = jax.device_put(opt_state, replicated)

        print(f"Resumed from epoch {start_epoch}, global step {global_step}")

    # Wandb
    all_config = {**vars(args), **arch_config}
    if WANDB_AVAILABLE:
        if args.wandb_dir is not None:
            os.makedirs(args.wandb_dir, exist_ok=True)
            os.environ["WANDB_DIR"] = args.wandb_dir
        wandb_kwargs = dict(
            project=args.wandb_project,
            name=args.wandb_name,
            config=all_config,
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
        for k, v in sorted(all_config.items()):
            f.write(f"{k}: {v}\n")

    # Dataloader sharding
    batch_sharding = NamedSharding(mesh, P("batch", None))

    # --- Training Loop ---
    n_trainable = len(trainable_indices)
    print(f"\nStarting training: {args.epochs} epochs, "
          f"{steps_per_epoch} steps/epoch, "
          f"{n_trainable} trainable scales")

    drop_key = jax.random.PRNGKey(args.seed + 1)

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        key, loader_key = jax.random.split(key)
        epoch_drop_key = jax.random.fold_in(drop_key, epoch)

        epoch_losses = []
        epoch_scale_losses = {idx: [] for idx in trainable_indices}
        epoch_scale_accs = {idx: [] for idx in trainable_indices}

        dataloader = create_paired_dataloader(
            indices, args.batch_size,
            sharding=batch_sharding,
            shuffle=True, seed=int(loader_key[0]),
        )

        for batch_idx, batch in enumerate(dataloader):
            step_key = jax.random.fold_in(epoch_drop_key, batch_idx)
            model, exp_heads, opt_state, loss, metrics = train_step(
                model, exp_heads, opt_state, batch, optimizer, step_key)

            loss_val = float(loss)
            epoch_losses.append(loss_val)

            scale_losses = metrics[:n_trainable]
            scale_accs = metrics[n_trainable:]

            for i, idx in enumerate(trainable_indices):
                epoch_scale_losses[idx].append(float(scale_losses[i]))
                epoch_scale_accs[idx].append(float(scale_accs[i]))

            if batch_idx % args.log_every == 0:
                parts = []
                for i, idx in enumerate(trainable_indices):
                    h, w = config.scales[idx]
                    parts.append(f"{h}x{w}={scale_accs[i]:.3f}")
                print(f"  Step {batch_idx}: loss={loss_val:.4f} "
                      f"acc=[{' '.join(parts)}]")

                if WANDB_AVAILABLE:
                    log_dict = {"loss": loss_val, "step": global_step}
                    for i, idx in enumerate(trainable_indices):
                        h, w = config.scales[idx]
                        log_dict[f"loss/scale_{h}x{w}"] = float(scale_losses[i])
                        log_dict[f"acc/scale_{h}x{w}"] = float(scale_accs[i])
                    wandb.log(log_dict)

            global_step += 1

        # Epoch summary
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        print(f"--- Epoch {epoch + 1} Summary (avg loss: {avg_loss:.4f}) ---")

        epoch_log = {"epoch": epoch + 1, "epoch/loss": avg_loss}
        for idx in trainable_indices:
            h, w = config.scales[idx]
            if epoch_scale_accs[idx]:
                avg_acc = np.mean(epoch_scale_accs[idx])
                avg_sloss = np.mean(epoch_scale_losses[idx])
                print(f"  Scale {h}x{w}: acc={avg_acc:.4f} loss={avg_sloss:.4f}")
                epoch_log[f"epoch/acc_{h}x{w}"] = avg_acc
                epoch_log[f"epoch/loss_{h}x{w}"] = avg_sloss

        if WANDB_AVAILABLE:
            wandb.log(epoch_log)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, exp_heads, opt_state,
                            epoch + 1, global_step,
                            args.checkpoint_dir, arch_config)

    # Final save
    save_checkpoint(model, exp_heads, opt_state,
                    args.epochs, global_step,
                    args.checkpoint_dir, arch_config)

    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
