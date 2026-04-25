"""Autoregressive rollout for teacher-forced NSP model.

Given a starting frame from validation tokens, autoregressively predicts
N steps using greedy decoding (default) or temperature sampling, saving
the predicted token sequence. Decoding to fields is done separately
during analysis.

Outputs:
  - rollout_tokens.npz: predicted tokens (flat indices, per-scale indices)
  - rollout_metrics.json: per-step token accuracy vs ground truth

Usage:
    python rollout_nsp.py \
        --checkpoint_dir experiments/ar/medium-sc341-nsp-large \
        --tokens_path experiments/tokens/medium-sc341-val.npz \
        --n_steps 2000 \
        --output_dir experiments/rollouts/medium-sc341-nsp-large
"""

import argparse
import json
import os
import time

import jax
jax.config.update("jax_threefry_partitionable", False)
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from nsp_model import (
    NSPConfig, NSPModel, ExpansionHeads,
    create_nsp_model, generate_t1_frame,
    build_teacher_forced_mask,
)
from tokenizer import load_tokenized_data, unflatten_to_scales


def parse_args():
    parser = argparse.ArgumentParser(description="NSP autoregressive rollout")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory with model.eqx, exp_heads.eqx, training_state.json")
    parser.add_argument("--tokens_path", type=str, required=True,
                        help="Path to tokenized .npz (validation data)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Index of the starting frame (t0). All "
                             "trajectories share this same IC.")
    parser.add_argument("--n_steps", type=int, default=2000,
                        help="Number of autoregressive steps")
    parser.add_argument("--n_trajectories", type=int, default=1,
                        help="Number of trajectories to roll out in parallel "
                             "(default 1). All trajectories share the same "
                             "start frame; only the sampling seed varies "
                             "(seeds = seed, seed+1, ..., seed+N-1). The "
                             "rollout ensemble exposes sampling-noise variance "
                             "at fixed IC -- the right measure for blowup "
                             "(higher per-trajectory EMD vs GT pixel "
                             "distribution = trajectory drifted off-manifold). "
                             "Output rank changes to (N, n_steps+1, "
                             "tokens_per_frame) when N > 1.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0 = greedy argmax")
    return parser.parse_args()


def compute_token_accuracy(pred_tokens, gt_tokens, config):
    """Per-scale and overall token accuracy."""
    boundaries = config.scale_boundaries
    results = {}
    total_correct = 0
    total_tokens = 0

    for scale_idx in config.trainable_scale_indices:
        start = boundaries[scale_idx]
        end = boundaries[scale_idx + 1]
        correct = int(jnp.sum(pred_tokens[start:end] == gt_tokens[start:end]))
        n = end - start
        h, w = config.scales[scale_idx]
        results[f"scale_{h}x{w}"] = correct / n
        total_correct += correct
        total_tokens += n

    results["overall"] = total_correct / total_tokens
    return results


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenized data
    print(f"Loading tokens from {args.tokens_path}...")
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    scales_int = token_data["scales"]
    scale_masks = jnp.array(token_data["scale_masks"])
    print(f"  {len(indices)} frames, {sum(s*s for s in scales_int)} tokens/frame")

    # Load model config from checkpoint
    state_path = os.path.join(args.checkpoint_dir, "training_state.json")
    with open(state_path) as f:
        training_state = json.load(f)
    arch = training_state["arch_config"]
    print(f"  Architecture: {arch['n_layer']}L {arch['n_head']}H {arch['n_embd']}D")

    # Create model
    config = NSPConfig(
        n_layer=arch["n_layer"],
        n_head=arch["n_head"],
        n_embd=arch["n_embd"],
        dropout=0.0,
        rope_theta=arch["rope_theta"],
        n_refine_layers=arch["n_refine_layers"],
    )

    key = jax.random.PRNGKey(args.seed)
    model, exp_heads = create_nsp_model(token_data, config, key)

    # Load weights
    model = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "model.eqx"), model)
    exp_heads = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "exp_heads.eqx"), exp_heads)
    model = eqx.nn.inference_mode(model)
    exp_heads = eqx.nn.inference_mode(exp_heads)
    print(f"  Loaded checkpoint from {args.checkpoint_dir}")

    # Sequence layout
    trainable_indices = config.trainable_scale_indices
    scales_t0 = config.scales
    scales_t1 = config.scales[:-1]
    tokens_t0 = sum(h * w for h, w in scales_t0)
    tokens_t1 = sum(h * w for h, w in scales_t1)
    padded_t0 = ((tokens_t0 + 127) // 128) * 128
    padded_t1 = ((tokens_t1 + 127) // 128) * 128
    print(f"  Sequence: t0={tokens_t0}->{padded_t0}, t1={tokens_t1}->{padded_t1}")

    attn_bias = build_teacher_forced_mask(
        scales_t0, padded_t0, scales_t1, padded_t1)

    # Validate start frame and n_steps. All trajectories share the same
    # starting IC (args.start_frame); only the sampling seed varies. The
    # rollout ensemble exposes sampling-noise variance at fixed IC, which
    # is what we want for "blowup" measurement (EMD vs GT pixel distribution
    # at the matching frames).
    N = max(1, args.n_trajectories)
    max_steps = len(indices) - args.start_frame - 1
    if max_steps < 1:
        raise ValueError(
            f"Val window too short: {len(indices)} frames available, "
            f"start_frame={args.start_frame} leaves {max_steps} GT steps."
        )
    if args.n_steps > max_steps:
        print(f"  Clamped n_steps from {args.n_steps} to {max_steps} "
              f"({len(indices)} val frames, start_frame={args.start_frame})")
        args.n_steps = max_steps
    max_start = len(indices) - args.n_steps - 1
    if args.start_frame > max_start:
        args.start_frame = max(0, max_start)
        print(f"  Clamped start_frame to {args.start_frame}")

    # All trajectories share the same start frame; seeds vary per trajectory.
    start_frames = np.full(N, args.start_frame, dtype=np.int64)
    trajectory_seeds = np.array(
        [args.seed + i for i in range(N)], dtype=np.int64)

    # JIT the generation function (temperature captured as closure so the
    # argmax-vs-sample Python branch resolves at trace time). vmap over the
    # trajectory axis so a step advances all N trajectories in one forward.
    temperature = args.temperature

    def _generate_one(t0_tokens, key):
        return generate_t1_frame(
            model, exp_heads, config, t0_tokens,
            scales_t0, padded_t0, scales_t1, padded_t1,
            attn_bias, scale_masks, trainable_indices,
            key, temperature,
        )

    @jax.jit
    def generate_step_batched(t0_batch, keys_batch):
        # t0_batch: (N, tokens_per_frame);  keys_batch: (N, 2)
        return jax.vmap(_generate_one)(t0_batch, keys_batch)

    # --- Rollout ---
    decode_desc = "greedy" if temperature == 0.0 else f"T={temperature}"
    print(f"\nRolling out {args.n_steps} steps, {N} trajector"
          f"{'y' if N == 1 else 'ies'} "
          f"(start_frames={start_frames.tolist()}, seeds="
          f"{trajectory_seeds.tolist()}, {decode_desc})...")

    # Initial (N, tokens_per_frame) batch and matching GT.
    t0_batch = jnp.array(indices[start_frames])
    rollout_tokens = [np.array(t0_batch)]   # list of (N, tokens_per_frame)
    gt_tokens_list = [np.array(indices[start_frames])]
    all_accuracies = []   # each entry: mean-over-trajectories accuracy dict

    # Per-trajectory step-key chains: (N, n_steps, 2).
    # Build the N root keys in plain Python (seeds are concrete ints) — doing
    # this inside jax.vmap would try to int() a BatchTracer and crash with a
    # ConcretizationTypeError.
    traj_root_keys = jnp.stack(
        [jax.random.PRNGKey(int(s)) for s in trajectory_seeds])
    step_keys = jax.vmap(lambda k: jax.random.split(k, args.n_steps))(traj_root_keys)
    # shape (N, n_steps, 2)

    log_every = 1 if args.n_steps <= 20 else (10 if args.n_steps <= 200 else 50)
    t_start = time.time()

    for step in range(args.n_steps):
        keys_step = step_keys[:, step, :]     # (N, 2)
        t1_batch = generate_step_batched(t0_batch, keys_step)   # (N, tokens_per_frame)
        t1_batch.block_until_ready()

        # Accuracy per trajectory vs GT at (start_frame + step + 1).
        gt_indices_step = start_frames + step + 1
        gt_batch = jnp.array(indices[gt_indices_step])           # (N, tokens_per_frame)
        per_traj_acc = [
            compute_token_accuracy(t1_batch[i], gt_batch[i], config)
            for i in range(N)
        ]
        mean_acc = {
            k: float(np.mean([a[k] for a in per_traj_acc]))
            for k in per_traj_acc[0].keys()
        }
        all_accuracies.append(mean_acc)

        if (step + 1) % log_every == 0 or step == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t_start
            sec_per_step = elapsed / (step + 1)
            eta = sec_per_step * (args.n_steps - step - 1)
            scale_parts = []
            for si in trainable_indices:
                h, w = config.scales[si]
                scale_parts.append(f"{h}x{w}={mean_acc[f'scale_{h}x{w}']:.3f}")
            tag = "" if N == 1 else f" [N={N} mean]"
            print(f"  Step {step+1}/{args.n_steps}{tag}: "
                  f"acc={mean_acc['overall']:.4f} [{' '.join(scale_parts)}] "
                  f"({sec_per_step:.2f}s/step, ETA {eta/60:.1f}min)")

        rollout_tokens.append(np.array(t1_batch))
        gt_tokens_list.append(np.array(gt_batch))
        t0_batch = t1_batch

    elapsed_total = time.time() - t_start
    print(f"\nDone: {args.n_steps} steps x {N} trajectories "
          f"in {elapsed_total/60:.1f} min "
          f"({elapsed_total/args.n_steps:.2f}s/step)")

    # Stack -> (N, n_steps+1, tokens_per_frame). If N==1, squeeze for
    # backward-compat with existing analyze_rollout.py (rank-3).
    rollout_tokens = np.stack(rollout_tokens, axis=1)   # (N, T+1, tokens)
    gt_tokens_arr  = np.stack(gt_tokens_list, axis=1)
    if N == 1:
        rollout_tokens = rollout_tokens[0]   # (T+1, tokens)
        gt_tokens_arr  = gt_tokens_arr[0]

    # --- Save tokens ---
    print("Saving...")
    save_dict = {
        "rollout_indices": rollout_tokens,
        "gt_indices": gt_tokens_arr,
        "scales": np.array(scales_int),
        "start_frame": int(start_frames[0]),          # scalar for back-compat
        "start_frames": start_frames.astype(np.int64), # (N,) — source of truth when N>1
        "trajectory_seeds": trajectory_seeds.astype(np.int64),
        "n_trajectories": int(N),
        "n_steps": args.n_steps,
        "codebook": np.array(token_data["codebook"]),
        "effective_vocab_size": token_data["effective_vocab_size"],
        "codebook_dim": token_data["codebook_dim"],
        "new_to_old": token_data["new_to_old"],
        "scale_masks": np.array(token_data["scale_masks"]),
    }

    # Per-scale indices: only saved for the N=1 (backward-compat) case.
    # At N>1 these fields would triple the npz size without being consumed by
    # analyze_rollout.py (which unflattens from the flat indices internally).
    if N == 1:
        for frame_key, frame_arr in [("rollout", rollout_tokens), ("gt", gt_tokens_arr)]:
            for si, s in enumerate(scales_int):
                per_scale = []
                for frame in frame_arr:
                    idx_list = unflatten_to_scales(frame, scales_int)
                    per_scale.append(np.array(idx_list[si]))
                save_dict[f"{frame_key}_scale_{s}"] = np.stack(per_scale)

    tokens_path = os.path.join(args.output_dir, "rollout_tokens.npz")
    np.savez_compressed(tokens_path, **save_dict)
    print(f"  Tokens: {tokens_path} (shape {rollout_tokens.shape}, N={N})")

    # --- Save metrics ---
    metrics = {
        "start_frame": int(start_frames[0]),
        "start_frames": start_frames.tolist(),
        "trajectory_seeds": trajectory_seeds.tolist(),
        "n_trajectories": int(N),
        "n_steps": args.n_steps,
        "elapsed_seconds": elapsed_total,
        "per_step": all_accuracies,
    }
    metrics_path = os.path.join(args.output_dir, "rollout_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Summary
    if all_accuracies:
        final_acc = all_accuracies[-1]["overall"]
        avg_acc = np.mean([a["overall"] for a in all_accuracies])
        tag = "" if N == 1 else " (mean over trajectories)"
        print(f"\n  Final accuracy{tag}:   {final_acc:.4f}")
        print(f"  Average accuracy{tag}: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
