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
                        help="Index of starting frame (t0)")
    parser.add_argument("--n_steps", type=int, default=2000,
                        help="Number of autoregressive steps")
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

    # Validate start frame and n_steps
    max_steps = len(indices) - args.start_frame - 1
    if args.n_steps > max_steps:
        print(f"  Clamped n_steps from {args.n_steps} to {max_steps} "
              f"({len(indices)} frames available)")
        args.n_steps = max_steps
    max_start = len(indices) - args.n_steps - 1
    if args.start_frame > max_start:
        args.start_frame = max(0, max_start)
        print(f"  Clamped start_frame to {args.start_frame}")

    # JIT the generation function (temperature captured as closure so the
    # argmax-vs-sample Python branch resolves at trace time).
    temperature = args.temperature

    @jax.jit
    def generate_step(t0_tokens, key):
        return generate_t1_frame(
            model, exp_heads, config, t0_tokens,
            scales_t0, padded_t0, scales_t1, padded_t1,
            attn_bias, scale_masks, trainable_indices,
            key, temperature,
        )

    # --- Rollout ---
    decode_desc = "greedy" if temperature == 0.0 else f"T={temperature}"
    print(f"\nRolling out {args.n_steps} steps from frame {args.start_frame} "
          f"({decode_desc})...")
    t0_tokens = jnp.array(indices[args.start_frame])

    rollout_tokens = [np.array(t0_tokens)]
    gt_tokens_list = [np.array(indices[args.start_frame])]
    all_accuracies = []

    step_keys = jax.random.split(jax.random.PRNGKey(args.seed), args.n_steps)

    log_every = 1 if args.n_steps <= 20 else (10 if args.n_steps <= 200 else 50)
    t_start = time.time()

    for step in range(args.n_steps):
        t1_pred = generate_step(t0_tokens, step_keys[step])
        t1_pred.block_until_ready()

        # Accuracy vs ground truth
        gt_idx = args.start_frame + step + 1
        gt_t1 = jnp.array(indices[gt_idx])
        acc = compute_token_accuracy(t1_pred, gt_t1, config)
        all_accuracies.append(acc)

        if (step + 1) % log_every == 0 or step == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t_start
            sec_per_step = elapsed / (step + 1)
            eta = sec_per_step * (args.n_steps - step - 1)

            scale_parts = []
            for si in trainable_indices:
                h, w = config.scales[si]
                scale_parts.append(f"{h}x{w}={acc[f'scale_{h}x{w}']:.3f}")
            print(f"  Step {step+1}/{args.n_steps}: "
                  f"acc={acc['overall']:.4f} [{' '.join(scale_parts)}] "
                  f"({sec_per_step:.1f}s/step, ETA {eta/60:.1f}min)")

        rollout_tokens.append(np.array(t1_pred))
        gt_tokens_list.append(np.array(indices[gt_idx]))
        t0_tokens = t1_pred

    elapsed_total = time.time() - t_start
    print(f"\nDone: {args.n_steps} steps in {elapsed_total/60:.1f} min "
          f"({elapsed_total/args.n_steps:.1f}s/step)")

    rollout_tokens = np.stack(rollout_tokens)
    gt_tokens_arr = np.stack(gt_tokens_list)

    # --- Save tokens ---
    print("Saving...")
    save_dict = {
        "rollout_indices": rollout_tokens,
        "gt_indices": gt_tokens_arr,
        "scales": np.array(scales_int),
        "start_frame": args.start_frame,
        "n_steps": args.n_steps,
        "codebook": np.array(token_data["codebook"]),
        "effective_vocab_size": token_data["effective_vocab_size"],
        "codebook_dim": token_data["codebook_dim"],
        "new_to_old": token_data["new_to_old"],
        "scale_masks": np.array(token_data["scale_masks"]),
    }

    # Per-scale indices
    for frame_key, frame_arr in [("rollout", rollout_tokens), ("gt", gt_tokens_arr)]:
        for si, s in enumerate(scales_int):
            per_scale = []
            for frame in frame_arr:
                idx_list = unflatten_to_scales(frame, scales_int)
                per_scale.append(np.array(idx_list[si]))
            save_dict[f"{frame_key}_scale_{s}"] = np.stack(per_scale)

    tokens_path = os.path.join(args.output_dir, "rollout_tokens.npz")
    np.savez_compressed(tokens_path, **save_dict)
    print(f"  Tokens: {tokens_path} ({rollout_tokens.shape})")

    # --- Save metrics ---
    metrics = {
        "start_frame": args.start_frame,
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
        print(f"\n  Final accuracy: {final_acc:.4f}")
        print(f"  Average accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
