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
from samplers import SamplerConfig
from tokenizer import load_tokenized_data, unflatten_to_scales


def atomic_savez_compressed(path, **arrays):
    """np.savez_compressed to a temp file, then atomically rename into place.

    A killed job (e.g. scancel / walltime SIGKILL mid-save) leaves either the
    complete .npz or nothing at `path` — never a truncated file. This matters
    because the sweep's rollout-reuse skip checks for rollout_tokens.npz and a
    partial file would be fed to np.load downstream (BadZipFile). Writing to a
    file object (not a str) stops np.savez from appending .npz to the temp name.
    """
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)   # atomic on the same filesystem


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
    parser.add_argument("--n_ics", type=int, default=0,
                        help="Forecast mode: when > 0, roll out N=n_ics "
                             "trajectories each starting from a DISTINCT "
                             "ground-truth frame spread across the val set "
                             "(short-horizon forecast skill), instead of N "
                             "trajectories sharing one IC. Use with a short "
                             "--n_steps (e.g. 10). All ICs share one seed "
                             "(IC spread is the variable, not sampling "
                             "noise). Mutually exclusive with "
                             "--n_trajectories > 1.")
    parser.add_argument("--ic_stride", type=int, default=0,
                        help="Forecast mode IC spacing. 0 (default) spreads "
                             "the n_ics ICs evenly across the usable val "
                             "window; > 0 places them at frames "
                             "0, ic_stride, 2*ic_stride, ... (clamped to the "
                             "window).")
    parser.add_argument("--ic_chunk", type=int, default=0,
                        help="Forecast mode memory guard. 0 (default) runs "
                             "the whole IC batch through one vmap'd forward "
                             "per step; > 0 splits the IC axis into "
                             "sub-batches of this size (bounds peak "
                             "activation memory to ic_chunk ICs). Set ~32 "
                             "for large configs (sc1941) at n_ics=128.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature; 0 = greedy argmax")
    parser.add_argument("--top_k", type=int, default=0,
                        help="Top-k logit truncation per token (0 disables; "
                             "ignored when temperature == 0).")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus (top-p) threshold in (0, 1]; 1.0 "
                             "disables (ignored when temperature == 0).")
    parser.add_argument("--log_topk", type=int, default=0,
                        help="If > 0, also save per-emission top-K raw "
                             "logits + token indices (post scale-mask, "
                             "pre temperature/top_k/top_p) to "
                             "rollout_logits.npz for offline diagnostics. "
                             "Storage scales as N * n_steps * "
                             "tokens_per_frame * log_topk * ~4 bytes "
                             "(fp16 logits + int16 indices).")
    parser.add_argument("--train_tokens_path", type=str, default=None,
                        help="Path to training tokens .npz. When given, "
                             "build a per-position vocabulary mask from "
                             "training-set co-occurrences and AND it with "
                             "the per-scale scale_masks at every emission. "
                             "Restricts each position to tokens ever seen "
                             "at that exact (scale, row, col) during "
                             "training. Must come from the same VQ-VAE "
                             "(same effective_vocab_size and new_to_old "
                             "mapping) as --tokens_path. None disables "
                             "(default; only per-scale masking applies).")
    # --- Inference samplers (samplers.py). Default 'ancestral' == the legacy
    #     temperature/top_k/top_p path, reproduced bit-for-bit. ---
    parser.add_argument("--sampler", type=str, default="ancestral",
                        choices=["ancestral", "entropy_target", "top_h",
                                 "typical", "min_p", "inverted_edt", "data_mix"],
                        help="Inference sampler. 'ancestral' (default) uses "
                             "--temperature/--top_k/--top_p; the others ignore "
                             "those. entropy_target = per-scale-anchored "
                             "entropy homeostat; top_h = Top-H bounded-entropy "
                             "truncation; typical = locally typical; "
                             "min_p / inverted_edt = negative controls; "
                             "data_mix = convex mixture toward the per-scale "
                             "data prior, confidence-deficit gated (needs "
                             "--data_prior_path + a model-conditional "
                             "--entropy_anchor_path).")
    parser.add_argument("--base_temperature", type=float, default=1.0,
                        help="top_h only: pre-warm factor before the entropy "
                             "cap (>1 warms -> warm->cap variant).")
    parser.add_argument("--top_h_alpha", type=float, default=0.4,
                        help="top_h relative mode: keep set with renormalized "
                             "entropy <= alpha * H(p).")
    parser.add_argument("--top_h_mode", type=str, default="relative",
                        choices=["relative", "absolute"],
                        help="top_h bound: 'relative' (alpha*H(p)) or "
                             "'absolute' (per-scale anchor; needs "
                             "--entropy_anchor_path).")
    parser.add_argument("--typical_tau", type=float, default=0.95,
                        help="typical: cumulative typical-mass threshold.")
    parser.add_argument("--min_p", type=float, default=0.05,
                        help="min_p: keep tokens with p >= min_p * max_p.")
    parser.add_argument("--inverted_edt_strength", type=float, default=0.5,
                        help="inverted_edt: entropy-shrink strength in [0,1).")
    parser.add_argument("--entropy_anchor_path", type=str, default=None,
                        help="Per-scale entropy anchor JSON from "
                             "measure_tokenizer_entropy.py. Required for "
                             "--sampler entropy_target and for --sampler top_h "
                             "--top_h_mode absolute. Must come from the same "
                             "VQ-VAE/sc-config as the checkpoint.")
    parser.add_argument("--anchor_stat", type=str, default="auto",
                        choices=["auto", "pooled", "per_position"],
                        help="Which T2 statistic anchors the sampler. 'auto' = "
                             "pooled-marginal for entropy_target, "
                             "mean-per-position for top_h absolute.")
    parser.add_argument("--data_prior_path", type=str, default=None,
                        help="Per-scale data-prior .npz (data_prior table) from "
                             "measure_tokenizer_entropy.py --data_prior_out. "
                             "Required for --sampler data_mix. Must share the "
                             "VQ-VAE/effective_vocab of --tokens_path.")
    parser.add_argument("--mix_gain", type=float, default=1.0,
                        help="data_mix: gain on the confidence-deficit gate "
                             "(single global knob; 1.0 = parameter-free).")
    parser.add_argument("--mix_lam_max", type=float, default=1.0,
                        help="data_mix: cap on the injected data-prior fraction.")
    return parser.parse_args()


def load_anchor_array(path, trainable_indices, stat):
    """Load a per-scale entropy anchor JSON -> (n_trainable,) jnp float32 array.

    The returned array aligns 1:1 with enumerate(trainable_indices) (the head
    index i in generate_t1_frame). Asserts the anchor's trainable scales match
    the model's, mirroring the VQ-mismatch guards above.
    """
    with open(path) as f:
        anchor = json.load(f)
    file_tr = [int(x) for x in anchor.get("trainable_scale_indices", [])]
    cfg_tr = [int(x) for x in trainable_indices]
    if file_tr != cfg_tr:
        raise SystemExit(
            f"Anchor/model trainable-scale mismatch: anchor {file_tr} vs "
            f"model {cfg_tr} ({path}). Use an anchor from the same "
            f"VQ-VAE/sc-config as the checkpoint.")
    key = ("per_trainable_pooled_marginal_nats" if stat == "pooled"
           else "per_trainable_mean_per_position_nats")
    vals = anchor[key]
    if len(vals) != len(cfg_tr):
        raise SystemExit(
            f"Anchor length {len(vals)} != n_trainable {len(cfg_tr)} ({path}).")
    return jnp.asarray(vals, dtype=jnp.float32)


def load_data_prior(path, token_data):
    """Load the per-scale data-prior table -> (tokens_per_frame, V) jnp float32.

    Threaded into generate_t1_frame exactly like position_mask. Guards that it
    shares the val tokens' effective vocab and frame layout (same VQ-VAE).
    """
    d = np.load(path, allow_pickle=True)
    table = np.asarray(d["data_prior"])
    V_prior = int(d["effective_vocab_size"])
    V_val = int(token_data["effective_vocab_size"])
    P = int(sum(s * s for s in token_data["scales"]))
    if V_prior != V_val:
        raise SystemExit(
            f"data_prior V={V_prior} != val tokens V={V_val} ({path}); build "
            f"the prior from train tokens of the same VQ-VAE as --tokens_path.")
    if table.shape != (P, V_val):
        raise SystemExit(
            f"data_prior shape {table.shape} != expected ({P}, {V_val}) ({path}).")
    return jnp.asarray(table, dtype=jnp.float32)


def compute_token_accuracy(pred_tokens, gt_tokens, config):
    """Per-scale and overall token accuracy for a single (pred, gt) frame.

    Kept for API compatibility / single-frame use. The rollout hot loop uses
    the vectorized numpy path below instead to avoid per-step device->host
    syncs.
    """
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


def compute_per_step_accuracy_np(pred_steps, gt_steps, config):
    """Vectorized per-step, mean-over-trajectories token accuracy.

    Single numpy pass over the already-collected rollout tokens — no
    device->host syncs in the hot loop. Reproduces the exact values the old
    per-step `compute_token_accuracy` Python loop produced (per-trajectory
    accuracy fractions averaged over trajectories), as a list of per-step
    dicts suitable for rollout_metrics.json["per_step"].

    Args:
        pred_steps: (n_steps, N, tokens_per_frame) predicted t1 tokens
        gt_steps:   (n_steps, N, tokens_per_frame) ground-truth t1 tokens
        config: NSPConfig

    Returns:
        list of length n_steps; each entry a dict with per-scale keys
        ("scale_HxW") and "overall", all floats (mean over trajectories).
    """
    boundaries = config.scale_boundaries
    eq = (np.asarray(pred_steps) == np.asarray(gt_steps))   # (n_steps, N, P)

    per_step = []
    n_steps = eq.shape[0]
    for step in range(n_steps):
        eq_step = eq[step]                                  # (N, P)
        results = {}
        total_correct = np.zeros(eq_step.shape[0], dtype=np.float64)
        total_tokens = 0
        for scale_idx in config.trainable_scale_indices:
            start = boundaries[scale_idx]
            end = boundaries[scale_idx + 1]
            n = end - start
            # per-trajectory correct fraction, then mean over trajectories,
            # matching the old float(np.mean([per-traj acc])) reduction.
            correct = eq_step[:, start:end].sum(axis=1)     # (N,)
            h, w = config.scales[scale_idx]
            results[f"scale_{h}x{w}"] = float(np.mean(correct / n))
            total_correct += correct
            total_tokens += n
        results["overall"] = float(np.mean(total_correct / total_tokens))
        per_step.append(results)
    return per_step


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

    # Optional: build per-position vocabulary mask from training tokens.
    # The mask must come from the same VQ-VAE as the val tokens above.
    position_mask_np = None
    position_mask = None
    if args.train_tokens_path is not None:
        print(f"Loading training tokens for per-position mask: "
              f"{args.train_tokens_path}")
        train_npz = np.load(args.train_tokens_path)
        train_idx = train_npz["indices_flat"]
        V_train = int(train_npz["effective_vocab_size"])
        V_val = int(token_data["effective_vocab_size"])
        if V_train != V_val:
            raise SystemExit(
                f"VQ-VAE mismatch: train tokens have V={V_train} but val "
                f"tokens have V={V_val}; use train tokens from the same "
                f"VQ-VAE as --tokens_path.")
        if not np.array_equal(train_npz["new_to_old"],
                              token_data["new_to_old"]):
            raise SystemExit(
                "VQ-VAE compact-vocab mapping (new_to_old) differs "
                "between train and val tokens; can't build a position "
                "mask in a consistent vocab space.")
        F_t, P_t = train_idx.shape
        if P_t != sum(s * s for s in scales_int):
            raise SystemExit(
                f"Train tokens have P={P_t} positions but val expects "
                f"{sum(s * s for s in scales_int)}.")
        # (P, V) bool: M[p, v] = 1 iff token v appears at position p in train.
        M = np.zeros((P_t, V_train), dtype=bool)
        flat_p = np.broadcast_to(np.arange(P_t), (F_t, P_t)).ravel()
        flat_v = train_idx.ravel().astype(np.int64)
        M[flat_p, flat_v] = True
        per_pos_count = M.sum(axis=1)
        print(f"  Built position mask: {F_t} train frames, "
              f"per-pos vocab min/med/max = "
              f"{per_pos_count.min()}/{int(np.median(per_pos_count))}/"
              f"{per_pos_count.max()}")
        # Sanity: zero positions allowed -> sampling will fail. Should not
        # happen for any position observed in training.
        if per_pos_count.min() == 0:
            raise SystemExit(
                f"Position mask has {(per_pos_count == 0).sum()} positions "
                f"with zero allowed tokens; can't sample.")
        position_mask_np = M
        position_mask = jnp.array(M)

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

    # --- Sampler config + optional per-scale entropy anchor ---
    # sampler_cfg is captured as a static closure (frozen dataclass -> hashable)
    # so its branches resolve at trace time; anchor_array is a traced (n_trainable,)
    # array broadcast (not vmapped) under the trajectory vmap.
    needs_anchor = (args.sampler in ("entropy_target", "data_mix")
                    or (args.sampler == "top_h" and args.top_h_mode == "absolute"))
    if args.anchor_stat == "auto":
        # data_mix / top_h use the conditional (per-position) anchor; the
        # entropy_target homeostat defaults to the pooled-marginal warm target.
        anchor_stat = ("per_position"
                       if args.sampler in ("top_h", "data_mix") else "pooled")
    else:
        anchor_stat = args.anchor_stat
    anchor_array = None
    if args.entropy_anchor_path is not None:
        anchor_array = load_anchor_array(
            args.entropy_anchor_path, trainable_indices, anchor_stat)
    if needs_anchor and anchor_array is None:
        raise SystemExit(
            f"--sampler {args.sampler}"
            + (" --top_h_mode absolute" if args.sampler == "top_h" else "")
            + " requires --entropy_anchor_path (per-scale entropy anchor JSON; "
            "for data_mix use the model-conditional anchor from "
            "measure_model_entropy.py).")

    # data_mix also needs the per-scale data-prior table (threaded like position_mask).
    data_prior_table = None
    if args.data_prior_path is not None:
        data_prior_table = load_data_prior(args.data_prior_path, token_data)
    if args.sampler == "data_mix" and data_prior_table is None:
        raise SystemExit(
            "--sampler data_mix requires --data_prior_path (per-scale data "
            "prior .npz from measure_tokenizer_entropy.py --data_prior_out).")

    sampler_cfg = SamplerConfig(
        method=args.sampler,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        base_temperature=args.base_temperature,
        top_h_alpha=args.top_h_alpha, top_h_mode=args.top_h_mode,
        typical_tau=args.typical_tau, min_p=args.min_p,
        inverted_edt_strength=args.inverted_edt_strength,
        mix_gain=args.mix_gain, mix_lam_max=args.mix_lam_max)

    # Two launch modes:
    #  (default) trajectory ensemble: N trajectories share one start frame
    #    (args.start_frame); only the sampling seed varies. Exposes
    #    sampling-noise variance at fixed IC (long-rollout blowup measure).
    #  (forecast) IC ensemble (--n_ics > 0): N trajectories each start from a
    #    DISTINCT ground-truth frame spread across the val set; all share one
    #    seed. With a short --n_steps this measures free-running forecast skill
    #    at lead times k, averaged over many ICs.
    forecast_mode = args.n_ics > 0
    if forecast_mode:
        if args.n_trajectories > 1:
            raise ValueError(
                "--n_ics and --n_trajectories > 1 are mutually exclusive "
                "(IC ensemble vs sampling ensemble). Pick one.")
        # The last usable IC must leave room for +n_steps of GT lookahead.
        max_start = len(indices) - args.n_steps - 1
        if max_start < 0:
            raise ValueError(
                f"Val window too short: {len(indices)} frames available, "
                f"n_steps={args.n_steps} leaves no room for an IC.")
        if args.ic_stride > 0:
            start_frames = (np.arange(args.n_ics) * args.ic_stride).astype(np.int64)
            start_frames = start_frames[start_frames <= max_start]
        else:
            # Evenly spaced across [0, max_start], distinct (de-duped).
            start_frames = np.unique(
                np.linspace(0, max_start, args.n_ics).round().astype(np.int64))
        N = len(start_frames)
        if N < 1:
            raise ValueError(
                f"No valid ICs: n_ics={args.n_ics}, ic_stride={args.ic_stride}, "
                f"max_start={max_start}.")
        if N < args.n_ics:
            print(f"  Clamped n_ics from {args.n_ics} to {N} distinct ICs "
                  f"(max_start={max_start})")
        # All ICs share one seed; IC spread is the variable of interest.
        trajectory_seeds = np.full(N, args.seed, dtype=np.int64)
    else:
        # Validate start frame and n_steps; all trajectories share the IC.
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
        start_frames = np.full(N, args.start_frame, dtype=np.int64)
        trajectory_seeds = np.array(
            [args.seed + i for i in range(N)], dtype=np.int64)

    # JIT the generation function (temperature/top_k/top_p/log_topk
    # captured as closures so the static Python branches resolve at
    # trace time). vmap over the trajectory axis so a step advances all
    # N trajectories in one forward.
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    log_topk = args.log_topk

    def _generate_one(t0_tokens, key):
        return generate_t1_frame(
            model, exp_heads, config, t0_tokens,
            scales_t0, padded_t0, scales_t1, padded_t1,
            attn_bias, scale_masks, trainable_indices,
            key, temperature, top_k, top_p, log_topk,
            position_mask=position_mask,
            sampler_cfg=sampler_cfg,
            anchor_array=anchor_array,
            data_prior_table=data_prior_table,
        )

    @jax.jit
    def generate_step_batched(t0_batch, keys_batch):
        # t0_batch: (N, tokens_per_frame);  keys_batch: (N, 2)
        # Returns (N, tokens_per_frame) when log_topk == 0,
        # else 3-tuple of (predicted, top_logits, top_indices).
        return jax.vmap(_generate_one)(t0_batch, keys_batch)

    ic_chunk = args.ic_chunk

    def generate_step_chunked(t0_batch, keys_batch):
        # Split the IC/trajectory axis into sub-batches of ic_chunk to bound
        # peak activation memory at large N (forecast mode, n_ics up to ~256
        # on sc1941). generate_step_batched is jitted once per chunk shape and
        # reused across sub-batches, so peak memory ~ ic_chunk, not N.
        if ic_chunk <= 0 or t0_batch.shape[0] <= ic_chunk:
            return generate_step_batched(t0_batch, keys_batch)
        outs = [
            generate_step_batched(t0_batch[s:s + ic_chunk],
                                  keys_batch[s:s + ic_chunk])
            for s in range(0, t0_batch.shape[0], ic_chunk)
        ]
        if log_topk > 0:
            return tuple(
                jnp.concatenate([o[j] for o in outs], axis=0)
                for j in range(3))
        return jnp.concatenate(outs, axis=0)

    # --- Rollout ---
    if args.sampler != "ancestral":
        parts = [args.sampler]
        if args.sampler == "entropy_target":
            parts.append(f"anchor={anchor_stat}")
        elif args.sampler == "top_h":
            parts.append(f"mode={args.top_h_mode}")
            parts.append(f"alpha={args.top_h_alpha}" if args.top_h_mode == "relative"
                         else f"anchor={anchor_stat}")
            if args.base_temperature != 1.0:
                parts.append(f"warm={args.base_temperature}")
        elif args.sampler == "typical":
            parts.append(f"tau={args.typical_tau}")
        elif args.sampler == "min_p":
            parts.append(f"min_p={args.min_p}")
        elif args.sampler == "inverted_edt":
            parts.append(f"strength={args.inverted_edt_strength}")
        elif args.sampler == "data_mix":
            parts.append(f"gain={args.mix_gain}")
            parts.append(f"lam_max={args.mix_lam_max}")
        decode_desc = ",".join(parts)
    elif temperature == 0.0:
        decode_desc = "greedy"
    else:
        parts = [f"T={temperature}"]
        if top_k > 0:
            parts.append(f"top_k={top_k}")
        if top_p < 1.0:
            parts.append(f"top_p={top_p}")
        decode_desc = ",".join(parts)
    if position_mask is not None:
        decode_desc += ",pos_mask"
    if forecast_mode:
        print(f"\nForecast rollout: {args.n_steps} steps x {N} ICs "
              f"(spread {int(start_frames[0])}..{int(start_frames[-1])}, "
              f"chunk={ic_chunk or N}, seed={args.seed}, {decode_desc})...")
    else:
        print(f"\nRolling out {args.n_steps} steps, {N} trajector"
              f"{'y' if N == 1 else 'ies'} "
              f"(start_frames={start_frames.tolist()}, seeds="
              f"{trajectory_seeds.tolist()}, {decode_desc})...")

    # Initial (N, tokens_per_frame) batch and matching GT.
    t0_batch = jnp.array(indices[start_frames])
    rollout_tokens = [np.array(t0_batch)]   # list of (N, tokens_per_frame)
    gt_tokens_list = [np.array(indices[start_frames])]
    all_accuracies = []   # each entry: mean-over-trajectories accuracy dict

    # Per-step top-K logits + indices, if logging is enabled. Each entry is
    # (N, tokens_per_frame, log_topk). The IC frame (step 0) has no logits
    # (it was loaded from data, not predicted), so the saved arrays cover
    # only the n_steps predicted frames.
    logit_logits_per_step = [] if log_topk > 0 else None
    logit_indices_per_step = [] if log_topk > 0 else None

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
        out = generate_step_chunked(t0_batch, keys_step)
        if log_topk > 0:
            t1_batch, top_logits_step, top_indices_step = out
            # Cast on-device to fp16 / int16 before host transfer to halve
            # bandwidth + storage. effective_vocab_size <= 4096 fits int16.
            top_logits_step = top_logits_step.astype(jnp.float16)
            top_indices_step = top_indices_step.astype(jnp.int16)
            logit_logits_per_step.append(np.array(top_logits_step))
            logit_indices_per_step.append(np.array(top_indices_step))
        else:
            t1_batch = out

        # Host copy of the predicted frame. np.array() forces the single
        # device->host transfer for this step (also acts as the sync point);
        # no separate block_until_ready needed. Per-step accuracy is NOT
        # computed here — it is recovered in one vectorized numpy pass after
        # the loop (compute_per_step_accuracy_np) to avoid the per-step
        # device->host sync storm. GT comes straight off the host-resident
        # `indices` array (no device round-trip).
        t1_host = np.array(t1_batch)
        gt_indices_step = start_frames + step + 1
        gt_host = np.asarray(indices[gt_indices_step])           # (N, tokens_per_frame)

        if (step + 1) % log_every == 0 or step == 0 or step == args.n_steps - 1:
            elapsed = time.time() - t_start
            sec_per_step = elapsed / (step + 1)
            eta = sec_per_step * (args.n_steps - step - 1)
            tag = "" if N == 1 else f" [N={N}]"
            print(f"  Step {step+1}/{args.n_steps}{tag}: "
                  f"({sec_per_step:.2f}s/step, ETA {eta/60:.1f}min)")

        rollout_tokens.append(t1_host)
        gt_tokens_list.append(gt_host)
        t0_batch = t1_batch

    elapsed_total = time.time() - t_start
    print(f"\nDone: {args.n_steps} steps x {N} trajectories "
          f"in {elapsed_total/60:.1f} min "
          f"({elapsed_total/args.n_steps:.2f}s/step)")

    # Per-step token accuracy, computed ONCE in a single vectorized numpy pass
    # over the host-resident rollout/GT tokens (no per-step device->host
    # syncs). Predictions are entries 1..n_steps; entry 0 is the IC frame.
    # Reproduces the exact per_step values the old in-loop Python sync loop
    # produced (per-trajectory accuracy averaged over trajectories).
    if args.n_steps > 0:
        pred_steps = np.stack(rollout_tokens[1:], axis=0)   # (n_steps, N, P)
        gt_steps = np.stack(gt_tokens_list[1:], axis=0)     # (n_steps, N, P)
        all_accuracies = compute_per_step_accuracy_np(
            pred_steps, gt_steps, config)

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
        "position_mask_used": bool(position_mask_np is not None),
    }
    if position_mask_np is not None:
        save_dict["train_tokens_path"] = str(args.train_tokens_path)

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
    atomic_savez_compressed(tokens_path, **save_dict)
    print(f"  Tokens: {tokens_path} (shape {rollout_tokens.shape}, N={N})")

    # Decode-parameter metadata for downstream diagnostics (cfg discovery,
    # temperature-keyed plot colors) — see diagnostics_common.load_cfg_meta.
    cfg_meta = {
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seed": args.seed,
        "n_trajectories": int(N),
        "n_steps": args.n_steps,
        "start_frame": int(start_frames[0]),
        "log_topk": log_topk,
        "position_mask_used": bool(position_mask_np is not None),
        "sampler": args.sampler,
        "base_temperature": args.base_temperature,
        "top_h_alpha": args.top_h_alpha,
        "top_h_mode": args.top_h_mode,
        "typical_tau": args.typical_tau,
        "min_p": args.min_p,
        "inverted_edt_strength": args.inverted_edt_strength,
        "mix_gain": args.mix_gain,
        "mix_lam_max": args.mix_lam_max,
        "data_prior_path": args.data_prior_path,
        "entropy_anchor_path": args.entropy_anchor_path,
        "anchor_stat": anchor_stat,
        "anchor_array": (anchor_array.tolist() if anchor_array is not None else None),
        "checkpoint_dir": args.checkpoint_dir,
        "tokens_path": args.tokens_path,
        "forecast_mode": bool(forecast_mode),
        "n_ics": int(N) if forecast_mode else 0,
        "ic_stride": int(args.ic_stride),
        "start_frames": start_frames.tolist(),
    }
    cfg_meta_path = os.path.join(args.output_dir, "cfg_meta.json")
    with open(cfg_meta_path, "w") as f:
        json.dump(cfg_meta, f, indent=2)
    print(f"  Meta: {cfg_meta_path}")

    # --- Save per-emission top-K logits for offline diagnostics ---
    if log_topk > 0:
        # Stack along a step axis -> (N, n_steps, tokens_per_frame, log_topk)
        top_logits_arr = np.stack(logit_logits_per_step, axis=1)
        top_indices_arr = np.stack(logit_indices_per_step, axis=1)
        if N == 1:
            top_logits_arr = top_logits_arr[0]   # (n_steps, tok, K)
            top_indices_arr = top_indices_arr[0]
        logits_path = os.path.join(args.output_dir, "rollout_logits.npz")
        atomic_savez_compressed(
            logits_path,
            top_logits=top_logits_arr,           # fp16
            top_indices=top_indices_arr,         # int16
            log_topk=np.int32(log_topk),
            scales=np.array(scales_int),
            first_trainable_scale=np.int32(
                token_data.get("first_trainable_scale", 0)),
            n_trajectories=int(N),
            n_steps=args.n_steps,
            start_frame=int(start_frames[0]),
            trajectory_seeds=trajectory_seeds.astype(np.int64),
            effective_vocab_size=token_data["effective_vocab_size"],
        )
        size_mb = os.path.getsize(logits_path) / (1024 ** 2)
        print(f"  Logits: {logits_path} (shape {top_logits_arr.shape}, "
              f"{size_mb:.1f} MB)")

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
