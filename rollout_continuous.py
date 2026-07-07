"""T3 — closed-loop rollout driver for continuous baselines (BASELINES_SPEC.md).

Free-runs a deterministic next-frame model from ground-truth initial
conditions and saves the predicted pixel fields. Serves:
  - B2 (next_vit): pixel-space ViT regressor from train_next_vit.py —
    x_{t+1} = decoder(encoder(x_t)), all in pixel space. The model
    re-encodes its own decoded output every step.
  - B1 (latent_mse): continuous-latent regressor from train_latent.py —
    closed loop runs ENTIRELY in latent space (z_{t+1} = f(z_t)); the frozen
    decoder is a per-step READOUT for the saved pixel fields and never feeds
    back into the recursion. That asymmetry is the point of the baseline.

Ensemble semantics: continuous baselines are DETERMINISTIC at inference, so
(unlike rollout_nsp.py's fixed-IC sampling ensembles) the only meaningful
ensemble axis is distinct start frames. --n_ics ICs are spread across the
val window (evenly spaced, or every --ic_stride frames), mirroring
rollout_nsp's forecast mode.

Rollouts run in float32 (no bf16): a 2000-step recursion is exactly where
half-precision artifacts could masquerade as model drift.

Output: <output_dir>/rollout_fields.npz
  fields        (n_ics, n_steps+1, H, W) float32 — frame 0 is the GT IC
  start_frames  (n_ics,) int64, in val-window coordinates
plus cfg_meta.json with the full invocation + arch config.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import argparse
import json
import os
import time
import numpy as np
import h5py

from train_next_vit import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Closed-loop rollout for continuous (deterministic) baselines")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Checkpoint dir with training_state.json + *.eqx")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to .h5 file (for GT initial conditions)")
    parser.add_argument("--field", type=str, default="omega")
    parser.add_argument("--sample_start", type=int, default=20000,
                        help="First frame of the IC window (val set start)")
    parser.add_argument("--sample_stop", type=int, default=22000,
                        help="End of the IC window, exclusive (val set end)")
    parser.add_argument("--n_ics", type=int, default=8,
                        help="Number of distinct GT start frames")
    parser.add_argument("--ic_stride", type=int, default=0,
                        help="Spacing between ICs (0 = spread evenly across "
                             "the window)")
    parser.add_argument("--n_steps", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--print_every", type=int, default=100)
    return parser.parse_args()


def load_next_vit(checkpoint_dir, arch_config):
    encoder, decoder = build_model(arch_config, jax.random.PRNGKey(0))
    encoder = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "encoder.eqx"), encoder)
    decoder = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "decoder.eqx"), decoder)
    encoder = eqx.nn.inference_mode(encoder)
    decoder = eqx.nn.inference_mode(decoder)
    return encoder, decoder


@eqx.filter_jit
def next_vit_step(model, xb):
    """(N, 1, H, W) f32 -> (N, 1, H, W) f32."""
    encoder, decoder = model
    return jax.vmap(lambda s: decoder(encoder(s)))(xb)


def load_latent_mse(checkpoint_dir, arch_config):
    from train_latent import load_latent_model
    dynamics, encoder, decoder, _ = load_latent_model(
        checkpoint_dir, arch_config)
    return dynamics, encoder, decoder


@eqx.filter_jit
def latent_init(model, xb):
    """(N, 1, H, W) f32 -> (N, C, hw, hw) f32: encode the GT ICs once."""
    _, encoder, _ = model
    return jax.vmap(encoder)(xb)


@eqx.filter_jit
def latent_step(model, zb):
    """(N, C, hw, hw) f32 -> (N, C, hw, hw) f32, latent-space recursion."""
    dynamics, _, _ = model
    return jax.vmap(dynamics)(zb)


@eqx.filter_jit
def latent_readout(model, zb):
    """(N, C, hw, hw) f32 -> (N, 1, H, W) f32, frozen-decoder readout."""
    _, _, decoder = model
    return jax.vmap(decoder)(zb)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load checkpoint metadata ---
    state_path = os.path.join(args.checkpoint_dir, "training_state.json")
    with open(state_path) as f:
        training_state = json.load(f)
    arch_config = training_state["arch_config"]
    model_type = arch_config.get("model_type", "next_vit")
    print(f"Checkpoint: {args.checkpoint_dir} (model_type={model_type}, "
          f"epoch {training_state['epoch']})")

    if model_type == "next_vit":
        model = load_next_vit(args.checkpoint_dir, arch_config)
        init_fn = lambda m, xb: xb
        step_fn = next_vit_step
        readout_fn = lambda m, state: state
    elif model_type == "latent_mse":
        model = load_latent_mse(args.checkpoint_dir, arch_config)
        init_fn = latent_init
        step_fn = latent_step
        readout_fn = latent_readout
    else:
        raise SystemExit(f"Unknown model_type {model_type!r}")

    # --- GT initial conditions ---
    with h5py.File(args.data_path, "r") as f:
        gt = f[f"fields/{args.field}"][args.sample_start:args.sample_stop]
    M = gt.shape[0]
    print(f"IC window: frames {args.sample_start}-{args.sample_stop} "
          f"({M} frames)")

    if args.ic_stride > 0:
        start_frames = (np.arange(args.n_ics) * args.ic_stride).astype(np.int64)
        start_frames = start_frames[start_frames < M]
    else:
        start_frames = np.unique(
            np.linspace(0, M - 1, args.n_ics).round().astype(np.int64))
    N = len(start_frames)
    if N < args.n_ics:
        print(f"  Clamped n_ics from {args.n_ics} to {N} distinct ICs")
    print(f"Start frames (window coords): {start_frames.tolist()}")

    H, W = gt.shape[1], gt.shape[2]
    fields = np.zeros((N, args.n_steps + 1, H, W), dtype=np.float32)
    fields[:, 0] = gt[start_frames]

    # --- Closed-loop rollout, all ICs advanced together ---
    x0 = jnp.asarray(fields[:, 0][:, None], dtype=jnp.float32)  # (N, 1, H, W)
    state = init_fn(model, x0)
    t_start = time.time()
    for t in range(args.n_steps):
        state = step_fn(model, state)
        pixels = readout_fn(model, state)
        fields[:, t + 1] = np.asarray(pixels[:, 0], dtype=np.float32)
        if (t + 1) % args.print_every == 0:
            elapsed = time.time() - t_start
            rate = (t + 1) / elapsed
            print(f"  step {t + 1}/{args.n_steps} "
                  f"({rate:.1f} steps/s, ETA {(args.n_steps - t - 1) / rate:.0f}s)")

    # --- Save ---
    out_path = os.path.join(args.output_dir, "rollout_fields.npz")
    np.savez(out_path,
             fields=fields,
             start_frames=start_frames,
             n_steps=args.n_steps,
             sample_start=args.sample_start,
             sample_stop=args.sample_stop,
             model_type=model_type,
             checkpoint_dir=args.checkpoint_dir)
    print(f"Saved rollout fields to {out_path} "
          f"({os.path.getsize(out_path) / 1e9:.2f} GB)")

    cfg_meta = {**vars(args), "arch_config": arch_config,
                "start_frames": start_frames.tolist(),
                "checkpoint_epoch": training_state["epoch"]}
    with open(os.path.join(args.output_dir, "cfg_meta.json"), "w") as f:
        json.dump(cfg_meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
