"""measure_model_entropy.py — model teacher-forced per-scale entropy anchor.

The MODEL variant of T2. Where `measure_tokenizer_entropy.py` reads the
tokenizer's *marginal* per-scale code entropy (entropy of the code distribution
over the dataset), this reads the NSP model's *conditional* per-scale predictive
entropy on real (t0, t1) transitions: mean over positions of H(softmax(logits))
when teacher-forced. That conditional entropy is far sharper than the marginal
(the model knows a lot about each cell from t0 + coarser scales), and is the
principled anchor for the entropy-target homeostat:

  * On a cold-optimal config (sc341, best-T≈1.0) the model is already calibrated
    at T=1, so its teacher-forced entropy ≈ what T=1 produces → the homeostat
    targets ≈ T=1 and does NOT over-warm into over-diffusion (the failure the
    marginal anchor causes there).
  * On warm-optimal configs (sc917/sc1941) free-running drifts BELOW the
    teacher-forced entropy (the collapse), so targeting it warms them back up.

Output JSON matches measure_tokenizer_entropy.py's schema, so the rollout's
`--entropy_anchor_path` loader needs zero changes. Both per-trainable slots
hold the same conditional entropy (the model has no "pooled vs per-position"
distinction), so `--anchor_stat per_position` (or pooled) both resolve to it.

Reuses eval_single_step.make_predict_and_loss — the exact validated teacher-
forced forward — so there is no new forward logic to get wrong.

Usage:
    python measure_model_entropy.py \
        --checkpoint_dir experiments/ar/small-sc341-nsp-s24 \
        --tokens_path experiments/tokens/small-sc341-val.npz \
        --output inference_anchors/small-sc341-model.json \
        --max_pairs 256
"""

import argparse
import json
import os

import jax
jax.config.update("jax_threefry_partitionable", False)
import jax.numpy as jnp
import equinox as eqx
import numpy as np

from nsp_model import (
    NSPConfig, create_nsp_model, build_teacher_forced_mask,
)
from eval_single_step import make_predict_and_loss
from tokenizer import load_tokenized_data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint_dir", required=True, help="NSP checkpoint dir")
    ap.add_argument("--tokens_path", required=True,
                    help="Tokenized .npz (val tokens — the transitions we forecast)")
    ap.add_argument("--output", required=True, help="Output anchor .json")
    ap.add_argument("--max_pairs", type=int, default=256,
                    help="Number of consecutive (t0,t1) pairs to average over "
                         "(the mean entropy stabilizes quickly; 256 is plenty).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # --- Load tokens ---
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    scales_int = [int(s) for s in token_data["scales"]]
    scale_masks = jnp.array(token_data["scale_masks"])
    first_trainable = int(token_data.get("first_trainable_scale", 0))

    # --- Load NSP model ---
    state_path = os.path.join(args.checkpoint_dir, "training_state.json")
    with open(state_path) as f:
        arch = json.load(f)["arch_config"]
    config = NSPConfig(
        n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
        dropout=0.0, rope_theta=arch["rope_theta"],
        n_refine_layers=arch["n_refine_layers"],
    )
    key = jax.random.PRNGKey(args.seed)
    model, exp_heads = create_nsp_model(token_data, config, key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "model.eqx"), model)
    exp_heads = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "exp_heads.eqx"), exp_heads)
    model = eqx.nn.inference_mode(model)
    exp_heads = eqx.nn.inference_mode(exp_heads)

    # --- Build the teacher-forced forward (reused from eval_single_step) ---
    trainable_indices = config.trainable_scale_indices
    scales_t0 = config.scales
    scales_t1 = config.scales[:-1]
    tokens_t0 = sum(h * w for h, w in scales_t0)
    tokens_t1 = sum(h * w for h, w in scales_t1)
    padded_t0 = ((tokens_t0 + 127) // 128) * 128
    padded_t1 = ((tokens_t1 + 127) // 128) * 128
    attn_bias = build_teacher_forced_mask(scales_t0, padded_t0, scales_t1, padded_t1)
    predict_and_loss = make_predict_and_loss(
        config, scales_t0, padded_t0, scales_t1, padded_t1,
        attn_bias, scale_masks, trainable_indices, temperature=0.0)

    # --- Average teacher-forced predictive entropy over pairs ---
    n_pairs = min(len(indices) - 1, args.max_pairs)
    keys = jax.random.split(key, n_pairs)
    acc = np.zeros(len(trainable_indices), dtype=np.float64)
    for i in range(n_pairs):
        t0 = jnp.array(indices[i])
        t1 = jnp.array(indices[i + 1])
        _, _, _, _, per_scale_entropy = predict_and_loss(
            model, exp_heads, t0, t1, keys[i])
        acc += np.asarray(per_scale_entropy, dtype=np.float64)
    per_trainable = (acc / n_pairs).tolist()   # mean conditional entropy per trainable scale

    # --- Assemble JSON (same schema as measure_tokenizer_entropy.py) ---
    trainable_scale_indices = list(range(first_trainable, len(scales_int)))
    assert len(trainable_scale_indices) == len(per_trainable), \
        (trainable_scale_indices, len(per_trainable))
    per_scale = {
        str(scales_int[idx]): {"model_entropy_nats": per_trainable[j]}
        for j, idx in enumerate(trainable_scale_indices)
    }
    anchor = {
        "source": "model_teacher_forced",
        "checkpoint_dir": args.checkpoint_dir,
        "tokens_path": args.tokens_path,
        "n_pairs": int(n_pairs),
        "scales": scales_int,
        "first_trainable_scale": first_trainable,
        "trainable_scale_indices": trainable_scale_indices,
        "per_scale": per_scale,
        # Both standard slots hold the model conditional entropy so the rollout
        # loader resolves either --anchor_stat to it.
        "per_trainable_pooled_marginal_nats": per_trainable,
        "per_trainable_mean_per_position_nats": per_trainable,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(anchor, f, indent=2)

    print(f"wrote {args.output}  (n_pairs={n_pairs})")
    print(f"  {'scale':>6} {'model_H_nats':>12}")
    for j, idx in enumerate(trainable_scale_indices):
        print(f"  {scales_int[idx]:>6} {per_trainable[j]:>12.4f}")


if __name__ == "__main__":
    main()
