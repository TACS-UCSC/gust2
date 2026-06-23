"""
T2 — measure the frozen tokenizer's per-scale code entropy.

Produces the per-scale entropy ANCHOR consumed by the entropy-target / Top-H
samplers (`samplers.py`, `rollout_nsp.py --entropy_anchor_path`). The anchor
converts the per-config rollout temperature sweep into a data-read: instead of
sweeping a global temperature, the homeostat sets each scale's sampling entropy
to the entropy the frozen tokenizer actually assigns to real data at that scale.

Two definitions are emitted per scale (the experiment picks whichever matches
the swept-T EMD optimum):
  * pooled_marginal_nats  — H of the pooled code histogram over ALL positions
        and frames at the scale. Larger (folds in inter-position variability);
        a generous WARM target, default anchor for `entropy_target`.
  * mean_per_position_nats — mean over positions of each cell's code-distribution
        entropy. The realistic per-cell entropy; default ceiling for Top-H
        absolute mode.

Pure numpy over a tokenized `.npz` (no model, no re-tokenization): recomputes
the equivalent of `tokenizer.scale_usage_counts` (which lives in memory during
`fit()` but is not saved) directly from `indices_flat` via a single bincount.

Usage:
    python measure_tokenizer_entropy.py --tokens_path TOK.npz --output ANCHOR.json
    python measure_tokenizer_entropy.py --selftest
"""

import argparse
import json

import numpy as np


def _entropy_nats(probs: np.ndarray) -> float:
    """Shannon entropy (nats) of a 1-D probability vector; 0 log 0 := 0."""
    p = probs[probs > 0]
    return float(-(p * np.log(p)).sum())


def per_scale_entropies(indices_flat: np.ndarray, scales: np.ndarray, V: int):
    """Per-scale pooled-marginal and mean-per-position code entropy (nats).

    Args:
        indices_flat: (N, tokens_per_frame) int compact code indices.
        scales: (n_scales,) int side lengths (square scales).
        V: effective vocab size.

    Returns:
        dict {scale_value(int) -> {...}}, list boundaries.
    """
    scales = [int(s) for s in scales]
    counts_per_scale = [s * s for s in scales]
    boundaries = np.concatenate([[0], np.cumsum(counts_per_scale)]).astype(int)
    out = {}
    for k, s in enumerate(scales):
        a, b = boundaries[k], boundaries[k + 1]
        block = indices_flat[:, a:b].astype(np.int64)      # (N, Npos)
        N, Npos = block.shape

        # (a) pooled-marginal: one histogram over all positions+frames.
        pooled_hist = np.bincount(block.ravel(), minlength=V).astype(np.float64)
        pooled = _entropy_nats(pooled_hist / pooled_hist.sum())

        # (b) mean-per-position: per-column histogram via a single combined
        #     bincount (position * V + code), reshaped to (Npos, V).
        pos_idx = np.broadcast_to(np.arange(Npos), (N, Npos)).ravel()
        combined = pos_idx.astype(np.int64) * V + block.ravel()
        per_pos_counts = np.bincount(
            combined, minlength=Npos * V).reshape(Npos, V).astype(np.float64)
        totals = per_pos_counts.sum(axis=1, keepdims=True)
        per_pos_probs = per_pos_counts / np.maximum(totals, 1.0)
        # entropy per position (vectorized 0 log 0 := 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            xlogx = np.where(per_pos_probs > 0,
                             per_pos_probs * np.log(per_pos_probs), 0.0)
        per_pos_ent = -xlogx.sum(axis=1)                   # (Npos,)
        mean_pp = float(per_pos_ent.mean())

        out[s] = {
            "pooled_marginal_nats": pooled,
            "mean_per_position_nats": mean_pp,
            "per_position_min_nats": float(per_pos_ent.min()),
            "per_position_max_nats": float(per_pos_ent.max()),
            "n_positions": int(Npos),
            "n_unique_codes_pooled": int((pooled_hist > 0).sum()),
        }
    return out, boundaries


def build_anchor(tokens_path: str) -> dict:
    d = np.load(tokens_path, allow_pickle=True)
    indices_flat = np.asarray(d["indices_flat"])
    scales = [int(s) for s in np.asarray(d["scales"]).ravel()]
    V = int(d["effective_vocab_size"])
    first_trainable = int(d["first_trainable_scale"]) if "first_trainable_scale" in d.files else 0
    if "scale_masks" in d.files:
        support = [int(np.asarray(d["scale_masks"])[k].sum()) for k in range(len(scales))]
    else:
        support = [None] * len(scales)

    per_scale, _ = per_scale_entropies(indices_flat, scales, V)
    for k, s in enumerate(scales):
        per_scale[s]["support_size"] = support[k]

    n_scales = len(scales)
    trainable_idx = list(range(first_trainable, n_scales))
    pooled_arr = [per_scale[scales[i]]["pooled_marginal_nats"] for i in trainable_idx]
    pp_arr = [per_scale[scales[i]]["mean_per_position_nats"] for i in trainable_idx]

    return {
        "tokens_path": tokens_path,
        "n_frames": int(indices_flat.shape[0]),
        "effective_vocab_size": V,
        "scales": scales,
        "first_trainable_scale": first_trainable,
        "trainable_scale_indices": trainable_idx,
        # per_scale keyed by scale VALUE (string in JSON)
        "per_scale": {str(s): per_scale[s] for s in scales},
        # pre-packed in trainable order -> aligns 1:1 with enumerate(trainable_indices)
        "per_trainable_pooled_marginal_nats": pooled_arr,
        "per_trainable_mean_per_position_nats": pp_arr,
    }


def _selftest():
    # Synthetic 2-scale dataset: scale 0 (1x1) single code -> H=0;
    # scale 1 (2x2) each position uniform over 2 codes -> H=log 2.
    N, V = 1000, 4
    scales = np.array([1, 2])
    # tokens_per_frame = 1 + 4 = 5
    idx = np.zeros((N, 5), dtype=np.int64)
    idx[:, 0] = 0                                   # scale 0: always code 0
    rng = np.random.default_rng(0)
    for c in range(1, 5):                           # scale 1 positions: codes {2,3} 50/50
        idx[:, c] = rng.integers(2, 4, size=N)
    per_scale, _ = per_scale_entropies(idx, scales, V)
    assert abs(per_scale[1]["pooled_marginal_nats"] - 0.0) < 1e-9, per_scale[1]
    assert abs(per_scale[2]["pooled_marginal_nats"] - np.log(2)) < 1e-2, per_scale[2]
    assert abs(per_scale[2]["mean_per_position_nats"] - np.log(2)) < 1e-2, per_scale[2]
    # per_trainable arrays length == len(trainable indices)
    np.savez("/tmp/_t2_self.npz", indices_flat=idx.astype(np.int32),
             scales=scales, effective_vocab_size=V, first_trainable_scale=1,
             scale_masks=np.ones((2, V), dtype=bool))
    anchor = build_anchor("/tmp/_t2_self.npz")
    assert len(anchor["per_trainable_pooled_marginal_nats"]) == 1
    assert anchor["trainable_scale_indices"] == [1]
    print("[ok] T2 self-test passed (H=0 single-code, H=log2 uniform-2, alignment)")


def build_data_prior(tokens_path: str) -> dict:
    """Per-scale pooled empirical code distribution, tiled to (tokens_per_frame, V).

    For each scale, the distribution is the pooled code histogram over ALL
    positions+frames at that scale (well-sampled, unlike the noisy per-position
    frequencies), tiled across the scale's positions so the array is
    SHAPE-IDENTICAL to the position support mask and threads through the rollout
    the same way. The data_mix sampler restricts it to each position's support
    at sample time. Returns a dict of arrays for np.savez.
    """
    d = np.load(tokens_path, allow_pickle=True)
    idx = np.asarray(d["indices_flat"])
    V = int(d["effective_vocab_size"])
    scales = [int(s) for s in np.asarray(d["scales"]).ravel()]
    cps = [s * s for s in scales]
    bnd = np.concatenate([[0], np.cumsum(cps)]).astype(int)
    table = np.zeros((int(sum(cps)), V), dtype=np.float32)
    for k, s in enumerate(scales):
        block = idx[:, bnd[k]:bnd[k + 1]]
        pooled = np.bincount(block.ravel(), minlength=V).astype(np.float64)
        g = (pooled / pooled.sum()).astype(np.float32)
        table[bnd[k]:bnd[k + 1], :] = g[None, :]
    first_trainable = int(d["first_trainable_scale"]) if "first_trainable_scale" in d.files else 0
    return {
        "data_prior": table,                                  # (tokens_per_frame, V) f32
        "scales": np.array(scales),
        "effective_vocab_size": np.int64(V),
        "new_to_old": np.asarray(d["new_to_old"]),
        "first_trainable_scale": np.int64(first_trainable),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokens_path", type=str, help="tokenized .npz")
    ap.add_argument("--output", type=str, help="output anchor .json")
    ap.add_argument("--data_prior_out", type=str, default=None,
                    help="If set, also write the per-scale data-prior table "
                         "(.npz) consumed by the data_mix sampler.")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.tokens_path or not (args.output or args.data_prior_out):
        ap.error("--tokens_path and at least one of --output / --data_prior_out "
                 "are required (or use --selftest)")

    if args.data_prior_out:
        prior = build_data_prior(args.tokens_path)
        np.savez(args.data_prior_out, **prior)
        t = prior["data_prior"]
        print(f"wrote {args.data_prior_out}  data_prior shape={t.shape} "
              f"V={int(prior['effective_vocab_size'])}  scales={list(prior['scales'])}")

    if not args.output:
        return

    anchor = build_anchor(args.tokens_path)
    with open(args.output, "w") as f:
        json.dump(anchor, f, indent=2)

    print(f"wrote {args.output}")
    print(f"  frames={anchor['n_frames']}  V={anchor['effective_vocab_size']}  "
          f"scales={anchor['scales']}  trainable={anchor['trainable_scale_indices']}")
    print(f"  {'scale':>6} {'pooled':>10} {'mean/pos':>10} {'support':>8} {'npos':>6}")
    for s in anchor["scales"]:
        e = anchor["per_scale"][str(s)]
        print(f"  {s:>6} {e['pooled_marginal_nats']:>10.4f} "
              f"{e['mean_per_position_nats']:>10.4f} "
              f"{str(e['support_size']):>8} {e['n_positions']:>6}")


if __name__ == "__main__":
    main()
