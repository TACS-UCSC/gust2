"""F5.1 — token-level OOD-rate traces for the E2 3-way mask ablation.

The E2 EMD read is quiet (no mode-A pixel explosion for the no-mask arm in
16x2000 warm steps, all arms trained WITH substitution noise), so M3's
evidence rides on COUNTABLE token-level failures: how often does each arm
emit a token outside the tokenizer's support? Two nested supports:

  per-scale OOD    token t at a scale-k position with scale_masks[k, t]=0.
                   The no-mask arm can do this; the per-scale and per-token
                   arms cannot by construction (exact-zero controls).
  per-position OOD token t at absolute position p never seen at p in the
                   TRAIN tokens (the per-token mask's support). The
                   per-scale arm CAN do this — per-scale support bounds the
                   marginal per scale, not per position — which is exactly
                   the M3 mechanism gap. The per-token arm is the zero
                   control here.

All index spaces are the tokenizer's compact mapping; the rollout npz and
the train-token npz must share new_to_old (asserted). Position support is
built from train tokens (what the per-token emission mask was fit on), NOT
from the rollout's own gt_indices.

Usage:
  python analyze_mask_ablation_ood.py \
      --ablation_root ~/gust2/artifacts/mask-ablation \
      --train_tokens  ~/gust2/artifacts/tokens/small-sc917.npz \
      --output_dir    plots/mask_ablation_ood
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["nomask", "perscale", "pertoken"]
ARM_LABELS = {"nomask": "no mask", "perscale": "per-scale mask",
              "pertoken": "per-token mask"}
ARM_COLORS = {"nomask": "#d62728", "perscale": "#ff7f0e",
              "pertoken": "#2ca02c"}


def parse_args():
    p = argparse.ArgumentParser(description="E2 mask-ablation OOD-rate traces")
    p.add_argument("--ablation_root", required=True,
                   help="Dir with <arm>/T<temp>/rollout_tokens.npz")
    p.add_argument("--train_tokens", required=True,
                   help="Train token npz sharing the rollouts' compact map")
    p.add_argument("--temps", nargs="+", default=["0p7", "1p6"])
    p.add_argument("--output_dir", default="plots/mask_ablation_ood")
    p.add_argument("--smooth", type=int, default=25,
                   help="Moving-average window (steps) for the rate traces")
    return p.parse_args()


def scale_position_ids(scales):
    """Per-token scale index and absolute position for one frame."""
    sids = np.concatenate([np.full(s * s, i, dtype=np.int64)
                           for i, s in enumerate(scales)])
    return sids, np.arange(len(sids))


def build_position_support(train_tokens_path, n_positions, vocab):
    """(P, V) bool: token v observed at absolute position p in train."""
    d = np.load(train_tokens_path, allow_pickle=True)
    idx = d["indices_flat"]                  # (n_frames, P) compact ids
    assert idx.shape[1] == n_positions, \
        f"train tokens P={idx.shape[1]} != rollout P={n_positions}"
    support = np.zeros((n_positions, vocab), dtype=bool)
    pos = np.arange(n_positions)
    for row in idx:
        support[pos, row] = True
    return support, d


def smooth(x, w):
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Shared geometry from one rollout npz ---
    ref_path = os.path.join(args.ablation_root, ARMS[0], f"T{args.temps[0]}",
                            "rollout_tokens.npz")
    ref = np.load(ref_path, allow_pickle=True)
    scales = tuple(int(s) for s in ref["scales"])
    scale_masks = ref["scale_masks"].astype(bool)      # (n_scales, V)
    vocab = scale_masks.shape[1]
    new_to_old_ref = ref["new_to_old"]
    sids, _pos = scale_position_ids(scales)
    P = len(sids)
    print(f"scales={scales}  P={P}  vocab={vocab}")

    print(f"Building position support from {args.train_tokens}...")
    pos_support, train_d = build_position_support(args.train_tokens, P, vocab)
    assert np.array_equal(np.asarray(train_d["new_to_old"]), np.asarray(new_to_old_ref)), \
        "compact map mismatch between train tokens and rollouts"
    print(f"  mean per-position support: {pos_support.sum(axis=1).mean():.1f} "
          f"tokens (vs per-scale mean "
          f"{scale_masks[sids].sum(axis=1).mean():.1f})")

    per_scale_lookup = scale_masks[sids]               # (P, V) bool

    results = {}
    for temp in args.temps:
        for arm in ARMS:
            path = os.path.join(args.ablation_root, arm, f"T{temp}",
                                "rollout_tokens.npz")
            d = np.load(path, allow_pickle=True)
            assert np.array_equal(np.asarray(d["new_to_old"]),
                                  np.asarray(new_to_old_ref)), \
                f"compact map mismatch: {path}"
            r = d["rollout_indices"]                   # (N, T, P)
            N, T, _ = r.shape

            pos_idx = np.broadcast_to(np.arange(P), r.shape)
            scale_ood = ~per_scale_lookup[pos_idx, r]  # (N, T, P)
            pos_ood = ~pos_support[pos_idx, r]

            entry = {
                "scale_ood_rate": float(scale_ood.mean()),
                "pos_ood_rate": float(pos_ood.mean()),
                "scale_ood_trace": scale_ood.mean(axis=(0, 2)),   # (T,)
                "pos_ood_trace": pos_ood.mean(axis=(0, 2)),
                "pos_ood_frame_frac": float(pos_ood.any(axis=2).mean()),
                "per_scale_breakdown": [
                    float(pos_ood[..., sids == k].mean())
                    for k in range(len(scales))],
                "N": N, "T": T,
            }
            results[(arm, temp)] = entry
            print(f"{arm:10s} T{temp}: scale-OOD {entry['scale_ood_rate']:.2e}  "
                  f"pos-OOD {entry['pos_ood_rate']:.2e}  "
                  f"(frames w/ any pos-OOD: {entry['pos_ood_frame_frac']:.1%})")

    # --- Plots: one panel per temp, pos-OOD rate traces per arm ---
    fig, axes = plt.subplots(1, len(args.temps),
                             figsize=(7 * len(args.temps), 5), sharey=True)
    if len(args.temps) == 1:
        axes = [axes]
    for ax, temp in zip(axes, args.temps):
        for arm in ARMS:
            e = results[(arm, temp)]
            ax.semilogy(smooth(e["pos_ood_trace"], args.smooth),
                        color=ARM_COLORS[arm], lw=1.5,
                        label=f"{ARM_LABELS[arm]} (mean {e['pos_ood_rate']:.1e})")
            if e["scale_ood_rate"] > 0:
                ax.semilogy(smooth(e["scale_ood_trace"], args.smooth),
                            color=ARM_COLORS[arm], lw=1.0, ls="--", alpha=0.6,
                            label=f"{ARM_LABELS[arm]} scale-OOD")
        ax.set_xlabel("Rollout step")
        ax.set_title(f"T = {temp.replace('p', '.')}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Per-token OOD rate (position support)")
    fig.suptitle("E2 mask ablation — token-level OOD emission rate "
                 "(solid: vs per-position support; dashed: vs per-scale mask)")
    out = os.path.join(args.output_dir, "ood_rate_traces.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # --- Per-scale breakdown bar chart (warm temp) ---
    warm = args.temps[-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(scales))
    width = 0.25
    for i, arm in enumerate(ARMS):
        ax.bar(x + (i - 1) * width, results[(arm, warm)]["per_scale_breakdown"],
               width, color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{s}x{s}" for s in scales])
    ax.set_xlabel("Scale")
    ax.set_ylabel("Position-OOD rate")
    ax.set_title(f"Position-OOD rate by scale (T={warm.replace('p', '.')})")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(args.output_dir, "ood_by_scale.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # --- JSON summary ---
    summary = {f"{arm}/T{temp}": {k: v for k, v in e.items()
                                  if not isinstance(v, np.ndarray)}
               for (arm, temp), e in results.items()}
    with open(os.path.join(args.output_dir, "ood_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(args.output_dir, "ood_traces.npz"),
             **{f"{arm}_T{temp}_{key}": e[key]
                for (arm, temp), e in results.items()
                for key in ("pos_ood_trace", "scale_ood_trace")})
    print("Done.")


if __name__ == "__main__":
    main()
