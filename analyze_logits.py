"""analyze_logits.py — per-rollout diagnostic plots from rollout_logits.npz.

Loads the top-K logits + indices recorded by `rollout_nsp.py --log_topk K`,
plus the predicted token IDs from rollout_tokens.npz, and computes per-token
diagnostics. The hypothesis being tested is that explosion is preceded by
either (a) a confident-but-wrong prediction (high top-1 prob, OOD sample)
or (b) a high-entropy frame where the sampler reaches into the tail and
poisons the autoregressive context.

We compute, per emitted token:
  - top-1 head softmax probability  (= max prob among captured top-K)
  - entropy over top-K head softmax  (lower bound on full entropy)
  - top-K coverage  (sum of head softmax — 1.0 if K covers the full mass)
  - rank of the sampled token within top-K (or -1 if it fell outside)
  - sampled-token head logprob (or NaN if outside top-K)

Aggregations per frame:
  - mean over emitted (trainable-scale) tokens
  - per-scale means

Outputs (per --rollout_dir):
  diagnostics.npz  — per-trajectory per-frame numeric traces
  diagnostics.png  — 2×3 panel:
      (a) mean top-1 prob vs t            (b) mean entropy vs t
      (c) frac outside top-K vs t         (d) per-scale entropy heatmap
      (e) per-scale outside-rate heatmap  (f) pre/post-explosion top-1 hist
  Lines colored by survival; vertical markers per traj at explosion time
  if the matching multitraj_survival.py output is available.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax


def per_token_stats(top_logits, top_indices, sampled_ids):
    """Compute per-emission diagnostics from top-K logits.

    Args:
        top_logits:  (..., K) float
        top_indices: (..., K) int — token IDs within top-K
        sampled_ids: (...,)   int — actually-sampled token ID per emission

    Returns dict of (...,) arrays:
        top1_prob, entropy, coverage, sampled_rank, sampled_logprob
    """
    L = top_logits.astype(np.float32)
    # Head softmax — note: sums to 1 by construction over the K entries,
    # so coverage is computed against the full distribution by re-softmaxing
    # against the *unnormalized* logits via logsumexp.
    max_l = L.max(axis=-1, keepdims=True)
    el = np.exp(L - max_l)
    sum_el = el.sum(axis=-1, keepdims=True)
    head_probs = el / sum_el                                # (..., K)

    # Coverage estimate vs *full* dist: sum_el / sum_el_full. We don't have
    # the full denom, but we can still report top-1 prob within head and
    # entropy within head. Coverage proxy = 1 - top-K-tail-mass-share, which
    # we don't know. Cleanest: report sum(head_probs) which is always 1, and
    # separately the *gap* between top-1 logit and K-th logit (small gap →
    # mass likely escapes top-K).
    top1_prob = head_probs[..., 0]
    # Truncated entropy (sums to 1 over top-K)
    entropy = -(head_probs * np.log(head_probs + 1e-30)).sum(axis=-1)
    # logit_gap: top-1 minus K-th captured logit (in nats); large = peaky
    logit_gap = L[..., 0] - L[..., -1]

    # Rank of sampled id within top-K
    matches = (top_indices == sampled_ids[..., None])        # (..., K)
    in_topk = matches.any(axis=-1)                            # (...,)
    rank = matches.argmax(axis=-1)                            # 0..K-1 if found
    sampled_rank = np.where(
        in_topk, rank.astype(np.float32), np.nan).astype(np.float32)
    # Probability assigned to the sampled token (within head). NaN if not.
    flat_probs = head_probs.reshape(-1, head_probs.shape[-1])
    sampled_prob_in = flat_probs[
        np.arange(rank.size), rank.ravel()
    ].reshape(rank.shape)
    sampled_prob = np.where(in_topk, sampled_prob_in, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        sampled_logprob = np.log(sampled_prob)

    return {
        "top1_prob": top1_prob,
        "entropy": entropy,
        "logit_gap": logit_gap,
        "in_topk": in_topk,
        "sampled_rank": sampled_rank,
        "sampled_logprob": sampled_logprob,
    }


def aggregate_per_frame(stats, mask_trainable, scale_ids, n_scales):
    """Aggregate per-token stats over trainable positions per frame.

    Args:
        stats: dict of (N, T, tokens) arrays
        mask_trainable: (tokens,) bool — True for trainable-scale slots
        scale_ids: (tokens,) int — scale index per slot
        n_scales: int

    Returns:
        frame: dict with (N, T) arrays — overall means
        per_scale: dict with (N, T, n_scales) arrays — per-scale means
                    (NaN for non-trainable scales)
    """
    N, T, tok = stats["top1_prob"].shape
    overall = {}
    per_scale = {}

    for k, v in stats.items():
        if v.dtype == bool:
            v = v.astype(np.float32)
        else:
            v = v.astype(np.float32)
        # Overall: nanmean over trainable positions (sampled_logprob has NaN
        # where the sampled token fell outside top-K).
        m = mask_trainable[None, None, :]
        masked = np.where(m, v, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            overall[k] = np.nanmean(masked, axis=-1)

        # Per-scale
        ps = np.full((N, T, n_scales), np.nan, dtype=np.float32)
        for s in range(n_scales):
            sm = (scale_ids == s) & mask_trainable
            if not sm.any():
                continue
            sub = v[..., sm]
            with np.errstate(invalid="ignore", divide="ignore"):
                ps[..., s] = np.nanmean(sub, axis=-1)
        per_scale[k] = ps

    return overall, per_scale


def build_scale_ids(scales, first_trainable_scale):
    """Build a (tokens,) array of scale index per slot, plus a trainable mask."""
    scale_ids = []
    for k, s in enumerate(scales):
        scale_ids.extend([k] * (s * s))
    scale_ids = np.array(scale_ids, dtype=np.int32)
    trainable = scale_ids >= first_trainable_scale
    return scale_ids, trainable


def load_explosion_times(rollout_dir, n_traj, n_steps):
    """Look for a sibling multitraj_survival.py output and return per-traj
    explosion times (or all-survived if nothing found)."""
    # Heuristic: ../survival/<cfg>/  or ../analysis/  — the analyzer's output
    # location varies per sweep. We just scan a few candidate roots.
    cfg = os.path.basename(os.path.dirname(rollout_dir.rstrip("/")))
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(rollout_dir)),
                     "survival", "survival.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            rollout_dir.rstrip("/")))), "survival", "survival.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            with open(c) as f:
                surv = json.load(f)
            if cfg in surv.get("configs", {}):
                et = np.array(surv["configs"][cfg]["explosion_t"])
                return et, c
    return np.full(n_traj, n_steps, dtype=np.int64), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True,
                   help="dir with rollout_tokens.npz + rollout_logits.npz")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--scale_ema", type=int, default=20,
                   help="EMA window for smoothing per-frame traces")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.rollout_dir}")
    tok_npz = np.load(os.path.join(args.rollout_dir, "rollout_tokens.npz"),
                      allow_pickle=True)
    log_npz = np.load(os.path.join(args.rollout_dir, "rollout_logits.npz"),
                      allow_pickle=True)
    scales = log_npz["scales"].tolist()
    n_scales = len(scales)
    first_trainable = int(log_npz["first_trainable_scale"])
    n_steps = int(log_npz["n_steps"])
    log_topk = int(log_npz["log_topk"])

    # rollout_indices may be (T+1, tok) (legacy N=1) or (N, T+1, tok). The
    # logits array is (N, n_steps, tok, K) in the multi-traj case, with the
    # IC frame at step 0 *not* logged. Slot logits[t] corresponds to the
    # token sampled at frame t+1.
    rollout_indices = np.asarray(tok_npz["rollout_indices"])
    if rollout_indices.ndim == 2:
        rollout_indices = rollout_indices[None]
    top_logits = np.asarray(log_npz["top_logits"])
    top_indices = np.asarray(log_npz["top_indices"])
    if top_logits.ndim == 3:
        top_logits = top_logits[None]
        top_indices = top_indices[None]
    N, T, tok_per_frame, K = top_logits.shape
    print(f"  N={N}, n_steps={T}, tokens/frame={tok_per_frame}, K={K}")

    # Sampled IDs at frames 1..T (logits[t] corresponds to frame t+1)
    sampled_ids = rollout_indices[:, 1:T + 1, :]              # (N, T, tok)
    assert sampled_ids.shape == (N, T, tok_per_frame)

    scale_ids, trainable_mask = build_scale_ids(scales, first_trainable)

    print("Computing per-token stats...")
    stats = per_token_stats(top_logits, top_indices, sampled_ids)
    overall, per_scale = aggregate_per_frame(
        stats, trainable_mask, scale_ids, n_scales)
    frac_outside = 1.0 - overall["in_topk"]                   # (N, T)

    explosion_t, surv_src = load_explosion_times(args.rollout_dir, N, n_steps)
    survived = explosion_t >= n_steps
    n_surv = int(survived.sum())
    print(f"  Explosion source: {surv_src or 'none — assumed all survived'}")
    print(f"  Survived: {n_surv}/{N}")

    # ---- Save numeric traces ----
    out_npz = os.path.join(args.output_dir, "diagnostics.npz")
    np.savez_compressed(
        out_npz,
        scales=np.array(scales),
        first_trainable_scale=first_trainable,
        explosion_t=explosion_t,
        **{f"frame_{k}": v for k, v in overall.items()},
        **{f"per_scale_{k}": v for k, v in per_scale.items()},
        frac_outside_topk=frac_outside,
    )
    print(f"  Saved {out_npz}")

    # ---- Plots ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ts = np.arange(1, T + 1)
    surv_color = "C2"
    coll_color = "C3"

    def plot_traces(ax, y, title, ylabel):
        for j in range(N):
            color = surv_color if survived[j] else coll_color
            ax.plot(ts, y[j], lw=0.6, alpha=0.55, color=color)
            if not survived[j]:
                ax.axvline(explosion_t[j], color=color,
                           lw=0.4, alpha=0.3, ls="--")
        ax.set_xlabel("rollout step t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plot_traces(axes[0, 0], overall["top1_prob"],
                f"Top-1 head prob (mean over trainable tokens)",
                "P(arg-max)")
    plot_traces(axes[0, 1], overall["entropy"],
                f"Top-{K} head entropy", "nats")
    plot_traces(axes[0, 2], frac_outside,
                f"Frac sampled outside top-{K}",
                "fraction")

    # Per-scale heatmap of entropy (mean over trajs)
    ent_ps = np.nanmean(per_scale["entropy"], axis=0)         # (T, n_scales)
    out_ps = 1.0 - np.nanmean(per_scale["in_topk"], axis=0)
    trainable_scales = list(range(first_trainable, n_scales))

    def heatmap(ax, M, title, cmap="viridis"):
        # M: (T, n_scales) — keep only trainable scales
        sub = M[:, trainable_scales].T                        # (S, T)
        im = ax.imshow(sub, aspect="auto", origin="lower", cmap=cmap,
                       extent=[1, T, -0.5, len(trainable_scales) - 0.5])
        ax.set_yticks(range(len(trainable_scales)))
        ax.set_yticklabels([f"{scales[s]}x{scales[s]}" for s in trainable_scales],
                           fontsize=8)
        ax.set_xlabel("rollout step t")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    heatmap(axes[1, 0], ent_ps,
            f"Per-scale top-{K} entropy (traj mean)")
    heatmap(axes[1, 1], out_ps,
            f"Per-scale frac outside top-{K} (traj mean)",
            cmap="magma")

    # (1, 2): pre/post explosion top-1 prob distribution for collapsed trajs
    ax = axes[1, 2]
    if (~survived).any():
        pre = []
        post = []
        for j in range(N):
            if survived[j]:
                continue
            et = int(explosion_t[j])
            pre.extend(overall["top1_prob"][j, :et].tolist())
            post.extend(overall["top1_prob"][j, et:].tolist())
        bins = np.linspace(0, 1, 41)
        ax.hist(pre, bins=bins, density=True, alpha=0.55, color=surv_color,
                label=f"pre-explosion ({len(pre)} frames)")
        ax.hist(post, bins=bins, density=True, alpha=0.55, color=coll_color,
                label=f"post-explosion ({len(post)} frames)")
        ax.set_xlabel("mean top-1 head prob (per frame)")
        ax.set_ylabel("density")
        ax.set_title("Top-1 prob distribution")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no collapsed trajectories",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    fig.suptitle(
        f"{os.path.basename(os.path.dirname(args.rollout_dir.rstrip('/')))}"
        f" — N={N}, K={K}, survived {n_surv}/{N}", y=1.00)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, "diagnostics.png")
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"  Saved {out_png}")


if __name__ == "__main__":
    main()
