"""analyze_logits.py — per-rollout diagnostic plots from rollout_logits.npz.

Loads the top-K logits + indices recorded by `rollout_nsp.py --log_topk K`,
plus the predicted token IDs from rollout_tokens.npz, and computes per-token
diagnostics. The hypothesis being tested is that collapse is preceded by
either (a) a confident-but-wrong prediction (high top-1 prob, OOD sample)
or (b) a high-entropy frame where the sampler reaches into the tail and
poisons the autoregressive context.

We compute, per emitted token:
  - top-1 head softmax probability  (= max prob among captured top-K)
  - entropy over top-K head softmax  (lower bound on full entropy)
  - logit gap (top-1 minus K-th captured logit; large = peaky head)
  - rank of the sampled token within top-K (or -1 if it fell outside)
  - sampled-token head logprob (or NaN if outside top-K)

Aggregations per frame:
  - mean over emitted (trainable-scale) tokens
  - per-scale means

Outputs (per --rollout_dir):
  diagnostics.npz  — per-trajectory per-frame numeric traces
  diagnostics.png  — 2×3 panel: survived/collapsed median+IQR bands for
      top-1 prob, entropy, frac-outside-top-K (or logit gap when that
      panel is identically zero), per-scale heatmaps, pre/post-explosion
      top-1 histogram. Explosion times drawn as a rug along the x-axis.

Pass --survival_json to align against multitraj_survival.py explosion
times; without it a sibling <root>/survival/survival.json is tried, else
all trajectories are assumed survived.
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from diagnostics_common import (
    add_wandb_args,
    band_plot,
    build_scale_ids,
    describe_decode,
    explosion_rug,
    get_explosion_times,
    init_wandb,
    is_effectively_zero,
    load_cfg_meta,
    load_survival,
    outside_legend,
    safe_median,
    set_diag_style,
    wandb_log_figs_and_scalars,
)


def per_token_stats(top_logits, top_indices, sampled_ids):
    """Compute per-emission diagnostics from top-K logits.

    Args:
        top_logits:  (..., K) float
        top_indices: (..., K) int — token IDs within top-K
        sampled_ids: (...,)   int — actually-sampled token ID per emission

    Returns dict of (...,) arrays:
        top1_prob, entropy, logit_gap, in_topk, sampled_rank, sampled_logprob
    """
    L = top_logits.astype(np.float32)
    # Head softmax — note: sums to 1 by construction over the K entries,
    # so coverage is computed against the full distribution by re-softmaxing
    # against the *unnormalized* logits via logsumexp.
    max_l = L.max(axis=-1, keepdims=True)
    el = np.exp(L - max_l)
    sum_el = el.sum(axis=-1, keepdims=True)
    head_probs = el / sum_el                                # (..., K)

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


def load_explosion_times(args, rollout_dir, n_traj, n_steps):
    """Explosion times from --survival_json, falling back to the sibling
    <root>/survival/survival.json heuristic, else all-survived."""
    cfg = os.path.basename(os.path.dirname(rollout_dir.rstrip("/")))
    candidates = []
    if args.survival_json:
        candidates.append(args.survival_json)
    candidates += [
        os.path.join(os.path.dirname(os.path.dirname(rollout_dir)),
                     "survival", "survival.json"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            rollout_dir.rstrip("/")))), "survival", "survival.json"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            surv = load_survival(c)
            if cfg in surv.get("configs", {}):
                et, _, _ = get_explosion_times(surv, cfg, n_traj)
                return et, c
    return np.full(n_traj, n_steps, dtype=np.int64), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rollout_dir", required=True,
                   help="dir with rollout_tokens.npz + rollout_logits.npz")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--survival_json", default=None,
                   help="multitraj_survival.py output; explosion times for "
                        "this cfg are looked up by directory name. Falls "
                        "back to a sibling-path heuristic when omitted.")
    p.add_argument("--cfg_name", default=None,
                   help="cfg label for titles/wandb (default: rollout_dir's "
                        "parent directory name)")
    p.add_argument("--scale_ema", type=int, default=20,
                   help="EMA window for smoothing per-frame traces in plots "
                        "(npz traces stay unsmoothed)")
    add_wandb_args(p)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cfg_name = args.cfg_name or os.path.basename(
        os.path.dirname(args.rollout_dir.rstrip("/")))
    cfg_meta = load_cfg_meta(os.path.dirname(args.rollout_dir.rstrip("/")))

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

    explosion_t, surv_src = load_explosion_times(
        args, args.rollout_dir, N, n_steps)
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
    set_diag_style()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ts = np.arange(1, T + 1)
    surv_color = "C2"
    coll_color = "C3"

    def plot_groups(ax, y, title, ylabel):
        """Survived/collapsed median+IQR bands; explosion times as a rug."""
        band_plot(ax, ts, y[survived], color=surv_color, label="survived",
                  smooth=args.scale_ema)
        band_plot(ax, ts, y[~survived], color=coll_color, label="collapsed",
                  smooth=args.scale_ema)
        explosion_rug(ax, explosion_t, n_steps, color=coll_color)
        ax.set_xlabel("rollout step t")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best")

    plot_groups(axes[0, 0], overall["top1_prob"],
                "Top-1 head prob (mean over trainable tokens)",
                "P(arg-max)")
    plot_groups(axes[0, 1], overall["entropy"],
                f"Top-{K} head entropy", "nats")
    # The outside-top-K panel is dead weight when sampling never leaves the
    # captured top-K (typical for cold temperatures); show the logit gap —
    # head peakiness — instead.
    if is_effectively_zero(frac_outside, tol=1e-6):
        plot_groups(axes[0, 2], overall["logit_gap"],
                    f"Logit gap top-1 − top-{K}\n"
                    f"(frac outside top-{K} ≡ 0)", "nats")
    else:
        plot_groups(axes[0, 2], frac_outside,
                    f"Frac sampled outside top-{K}", "fraction")

    # Per-scale heatmaps (mean over trajs)
    ent_ps = np.nanmean(per_scale["entropy"], axis=0)         # (T, n_scales)
    out_ps = 1.0 - np.nanmean(per_scale["in_topk"], axis=0)
    trainable_scales = list(range(first_trainable, n_scales))

    def heatmap(ax, M, title, cmap="viridis"):
        # M: (T, n_scales) — keep only trainable scales
        sub = M[:, trainable_scales].T                        # (S, T)
        im = ax.imshow(sub, aspect="auto", origin="lower", cmap=cmap,
                       extent=[1, T, -0.5, len(trainable_scales) - 0.5])
        ax.set_yticks(range(len(trainable_scales)))
        ax.set_yticklabels(
            [f"{scales[s]}x{scales[s]}" for s in trainable_scales])
        ax.set_xlabel("rollout step t")
        ax.set_title(title)
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    heatmap(axes[1, 0], ent_ps,
            f"Per-scale top-{K} entropy (traj mean)")
    # Same dead-panel treatment for the per-scale outside-rate heatmap.
    if is_effectively_zero(out_ps, tol=1e-6):
        gap_ps = np.nanmean(per_scale["logit_gap"], axis=0)
        heatmap(axes[1, 1], gap_ps,
                f"Per-scale logit gap (traj mean)\n"
                f"(outside-top-{K} rate ≡ 0)",
                cmap="magma")
    else:
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
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no collapsed trajectories",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")

    desc = describe_decode(cfg_meta)
    fig.suptitle(
        f"{cfg_name}{f'  [{desc}]' if desc else ''}"
        f" — N={N}, K={K}, survived {n_surv}/{N}")
    out_png = os.path.join(args.output_dir, "diagnostics.png")
    fig.savefig(out_png)
    print(f"  Saved {out_png}")

    # ---- Wandb ----
    run = init_wandb(args, job_type="logits", config={
        "cfg_name": cfg_name,
        "rollout_dir": args.rollout_dir,
        "n_trajectories": N,
        "n_steps": T,
        "log_topk": K,
        "scale_ema": args.scale_ema,
        **{k: v for k, v in cfg_meta.items()
           if k in ("temperature", "top_k", "top_p", "seed",
                    "position_mask_used")},
    })
    w = min(100, T)
    scalars = {
        "n_trajectories": N,
        "n_survived": n_surv,
        "frac_outside_mean": float(np.nanmean(frac_outside)),
        "top1_prob_mean_first100": float(
            np.nanmean(overall["top1_prob"][:, :w])),
        "top1_prob_mean_last100": float(
            np.nanmean(overall["top1_prob"][:, -w:])),
        "entropy_mean_first100": float(
            np.nanmean(overall["entropy"][:, :w])),
        "entropy_mean_last100": float(
            np.nanmean(overall["entropy"][:, -w:])),
    }
    series = {}
    for key, label in (("top1_prob", "top-1 prob"), ("entropy", "entropy")):
        ydict = {}
        if n_surv:
            ydict["survived"] = safe_median(overall[key][survived], axis=0)
        if n_surv < N:
            ydict["collapsed"] = safe_median(overall[key][~survived], axis=0)
        if ydict:
            series[f"{key}_median"] = (
                ts, ydict, "rollout step t", f"Median {label} vs t")
    wandb_log_figs_and_scalars(
        run, scalars=scalars, figs={"diagnostics": fig}, line_series=series)


if __name__ == "__main__":
    main()
