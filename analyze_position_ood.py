"""analyze_position_ood.py — does the model emit a token at a position
where it never saw that token during training?

Hypothesis: scale_masks is per-scale, but in practice each absolute
position p ∈ [0, 341) has only a small subset of tokens it ever takes in
the training set. The mask doesn't enforce that; the model could (and
maybe does) emit a token that is *scale-legal* but *position-OOD*, and
that emission is what kicks the trajectory off-manifold.

Pipeline:
  1) build per-position vocab Vp from training tokens (set of tokens
     observed at position p across all training frames).
  2) for each rollout token at (traj, t, p), flag whether it is OOD
     (token ∉ Vp).
  3) aggregate: per-frame OOD rate, per-scale OOD rate.
  4) align to explosion times; plot per-cfg + cross-cfg overlay.

Usage:
  python analyze_position_ood.py \\
    --train_tokens experiments/tokens/small-sc341.npz \\
    --logits_root  plots/sc341-multitraj/logits \\
    --survival_json plots/sc341-multitraj/survival/survival.json \\
    --output_dir   plots/sc341-multitraj/position_ood

Each <cfg>/rollout/rollout_tokens.npz must be present locally
(rsync from Derecho first).
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def build_position_membership(train_indices, vocab_size):
    """Returns a bool array (P, V) where M[p, v] = 1 iff token v appears
    at position p anywhere in the training set."""
    F, P = train_indices.shape
    M = np.zeros((P, vocab_size), dtype=bool)
    flat_p = np.broadcast_to(np.arange(P), (F, P)).ravel()
    flat_v = train_indices.ravel().astype(np.int64)
    M[flat_p, flat_v] = True
    return M


def position_scale_assignment(scales):
    """Map flat position p -> scale index si. scales = list of side
    lengths, one per scale. Returns (P,) int array."""
    P = sum(int(s) * int(s) for s in scales)
    pos_scale = np.zeros(P, dtype=np.int64)
    cursor = 0
    for si, s in enumerate(scales):
        n = int(s) * int(s)
        pos_scale[cursor:cursor + n] = si
        cursor += n
    return pos_scale


def aligned_window(traces, explosion_t, lo, hi):
    """traces (N, T) -> aligned (N, hi-lo) at τ = t - t_explode.
    Survived trajectories produce all-NaN rows."""
    N, T = traces.shape
    rel_T = hi - lo
    out = np.full((N, rel_T), np.nan, dtype=np.float32)
    for j in range(N):
        te = int(explosion_t[j])
        if te >= T:
            continue
        for k, tau in enumerate(range(lo, hi)):
            t_abs = te + tau
            if 0 <= t_abs < T:
                out[j, k] = traces[j, t_abs]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tokens", required=True,
                   help="path to training tokens npz "
                        "(e.g. small-sc341.npz)")
    p.add_argument("--logits_root", required=True,
                   help="dir containing <cfg>/rollout/rollout_tokens.npz")
    p.add_argument("--survival_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rel_lo", type=int, default=-500)
    p.add_argument("--rel_hi", type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading training tokens from {args.train_tokens}")
    td = np.load(args.train_tokens)
    train_idx = td["indices_flat"]                     # (F, 341)
    scales = td["scales"].tolist()                     # [1, 2, 4, 8, 16]
    first_trainable = int(td["first_trainable_scale"])
    effective_vocab = int(td["effective_vocab_size"])
    print(f"  {train_idx.shape[0]} frames, {train_idx.shape[1]} pos, "
          f"V={effective_vocab}, scales={scales}, "
          f"first_trainable={first_trainable}")

    # Sanity: the train tokens must come from the same VQ-VAE as the
    # rollout, or per-position vocabularies are nonsense. We check the
    # first available rollout's effective_vocab_size.
    sample_rollout_path = None
    for cfg in sorted(os.listdir(args.logits_root)):
        cand = os.path.join(args.logits_root, cfg, "rollout",
                            "rollout_tokens.npz")
        if os.path.isfile(cand):
            sample_rollout_path = cand
            break
    if sample_rollout_path is not None:
        rd = np.load(sample_rollout_path)
        roll_V = int(rd["effective_vocab_size"])
        if roll_V != effective_vocab:
            raise SystemExit(
                f"VQ-VAE mismatch: train tokens have V={effective_vocab} "
                f"but rollout {sample_rollout_path} has V={roll_V}. "
                f"Use the train tokens that came from the same VQ-VAE.")
        if not np.array_equal(td["new_to_old"], rd["new_to_old"]):
            raise SystemExit(
                f"VQ-VAE compact-vocab mapping (new_to_old) differs "
                f"between train tokens and rollout {sample_rollout_path}.")
        print(f"  vocab sanity OK (V={effective_vocab} matches rollout)")

    print("Building per-position vocab membership matrix...")
    M = build_position_membership(train_idx, effective_vocab)   # (P, V)
    pos_count = M.sum(axis=1)
    pos_scale = position_scale_assignment(scales)
    print(f"  per-position vocab size: "
          f"min={pos_count.min()} median={int(np.median(pos_count))} "
          f"max={pos_count.max()}")
    P = M.shape[0]
    boundaries = np.concatenate([[0], np.cumsum([s * s for s in scales])])
    for si, s in enumerate(scales):
        ps = pos_count[boundaries[si]:boundaries[si + 1]]
        print(f"  scale {si} ({s}x{s}, {len(ps)} pos):  "
              f"per-pos vocab min/med/max = "
              f"{ps.min()}/{int(np.median(ps))}/{ps.max()}")

    # Trainable positions only (matches NSP training: ignore scale 0).
    trainable_mask = pos_scale >= first_trainable      # (P,)
    n_trainable = int(trainable_mask.sum())
    print(f"  trainable positions: {n_trainable}/{P}")

    # ---------- per-cfg loop ----------
    with open(args.survival_json) as f:
        surv = json.load(f)
    n_frames = int(surv["n_frames"])
    cfgs = sorted(surv["configs"].keys())

    cfg_data = {}
    for cfg in cfgs:
        rpath = os.path.join(args.logits_root, cfg, "rollout",
                             "rollout_tokens.npz")
        if not os.path.isfile(rpath):
            print(f"[skip] missing {rpath}")
            continue
        rd = np.load(rpath)
        idx = rd["rollout_indices"]                    # (N, T+1, P)
        if idx.ndim == 2:
            idx = idx[None]
        N, Tp1, _ = idx.shape
        # The IC frame (t=0) is not generated; skip it so traces have
        # length T = T+1 - 1 = n_steps. That matches diagnostics.npz.
        gen = idx[:, 1:, :]                            # (N, T, P)
        T = gen.shape[1]
        # Look up per-position OOD: M[p, token] is in-vocab; OOD = ~M[...]
        ood_per_token = ~M[np.arange(P)[None, None, :],
                           gen.astype(np.int64)]       # (N, T, P)
        # Restrict to trainable positions
        ood_train = ood_per_token[..., trainable_mask].astype(np.float32)
        frame_ood = ood_train.mean(axis=2)              # (N, T)
        per_scale_ood = np.zeros((N, T, len(scales)), dtype=np.float32)
        for si in range(first_trainable, len(scales)):
            sel = (pos_scale == si)
            per_scale_ood[..., si] = ood_per_token[..., sel].mean(
                axis=2).astype(np.float32)

        # also: max-position OOD streak per traj (was ANY position OOD?)
        any_ood_per_frame = ood_train.any(axis=2)       # (N, T)

        et = np.array(surv["configs"][cfg]["explosion_t"])
        if et.shape[0] != N:
            print(f"[warn] {cfg}: surv N={et.shape[0]} != roll N={N}; "
                  f"truncating")
            n = min(et.shape[0], N)
            et = et[:n]; frame_ood = frame_ood[:n]
            per_scale_ood = per_scale_ood[:n]
            any_ood_per_frame = any_ood_per_frame[:n]
            N = n

        cfg_data[cfg] = dict(
            frame_ood=frame_ood, per_scale_ood=per_scale_ood,
            any_ood=any_ood_per_frame, et=et, N=N,
            collapsed=et < n_frames,
        )
        # quick textual readout
        first_ood_t = np.full(N, T, dtype=np.int64)
        for j in range(N):
            ts = np.where(any_ood_per_frame[j])[0]
            if ts.size:
                first_ood_t[j] = int(ts[0])
        med_first_ood = int(np.median(first_ood_t))
        gap_to_explode = et - first_ood_t
        print(f"\n[{cfg}]  N={N}, mean frame OOD rate = "
              f"{frame_ood.mean():.4f}, "
              f"median first-OOD t={med_first_ood}, "
              f"median (t_explode - first_ood) = "
              f"{int(np.median(gap_to_explode))}")

    if not cfg_data:
        raise SystemExit("No rollout_tokens.npz found in any cfg.")

    rel_axis = np.arange(args.rel_lo, args.rel_hi)

    # ---------- per-cfg figures ----------
    for cfg, cd in cfg_data.items():
        N = cd["N"]
        T = cd["frame_ood"].shape[1]
        et = cd["et"]
        coll = cd["collapsed"]
        n_coll = int(coll.sum())
        info = surv["configs"][cfg]
        title = (f"{cfg}  position-OOD diagnostic   "
                 f"S∞={info['survival_at_2000']:.0%}, "
                 f"med t_explode={info['median_t']}, N={N}")

        n_rows = 1 + (len(scales) - first_trainable)
        fig, axes = plt.subplots(n_rows, 2,
                                 figsize=(13, 2.0 * n_rows),
                                 sharex="col")
        axes = np.atleast_2d(axes)

        # row 0: overall frame OOD rate
        ax_abs, ax_rel = axes[0, 0], axes[0, 1]
        for j in range(N):
            color = "C3" if coll[j] else "C2"
            alpha = 0.55 if coll[j] else 0.35
            ax_abs.plot(np.arange(T), cd["frame_ood"][j], lw=0.5,
                        alpha=alpha, color=color)
            if coll[j]:
                ax_abs.axvline(et[j], color="C3", lw=0.3, alpha=0.2)
        ax_abs.set_ylabel("frame OOD rate\n(trainable pos)", fontsize=9)
        ax_abs.grid(True, alpha=0.3)

        if n_coll == 0:
            ax_rel.text(0.5, 0.5, "no collapse",
                        transform=ax_rel.transAxes,
                        ha="center", va="center", color="gray")
        else:
            aligned = aligned_window(cd["frame_ood"], et,
                                     args.rel_lo, args.rel_hi)
            for j in np.where(coll)[0]:
                ax_rel.plot(rel_axis, aligned[j], lw=0.4,
                            alpha=0.4, color="C3")
            med = np.nanmedian(aligned[coll], axis=0)
            q25 = np.nanpercentile(aligned[coll], 25, axis=0)
            q75 = np.nanpercentile(aligned[coll], 75, axis=0)
            ax_rel.fill_between(rel_axis, q25, q75, color="C0", alpha=0.2)
            ax_rel.plot(rel_axis, med, color="C0", lw=2,
                        label=f"median (n={n_coll})")
            ax_rel.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
            ax_rel.legend(loc="best", fontsize=8)
        ax_rel.grid(True, alpha=0.3)

        # rows 1..S: per-scale OOD rate
        for k, si in enumerate(range(first_trainable, len(scales))):
            row = 1 + k
            arr = cd["per_scale_ood"][..., si]              # (N, T)
            ax_abs, ax_rel = axes[row, 0], axes[row, 1]
            for j in range(N):
                color = "C3" if coll[j] else "C2"
                alpha = 0.55 if coll[j] else 0.35
                ax_abs.plot(np.arange(T), arr[j], lw=0.5,
                            alpha=alpha, color=color)
            ax_abs.set_ylabel(f"OOD scale {si}\n({scales[si]}×{scales[si]})",
                              fontsize=9)
            ax_abs.grid(True, alpha=0.3)

            if n_coll == 0:
                ax_rel.text(0.5, 0.5, "no collapse",
                            transform=ax_rel.transAxes,
                            ha="center", va="center", color="gray")
            else:
                aligned = aligned_window(arr, et,
                                         args.rel_lo, args.rel_hi)
                for j in np.where(coll)[0]:
                    ax_rel.plot(rel_axis, aligned[j], lw=0.4,
                                alpha=0.4, color="C3")
                med = np.nanmedian(aligned[coll], axis=0)
                q25 = np.nanpercentile(aligned[coll], 25, axis=0)
                q75 = np.nanpercentile(aligned[coll], 75, axis=0)
                ax_rel.fill_between(rel_axis, q25, q75,
                                    color="C0", alpha=0.2)
                ax_rel.plot(rel_axis, med, color="C0", lw=2)
                ax_rel.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
            ax_rel.grid(True, alpha=0.3)

        axes[0, 0].set_title("absolute time   "
                             "(red=collapsed, green=survived)")
        axes[0, 1].set_title(f"aligned to explosion   "
                             f"(τ ∈ [{args.rel_lo}, {args.rel_hi}))")
        axes[-1, 0].set_xlabel("absolute rollout step t")
        axes[-1, 1].set_xlabel("τ = t - t_explode")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out_path = os.path.join(args.output_dir, f"cfg_{cfg}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"saved {out_path}")

    # ---------- cross-cfg overlay ----------
    cfg_order = sorted(
        cfg_data.keys(),
        key=lambda c: -surv["configs"][c]["survival_at_2000"],
    )
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(cfg_order)))

    n_rows = 1 + (len(scales) - first_trainable)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(11, 2.4 * n_rows), sharex=True)

    ax = axes[0]
    for ci, cfg in enumerate(cfg_order):
        cd = cfg_data[cfg]
        coll = cd["collapsed"]
        if coll.sum() == 0:
            continue
        aligned = aligned_window(cd["frame_ood"], cd["et"],
                                 args.rel_lo, args.rel_hi)
        med = np.nanmedian(aligned[coll], axis=0)
        ax.plot(rel_axis, med, color=cmap[ci], lw=1.6, label=cfg)
    ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
    ax.set_ylabel("frame OOD rate", fontsize=9)
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    for k, si in enumerate(range(first_trainable, len(scales))):
        row = 1 + k
        ax = axes[row]
        for ci, cfg in enumerate(cfg_order):
            cd = cfg_data[cfg]
            coll = cd["collapsed"]
            if coll.sum() == 0:
                continue
            aligned = aligned_window(cd["per_scale_ood"][..., si], cd["et"],
                                     args.rel_lo, args.rel_hi)
            med = np.nanmedian(aligned[coll], axis=0)
            ax.plot(rel_axis, med, color=cmap[ci], lw=1.6, label=cfg)
        ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
        ax.set_ylabel(f"OOD scale {si}\n({scales[si]}×{scales[si]})",
                      fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("τ = t - t_explode")
    fig.suptitle(
        "Cross-cfg medians: position-OOD rate of collapsed trajectories",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path = os.path.join(args.output_dir, "overlay_relative.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")

    # ---------- summary npz ----------
    out_npz = os.path.join(args.output_dir, "position_ood.npz")
    save_dict = {"scales": np.array(scales),
                 "first_trainable_scale": first_trainable,
                 "pos_scale": pos_scale,
                 "per_pos_vocab_size": pos_count}
    for cfg, cd in cfg_data.items():
        save_dict[f"frame_ood_{cfg}"] = cd["frame_ood"]
        save_dict[f"per_scale_ood_{cfg}"] = cd["per_scale_ood"]
        save_dict[f"explosion_t_{cfg}"] = cd["et"]
    np.savez_compressed(out_npz, **save_dict)
    print(f"saved {out_npz}")


if __name__ == "__main__":
    main()
