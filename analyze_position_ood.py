"""analyze_position_ood.py — does the model emit a token at a position
where it never saw that token during training?

Hypothesis: scale_masks is per-scale, but in practice each absolute
position p has only a small subset of tokens it ever takes in the training
set. The per-scale mask doesn't enforce that; the model could (and does,
without the per-position mask) emit a token that is *scale-legal* but
*position-OOD*, and that emission is what kicks the trajectory
off-manifold.

When the rollouts were generated WITH the per-position mask
(rollout_nsp.py --train_tokens_path), this stage is a sanity check: the
OOD rate must be identically zero. A nonzero rate means the mask was not
actually applied (or train tokens mismatch) — the figure collapses to a
single PASS/FAIL panel in that case.

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
    --logits_root  <sweep_root> \\
    --survival_json <sweep_root>/survival/survival.json \\
    --output_dir   <sweep_root>/position_ood
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from diagnostics_common import (
    add_wandb_args,
    aligned_window,
    assign_cfg_styles,
    band_plot,
    build_scale_ids,
    describe_decode,
    get_explosion_times,
    init_wandb,
    is_effectively_zero,
    load_cfg_meta,
    load_rollout_tokens,
    load_survival,
    outside_legend,
    safe_median,
    set_diag_style,
    wandb_log_figs_and_scalars,
)


def build_position_membership(train_indices, vocab_size):
    """Returns a bool array (P, V) where M[p, v] = 1 iff token v appears
    at position p anywhere in the training set."""
    F, P = train_indices.shape
    M = np.zeros((P, vocab_size), dtype=bool)
    flat_p = np.broadcast_to(np.arange(P), (F, P)).ravel()
    flat_v = train_indices.ravel().astype(np.int64)
    M[flat_p, flat_v] = True
    return M


def ood_floor(n_positions):
    """Display floor for log-scale OOD axes: a tenth of one position."""
    return 0.1 / max(1, n_positions)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_tokens", required=True,
                   help="path to training tokens npz "
                        "(e.g. experiments/tokens/small-sc341.npz)")
    p.add_argument("--logits_root", required=True,
                   help="dir containing <cfg>/rollout/rollout_tokens.npz")
    p.add_argument("--survival_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rel_lo", type=int, default=-500)
    p.add_argument("--rel_hi", type=int, default=100)
    add_wandb_args(p)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_diag_style()

    print(f"Loading training tokens from {args.train_tokens}")
    td = np.load(args.train_tokens)
    train_idx = td["indices_flat"]                     # (F, P)
    scales = td["scales"].tolist()
    first_trainable = int(td["first_trainable_scale"])
    effective_vocab = int(td["effective_vocab_size"])
    print(f"  {train_idx.shape[0]} frames, {train_idx.shape[1]} pos, "
          f"V={effective_vocab}, scales={scales}, "
          f"first_trainable={first_trainable}")

    surv = load_survival(args.survival_json)
    n_frames = int(surv["n_frames"])
    cfgs = sorted(surv["configs"].keys())

    # Sanity: the train tokens must come from the same VQ-VAE as the
    # rollout, or per-position vocabularies are nonsense. We check the
    # first available rollout's effective_vocab_size.
    sample_rollout_path = None
    for cfg in cfgs:
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
    pos_scale, trainable_mask = build_scale_ids(scales, first_trainable)
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
    n_trainable = int(trainable_mask.sum())
    print(f"  trainable positions: {n_trainable}/{P}")

    # ---------- per-cfg loop ----------
    cfg_data = {}
    for cfg in cfgs:
        rpath = os.path.join(args.logits_root, cfg, "rollout",
                             "rollout_tokens.npz")
        if not os.path.isfile(rpath):
            print(f"[skip] missing {rpath}")
            continue
        rd = load_rollout_tokens(os.path.dirname(rpath))
        idx = rd["rollout_indices"]                    # (N, T+1, P)
        N = idx.shape[0]
        # The IC frame (t=0) is not generated; skip it so traces have
        # length T = n_steps. That matches diagnostics.npz.
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

        # also: was ANY trainable position OOD this frame?
        any_ood_per_frame = ood_train.any(axis=2)       # (N, T)

        et, collapsed, n = get_explosion_times(surv, cfg, N)
        frame_ood = frame_ood[:n]
        per_scale_ood = per_scale_ood[:n]
        any_ood_per_frame = any_ood_per_frame[:n]
        N = n

        cfg_data[cfg] = dict(
            frame_ood=frame_ood, per_scale_ood=per_scale_ood,
            any_ood=any_ood_per_frame, et=et, N=N,
            collapsed=collapsed,
        )
        # quick textual readout
        first_ood_t = np.full(N, T, dtype=np.int64)
        for j in range(N):
            ts = np.where(any_ood_per_frame[j])[0]
            if ts.size:
                first_ood_t[j] = int(ts[0])
        med_first_ood = int(np.median(first_ood_t))
        gap_to_explode = et - first_ood_t
        cfg_data[cfg]["mean_frame_ood"] = float(frame_ood.mean())
        cfg_data[cfg]["median_first_ood_t"] = med_first_ood
        cfg_data[cfg]["median_gap_to_explode"] = int(
            np.median(gap_to_explode))
        print(f"\n[{cfg}]  N={N}, mean frame OOD rate = "
              f"{frame_ood.mean():.4f}, "
              f"median first-OOD t={med_first_ood}, "
              f"median (t_explode - first_ood) = "
              f"{int(np.median(gap_to_explode))}")

    if not cfg_data:
        raise SystemExit("No rollout_tokens.npz found in any cfg.")

    metas = {cfg: load_cfg_meta(os.path.join(args.logits_root, cfg))
             for cfg in cfg_data}
    styles = assign_cfg_styles(metas)
    rel_axis = np.arange(args.rel_lo, args.rel_hi)
    floor = ood_floor(n_trainable)

    # With the per-position mask active in the rollout, every OOD rate is
    # exactly zero by construction — this run is then a sanity check, not
    # a diagnostic, and gets a single PASS panel per cfg.
    all_zero = all(is_effectively_zero(cd["frame_ood"])
                   for cd in cfg_data.values())

    figs = {}

    # ---------- per-cfg figures ----------
    for cfg, cd in cfg_data.items():
        N = cd["N"]
        T = cd["frame_ood"].shape[1]
        et = cd["et"]
        coll = cd["collapsed"]
        n_coll = int(coll.sum())
        info = surv["configs"][cfg]
        desc = describe_decode(metas[cfg])
        title = (f"{cfg}{f'  [{desc}]' if desc else ''}  "
                 f"position-OOD   S_end={info['survival_at_end']:.0%}, "
                 f"med t_explode={info['median_t']}, N={N}")

        if is_effectively_zero(cd["frame_ood"]):
            fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
            ax.text(0.5, 0.5,
                    "position-OOD rate identically 0\n"
                    "(per-position mask active — sanity PASS)",
                    ha="center", va="center", fontsize=14, color="C2",
                    transform=ax.transAxes)
            ax.set_axis_off()
            fig.suptitle(title)
            out_path = os.path.join(args.output_dir, f"cfg_{cfg}.png")
            fig.savefig(out_path)
            figs[f"cfg_{cfg}"] = fig
            print(f"saved {out_path} (sanity PASS)")
            continue

        # Rows: overall frame OOD rate + per-scale rates with real signal.
        # "Real" = mean above the log-display floor; a handful of isolated
        # emissions across 2000 frames would otherwise keep a dead row.
        rows = [("frame OOD rate\n(trainable pos)", cd["frame_ood"])]
        for si in range(first_trainable, len(scales)):
            arr = cd["per_scale_ood"][..., si]
            if np.nanmean(arr) <= floor:
                continue
            rows.append((f"OOD scale {si}\n({scales[si]}×{scales[si]})",
                         arr))

        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, 2,
                                 figsize=(14, 2.6 * n_rows),
                                 sharex="col", constrained_layout=True)
        axes = np.atleast_2d(axes)

        for r, (label, arr) in enumerate(rows):
            arr = np.maximum(arr, floor)   # log-scale display floor
            ax_abs, ax_rel = axes[r, 0], axes[r, 1]
            band_plot(ax_abs, np.arange(T), arr[~coll], color="C2",
                      label="survived")
            band_plot(ax_abs, np.arange(T), arr[coll], color="C3",
                      label="collapsed")
            ax_abs.set_yscale("log")
            ax_abs.set_ylabel(label)
            if r == 0:
                ax_abs.legend(loc="best")

            if n_coll == 0:
                ax_rel.text(0.5, 0.5, "no collapse",
                            transform=ax_rel.transAxes,
                            ha="center", va="center", color="gray")
            else:
                aligned = aligned_window(arr, et, args.rel_lo, args.rel_hi)
                band_plot(ax_rel, rel_axis, aligned[coll], color="C3",
                          label="collapsed")
                ax_rel.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
                ax_rel.set_yscale("log")
                if r == 0:
                    ax_rel.legend(loc="best")

        axes[0, 0].set_title("absolute time (median + IQR, log scale)")
        axes[0, 1].set_title(f"aligned to explosion   "
                             f"(τ ∈ [{args.rel_lo}, {args.rel_hi}))")
        axes[-1, 0].set_xlabel("absolute rollout step t")
        axes[-1, 1].set_xlabel("τ = t - t_explode")
        fig.suptitle(title)
        out_path = os.path.join(args.output_dir, f"cfg_{cfg}.png")
        fig.savefig(out_path)
        figs[f"cfg_{cfg}"] = fig
        print(f"saved {out_path}")

    # ---------- cross-cfg overlay ----------
    overlay_fig = None
    overlay_series = {}
    if not all_zero:
        cfg_order = sorted(
            cfg_data.keys(),
            key=lambda c: -surv["configs"][c]["survival_at_end"],
        )
        # Overlay rows: frame OOD + per-scale rows with real signal in at
        # least one cfg (mean above the log-display floor).
        scale_rows = [
            si for si in range(first_trainable, len(scales))
            if any(np.nanmean(cd["per_scale_ood"][..., si]) > floor
                   for cd in cfg_data.values())
        ]
        n_rows = 1 + len(scale_rows)
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(12, 2.6 * n_rows), sharex=True,
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)

        def overlay_row(ax, get_arr, label):
            ydict = {}
            for cfg in cfg_order:
                cd = cfg_data[cfg]
                coll = cd["collapsed"]
                if coll.sum() == 0:
                    continue
                aligned = aligned_window(
                    np.maximum(get_arr(cd), floor), cd["et"],
                    args.rel_lo, args.rel_hi)
                med = safe_median(aligned[coll], axis=0)
                desc = describe_decode(metas[cfg])
                ax.plot(rel_axis, med, lw=1.8,
                        color=styles[cfg]["color"], ls=styles[cfg]["ls"],
                        label=f"{cfg}{f' [{desc}]' if desc else ''}")
                ydict[cfg] = med
            ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
            ax.set_yscale("log")
            ax.set_ylabel(label)
            return ydict

        ydict = overlay_row(axes[0], lambda cd: cd["frame_ood"],
                            "frame OOD rate")
        if ydict:
            overlay_series["overlay/frame_ood"] = (
                rel_axis, ydict, "tau = t - t_explode",
                "Median frame OOD rate vs tau")
        outside_legend(axes[0])
        for k, si in enumerate(scale_rows):
            overlay_row(axes[1 + k],
                        lambda cd, si=si: cd["per_scale_ood"][..., si],
                        f"OOD scale {si}\n({scales[si]}×{scales[si]})")

        axes[-1].set_xlabel("τ = t - t_explode")
        fig.suptitle(
            "Cross-cfg medians: position-OOD rate of collapsed trajectories")
        out_path = os.path.join(args.output_dir, "overlay_relative.png")
        fig.savefig(out_path)
        overlay_fig = fig
        figs["overlay_relative"] = fig
        print(f"saved {out_path}")
    else:
        print("All cfgs have OOD rate ≡ 0 (position mask active) — "
              "overlay skipped.")

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

    # ---------- wandb ----------
    run = init_wandb(args, job_type="position_ood", config={
        "train_tokens": args.train_tokens,
        "logits_root": args.logits_root,
        "rel_lo": args.rel_lo,
        "rel_hi": args.rel_hi,
        "n_cfgs": len(cfg_data),
        "n_trainable_positions": n_trainable,
    })
    scalars = {"position_mask_ood_zero": bool(all_zero)}
    for cfg, cd in cfg_data.items():
        scalars[f"ood/{cfg}/mean_frame_ood"] = cd["mean_frame_ood"]
        scalars[f"ood/{cfg}/median_first_ood_t"] = cd["median_first_ood_t"]
        scalars[f"ood/{cfg}/median_gap_to_explode"] = (
            cd["median_gap_to_explode"])
    wandb_log_figs_and_scalars(run, scalars=scalars, figs=figs,
                               line_series=overlay_series)


if __name__ == "__main__":
    main()
