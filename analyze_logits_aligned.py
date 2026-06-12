"""analyze_logits_aligned.py — re-do logit diagnostics aligned to the
actual explosion times from multitraj_survival.py.

Reads each cfg's diagnostics.npz (numeric traces over absolute t) plus a
sweep-level survival.json (true explosion times per trajectory), and
produces:

  per-cfg figures — cfg_<NAME>.png
    rows: (1) top-1 prob,  (2) frame entropy,  (3) frac outside top-K
          (auto-dropped when identically zero), (4+) per-scale entropy —
          coarse trainable scales merged into one row, the two finest
          scales kept individually
    cols: (a) absolute t — survived/collapsed median+IQR bands,
          (b) relative τ = t - t_explode (collapsed trajs only)

  cross-cfg overlay — overlay_relative.png
    median trace across collapsed trajs vs τ for each cfg, one axes per
    metric, colored by temperature (consistent across all diagnostics
    figures), so we can see whether the precursor shape is universal
    across temperatures / truncation strategies.

Run:
  python analyze_logits_aligned.py \\
    --logits_root <sweep_root> \\
    --survival_json <sweep_root>/survival/survival.json \\
    --output_dir <sweep_root>/logits_aligned
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
    describe_decode,
    ema,
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


def load_cfg(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def build_rows(d):
    """Row specs for one cfg's diagnostics dict.

    Returns a list of (label, (N, T) array, ylim) rows: the scalar frame
    metrics (frac-outside dropped when identically zero), then per-scale
    entropy with coarse trainable scales merged and the two finest scales
    individual. Rows with no signal (all-NaN or constant) are dropped.
    """
    scales = np.asarray(d["scales"]).tolist()
    n_scales = len(scales)
    first_trainable = int(d["first_trainable_scale"])

    rows = [
        ("top-1 prob", d["frame_top1_prob"], (0, 1)),
        ("entropy (nats)", d["frame_entropy"], None),
    ]
    if not is_effectively_zero(d["frac_outside_topk"], tol=1e-6):
        rows.append(("frac outside top-K", d["frac_outside_topk"], None))

    per_scale_ent = d["per_scale_entropy"]                    # (N, T, S)
    trainable = list(range(first_trainable, n_scales))
    fine = trainable[-2:]
    coarse = [s for s in trainable if s not in fine]
    if coarse:
        labels = ",".join(f"{scales[s]}×{scales[s]}" for s in coarse)
        merged = np.nanmean(per_scale_ent[..., coarse], axis=-1)
        rows.append((f"H coarse ({labels})", merged, None))
    for s in fine:
        rows.append((f"H {scales[s]}×{scales[s]}",
                     per_scale_ent[..., s], None))

    # Drop rows that carry no signal (all-NaN or constant).
    kept = []
    for label, arr, ylim in rows:
        if np.all(np.isnan(arr)):
            continue
        with np.errstate(all="ignore"):
            if np.nanmax(arr) - np.nanmin(arr) <= 1e-9:
                continue
        kept.append((label, arr, ylim))
    return kept


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logits_root", required=True,
                   help="dir containing <cfg>/logits/diagnostics.npz")
    p.add_argument("--survival_json", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--rel_lo", type=int, default=-500,
                   help="τ window start (frames before explosion)")
    p.add_argument("--rel_hi", type=int, default=100,
                   help="τ window end (frames after explosion)")
    p.add_argument("--ema", type=int, default=10,
                   help="EMA window for plotted traces (0 disables)")
    add_wandb_args(p)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_diag_style()

    surv = load_survival(args.survival_json)
    n_frames = int(surv["n_frames"])

    cfgs = sorted(surv["configs"].keys())
    cfg_data = {}
    for cfg in cfgs:
        npz_path = os.path.join(args.logits_root, cfg, "logits",
                                "diagnostics.npz")
        if not os.path.isfile(npz_path):
            print(f"[skip] missing {npz_path}")
            continue
        d = load_cfg(npz_path)
        N_npz = d["frame_top1_prob"].shape[0]
        et, collapsed, n = get_explosion_times(surv, cfg, N_npz)
        cfg_data[cfg] = {"d": d, "et": et, "collapsed": collapsed, "n": n}

    if not cfg_data:
        raise SystemExit("No diagnostics.npz found for any cfg.")
    print(f"Loaded {len(cfg_data)} configs.")

    metas = {cfg: load_cfg_meta(os.path.join(args.logits_root, cfg))
             for cfg in cfg_data}
    styles = assign_cfg_styles(metas)
    rel_axis = np.arange(args.rel_lo, args.rel_hi)

    # overlay[row_label][cfg] = median collapsed trace vs τ
    overlay = {}
    overlay_order = []
    fig_paths = {}

    # ---------- per-cfg figures ----------
    for cfg, cd in cfg_data.items():
        d = cd["d"]
        et = cd["et"][:cd["n"]]
        collapsed = cd["collapsed"]
        n_coll = int(collapsed.sum())
        N = cd["n"]
        T = d["frame_top1_prob"].shape[1]
        ts = np.arange(T)
        info = surv["configs"][cfg]
        desc = describe_decode(metas[cfg])
        title = (f"{cfg}{f'  [{desc}]' if desc else ''}   "
                 f"S_end={info['survival_at_end']:.0%}, "
                 f"med t_explode={info['median_t']}, N={N}")

        rows = build_rows(d)
        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, 2,
                                 figsize=(14, 2.6 * n_rows),
                                 sharex="col", constrained_layout=True)
        axes = np.atleast_2d(axes)

        for r, (label, arr, ylim) in enumerate(rows):
            arr = arr[:N]
            ax_abs = axes[r, 0]
            band_plot(ax_abs, ts, arr[~collapsed], color="C2",
                      label="survived", smooth=args.ema)
            band_plot(ax_abs, ts, arr[collapsed], color="C3",
                      label="collapsed", smooth=args.ema)
            ax_abs.set_ylabel(label)
            if ylim:
                ax_abs.set_ylim(*ylim)
            if r == 0:
                ax_abs.legend(loc="best")

            ax_rel = axes[r, 1]
            if n_coll == 0:
                ax_rel.text(0.5, 0.5, "no collapse",
                            transform=ax_rel.transAxes,
                            ha="center", va="center", color="gray")
            else:
                aligned = aligned_window(arr, et, args.rel_lo, args.rel_hi)
                med = band_plot(ax_rel, rel_axis, aligned[collapsed],
                                color="C3", label="collapsed",
                                smooth=args.ema)
                ax_rel.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
                if r == 0:
                    ax_rel.legend(loc="best")
                overlay.setdefault(label, {})[cfg] = med
                if label not in overlay_order:
                    overlay_order.append(label)
            if ylim:
                ax_rel.set_ylim(*ylim)

        axes[-1, 0].set_xlabel("absolute rollout step t")
        axes[-1, 1].set_xlabel("τ = t - t_explode")
        axes[0, 0].set_title("absolute time (median + IQR)")
        axes[0, 1].set_title(f"aligned to explosion   "
                             f"(τ ∈ [{args.rel_lo}, {args.rel_hi}))")
        fig.suptitle(title)
        out_path = os.path.join(args.output_dir, f"cfg_{cfg}.png")
        fig.savefig(out_path)
        fig_paths[f"cfg_{cfg}"] = fig
        print(f"saved {out_path}")

    # ---------- cross-cfg overlay ----------
    overlay_fig = None
    if overlay:
        n_rows = len(overlay_order)
        fig, axes = plt.subplots(n_rows, 1,
                                 figsize=(12, 2.6 * n_rows),
                                 sharex=True, constrained_layout=True)
        axes = np.atleast_1d(axes)
        cfg_order = sorted(
            cfg_data.keys(),
            key=lambda c: -surv["configs"][c]["survival_at_end"],
        )
        for r, label in enumerate(overlay_order):
            ax = axes[r]
            for cfg in cfg_order:
                if cfg not in overlay[label] or overlay[label][cfg] is None:
                    continue
                med = overlay[label][cfg]
                if args.ema > 1:
                    med = ema(med, args.ema)
                desc = describe_decode(metas[cfg])
                ax.plot(rel_axis, med, lw=1.8,
                        color=styles[cfg]["color"], ls=styles[cfg]["ls"],
                        label=f"{cfg}{f' [{desc}]' if desc else ''}")
            ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
            ax.set_ylabel(label)
            if r == 0:
                outside_legend(ax)
        axes[-1].set_xlabel("τ = t - t_explode")
        fig.suptitle(
            "Cross-cfg medians of collapsed trajectories aligned to explosion")
        out_path = os.path.join(args.output_dir, "overlay_relative.png")
        fig.savefig(out_path)
        overlay_fig = fig
        print(f"saved {out_path}")
    else:
        print("No collapsed trajectories in any cfg — overlay skipped.")

    # ---------- wandb ----------
    run = init_wandb(args, job_type="logits_aligned", config={
        "logits_root": args.logits_root,
        "survival_json": args.survival_json,
        "rel_lo": args.rel_lo,
        "rel_hi": args.rel_hi,
        "n_cfgs": len(cfg_data),
    })
    scalars = {"n_cfgs": len(cfg_data)}
    for cfg, cd in cfg_data.items():
        scalars[f"aligned/{cfg}/n_collapsed"] = int(cd["collapsed"].sum())
    figs = dict(fig_paths)
    if overlay_fig is not None:
        figs["overlay_relative"] = overlay_fig
    series = {}
    for label in overlay_order:
        ydict = {cfg: med for cfg, med in overlay[label].items()
                 if med is not None}
        if ydict:
            key = label.split(" (")[0].replace(" ", "_").replace("×", "x")
            series[f"overlay/{key}"] = (
                rel_axis, ydict, "tau = t - t_explode",
                f"Median {label} vs tau")
    wandb_log_figs_and_scalars(run, scalars=scalars, figs=figs,
                               line_series=series)


if __name__ == "__main__":
    main()
