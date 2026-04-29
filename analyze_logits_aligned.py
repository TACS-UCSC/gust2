"""analyze_logits_aligned.py — re-do logit diagnostics aligned to the
actual explosion times from multitraj_survival.py.

Reads each cfg's diagnostics.npz (numeric traces over absolute t) plus a
sweep-level survival.json (true explosion times per trajectory), and
produces:

  per-cfg figures — cfg_<NAME>.png
    rows: (1) top-1 prob,  (2) frame entropy,  (3) frac outside top-K,
          (4-8) per-scale entropy (one row per scale)
    cols: (a) absolute t with explosion markers,
          (b) relative τ = t - t_explode (only collapsed trajs)
    colored by collapse status (red=collapsed, green=survived)

  cross-cfg overlay — overlay_relative.png
    median trace across collapsed trajs vs τ for each cfg, on a single
    axes per metric, so we can see whether the precursor shape is
    universal across temperatures / truncation strategies.

Run:
  python analyze_logits_aligned.py \\
    --logits_root plots/sc341-multitraj/logits \\
    --survival_json plots/sc341-multitraj/survival/survival.json \\
    --output_dir plots/sc341-multitraj/logits_aligned
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_cfg(npz_path):
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def aligned_window(traces, explosion_t, lo, hi):
    """traces: (N, T) absolute. Returns (N, hi-lo) aligned at τ = t-t_e.
    Slots with t<0 or t>=T are NaN. Survived trajectories are dropped."""
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


def safe_median(arr, axis):
    with np.errstate(all="ignore"):
        return np.nanmedian(arr, axis=axis)


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
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.survival_json) as f:
        surv = json.load(f)
    n_frames = int(surv["n_frames"])
    threshold_emd = float(surv["threshold_emd"])

    cfgs = sorted(surv["configs"].keys())
    cfg_data = {}
    for cfg in cfgs:
        npz_path = os.path.join(args.logits_root, cfg, "logits",
                                "diagnostics.npz")
        if not os.path.isfile(npz_path):
            print(f"[skip] missing {npz_path}")
            continue
        d = load_cfg(npz_path)
        et = np.array(surv["configs"][cfg]["explosion_t"])
        N_npz = d["frame_top1_prob"].shape[0]
        if et.shape[0] != N_npz:
            print(f"[warn] {cfg}: survival N={et.shape[0]} != diag N={N_npz}")
            et = et[:N_npz]
        cfg_data[cfg] = {"d": d, "et": et}

    print(f"Loaded {len(cfg_data)} configs.")
    n_scales = int(next(iter(cfg_data.values()))["d"]["scales"].shape[0])
    first_trainable = int(
        next(iter(cfg_data.values()))["d"]["first_trainable_scale"])
    scales = np.array(
        next(iter(cfg_data.values()))["d"]["scales"]).tolist()
    rel_axis = np.arange(args.rel_lo, args.rel_hi)

    metric_specs = [
        ("frame_top1_prob",
         "top-1 prob (mean over trainable tokens)", (0, 1)),
        ("frame_entropy",
         "top-K entropy (nats)", None),
        ("frac_outside_topk",
         "frac sampled outside top-K", None),
    ]

    overlay = {key: {} for key, _, _ in metric_specs}
    overlay_per_scale = {s: {} for s in range(first_trainable, n_scales)}

    # ---------- per-cfg figures ----------
    for cfg, cd in cfg_data.items():
        d = cd["d"]
        et = cd["et"]
        N = d["frame_top1_prob"].shape[0]
        T = d["frame_top1_prob"].shape[1]
        collapsed = et < n_frames
        n_coll = int(collapsed.sum())
        cfg_info = surv["configs"][cfg]
        title = (f"{cfg}  (S∞={cfg_info['survival_at_2000']:.0%}, "
                 f"med={cfg_info['median_t']}, N={N})")

        n_rows = 3 + (n_scales - first_trainable)
        fig, axes = plt.subplots(n_rows, 2,
                                 figsize=(13, 2.0 * n_rows),
                                 sharex="col")

        # ---- rows 0-2: scalar per-frame metrics ----
        for r, (key, ylabel, ylim) in enumerate(metric_specs):
            arr = d[key]                               # (N, T)
            ax_abs = axes[r, 0]
            for j in range(N):
                color = "C3" if collapsed[j] else "C2"
                alpha = 0.45 if collapsed[j] else 0.25
                ax_abs.plot(np.arange(T), arr[j], lw=0.5,
                            alpha=alpha, color=color)
                if collapsed[j]:
                    ax_abs.axvline(et[j], color="C3", lw=0.3,
                                   alpha=0.25)
            ax_abs.set_ylabel(ylabel, fontsize=9)
            if ylim:
                ax_abs.set_ylim(*ylim)
            ax_abs.grid(True, alpha=0.3)

            ax_rel = axes[r, 1]
            if n_coll == 0:
                ax_rel.text(0.5, 0.5, "no collapse",
                            transform=ax_rel.transAxes,
                            ha="center", va="center", color="gray")
            else:
                aligned = aligned_window(arr, et, args.rel_lo, args.rel_hi)
                for j in np.where(collapsed)[0]:
                    ax_rel.plot(rel_axis, aligned[j], lw=0.4,
                                alpha=0.35, color="C3")
                med = safe_median(aligned[collapsed], axis=0)
                q25 = np.nanpercentile(aligned[collapsed], 25, axis=0)
                q75 = np.nanpercentile(aligned[collapsed], 75, axis=0)
                ax_rel.fill_between(rel_axis, q25, q75,
                                    color="C0", alpha=0.2)
                ax_rel.plot(rel_axis, med, color="C0", lw=2,
                            label=f"median (n={n_coll})")
                ax_rel.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
                overlay[key][cfg] = med
                ax_rel.legend(loc="best", fontsize=8)
            if ylim:
                ax_rel.set_ylim(*ylim)
            ax_rel.grid(True, alpha=0.3)

        # ---- per-scale entropy rows ----
        per_scale_ent = d["per_scale_entropy"]            # (N, T, S)
        for s_idx in range(first_trainable, n_scales):
            row = 3 + (s_idx - first_trainable)
            arr = per_scale_ent[..., s_idx]               # (N, T)
            scale_label = f"scale {s_idx} ({scales[s_idx]}×{scales[s_idx]})"
            ax_abs = axes[row, 0]
            for j in range(N):
                color = "C3" if collapsed[j] else "C2"
                alpha = 0.45 if collapsed[j] else 0.25
                ax_abs.plot(np.arange(T), arr[j], lw=0.5,
                            alpha=alpha, color=color)
            ax_abs.set_ylabel(f"H {scale_label}", fontsize=9)
            ax_abs.grid(True, alpha=0.3)

            ax_rel = axes[row, 1]
            if n_coll == 0:
                ax_rel.text(0.5, 0.5, "no collapse",
                            transform=ax_rel.transAxes,
                            ha="center", va="center", color="gray")
            else:
                aligned = aligned_window(arr, et, args.rel_lo, args.rel_hi)
                for j in np.where(collapsed)[0]:
                    ax_rel.plot(rel_axis, aligned[j], lw=0.4,
                                alpha=0.35, color="C3")
                med = safe_median(aligned[collapsed], axis=0)
                q25 = np.nanpercentile(aligned[collapsed], 25, axis=0)
                q75 = np.nanpercentile(aligned[collapsed], 75, axis=0)
                ax_rel.fill_between(rel_axis, q25, q75,
                                    color="C0", alpha=0.2)
                ax_rel.plot(rel_axis, med, color="C0", lw=2)
                ax_rel.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
                overlay_per_scale[s_idx][cfg] = med
            ax_rel.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("absolute rollout step t")
        axes[-1, 1].set_xlabel("τ = t - t_explode")
        axes[0, 0].set_title("absolute time   "
                             "(red=collapsed, green=survived)")
        axes[0, 1].set_title(f"aligned to explosion   "
                             f"(τ ∈ [{args.rel_lo}, {args.rel_hi}))")
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        out_path = os.path.join(args.output_dir, f"cfg_{cfg}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"saved {out_path}")

    # ---------- cross-cfg overlay ----------
    n_metric_rows = len(metric_specs) + (n_scales - first_trainable)
    fig, axes = plt.subplots(n_metric_rows, 1,
                             figsize=(11, 2.4 * n_metric_rows),
                             sharex=True)

    cfg_order = sorted(
        cfg_data.keys(),
        key=lambda c: -surv["configs"][c]["survival_at_2000"],
    )
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(cfg_order)))

    for r, (key, ylabel, ylim) in enumerate(metric_specs):
        ax = axes[r]
        for ci, cfg in enumerate(cfg_order):
            if cfg not in overlay[key]:
                continue
            ax.plot(rel_axis, overlay[key][cfg], color=cmap[ci],
                    lw=1.6, label=cfg)
        ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        if r == 0:
            ax.legend(loc="best", fontsize=8, ncol=2)

    for s_idx in range(first_trainable, n_scales):
        row = len(metric_specs) + (s_idx - first_trainable)
        ax = axes[row]
        for ci, cfg in enumerate(cfg_order):
            if cfg not in overlay_per_scale[s_idx]:
                continue
            ax.plot(rel_axis, overlay_per_scale[s_idx][cfg],
                    color=cmap[ci], lw=1.6, label=cfg)
        ax.axvline(0, color="k", ls="--", lw=0.6, alpha=0.6)
        ax.set_ylabel(f"H scale {s_idx}\n({scales[s_idx]}×{scales[s_idx]})",
                      fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("τ = t - t_explode")
    fig.suptitle(
        "Cross-cfg medians of collapsed trajectories aligned to explosion",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path = os.path.join(args.output_dir, "overlay_relative.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
