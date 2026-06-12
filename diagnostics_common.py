"""diagnostics_common.py — shared utilities for the collapse-diagnostics
pipeline (multitraj_survival.py, analyze_logits.py, analyze_logits_aligned.py,
analyze_position_ood.py).

Pure NumPy + matplotlib (+ optional wandb). No JAX imports, so the CPU-only
analyzers stay importable on nodes without a GPU runtime.

Layout contract shared by all stages (one sweep root per experiment):
    <sweep_root>/<cfg>/rollout/rollout_tokens.npz   (+ rollout_logits.npz,
                                                       cfg_meta.json)
    <sweep_root>/<cfg>/logits/diagnostics.npz
    <sweep_root>/survival/survival.json
"""
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# Layout / loading
# =============================================================================


def discover_cfgs(sweep_root):
    """Sorted cfg names under sweep_root that have rollout/rollout_tokens.npz.

    No assumption about cfg naming — a cfg is any directory with a completed
    rollout in it.
    """
    cfgs = []
    for name in sorted(os.listdir(sweep_root)):
        if os.path.isfile(os.path.join(sweep_root, name, "rollout",
                                       "rollout_tokens.npz")):
            cfgs.append(name)
    return cfgs


def load_rollout_tokens(rollout_dir):
    """Load rollout_tokens.npz and normalize index arrays to (N, T+1, P).

    Legacy N=1 rollouts saved rank-2 arrays; insert the trajectory axis so
    every consumer can assume rank 3.
    """
    d = np.load(os.path.join(rollout_dir, "rollout_tokens.npz"),
                allow_pickle=True)
    out = {k: d[k] for k in d.files}
    for key in ("rollout_indices", "gt_indices"):
        if key in out and out[key].ndim == 2:
            out[key] = out[key][None]
    return out


def load_survival(survival_json):
    """Load survival.json; derive 'survival_at_end' per cfg.

    Older runs only recorded survival_at_2000 (hardcoded to the n_steps of
    that sweep). survival_at_end = survived / n_trajectories is step-count
    agnostic; consumers should prefer it.
    """
    with open(survival_json) as f:
        surv = json.load(f)
    for cfg, info in surv.get("configs", {}).items():
        if "survival_at_end" not in info:
            info["survival_at_end"] = info["survived"] / info["n_trajectories"]
    return surv


def get_explosion_times(surv, cfg, n_expected):
    """Per-trajectory explosion times for cfg, reconciled against the
    trajectory count of the array being aligned.

    Returns (explosion_t (n,), collapsed (n,) bool, n) with
    n = min(n_expected, len(explosion_t)); warns on mismatch (happens when
    a rollout was re-run with a different --n_trajectories than the
    survival pass).
    """
    et = np.asarray(surv["configs"][cfg]["explosion_t"], dtype=np.int64)
    n_frames = int(surv["n_frames"])
    if et.shape[0] != n_expected:
        print(f"[warn] {cfg}: survival N={et.shape[0]} != data N={n_expected}"
              f"; truncating to min")
        et = et[:n_expected]
    collapsed = et < n_frames
    return et, collapsed, et.shape[0]


def _parse_legacy_cfg_name(name):
    """Best-effort decode params from legacy cfg names (T07, T10_topp95,
    T10_topk50) and new-style names (T0p7-pm)."""
    meta = {}
    m = re.match(r"^T(\d+)p(\d+)", name)
    if m is None:
        m = re.match(r"^T(\d)(\d)", name)
    if m is not None:
        meta["temperature"] = float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"topp(\d+)", name)
    if m is not None:
        meta["top_p"] = int(m.group(1)) / 100.0
    m = re.search(r"topk(\d+)", name)
    if m is not None:
        meta["top_k"] = int(m.group(1))
    if re.search(r"(^|[-_])pm($|[-_])", name):
        meta["position_mask_used"] = True
    return meta


def load_cfg_meta(cfg_dir):
    """Decode-parameter metadata for one cfg directory.

    Reads cfg_meta.json (written by rollout_nsp.py) from <cfg_dir>/rollout/
    or <cfg_dir>/; falls back to parsing the directory name for legacy
    sweeps. Always returns a dict; 'temperature' may be absent for
    unparseable legacy names.
    """
    meta = _parse_legacy_cfg_name(os.path.basename(cfg_dir.rstrip("/")))
    for cand in (os.path.join(cfg_dir, "rollout", "cfg_meta.json"),
                 os.path.join(cfg_dir, "cfg_meta.json")):
        if os.path.isfile(cand):
            with open(cand) as f:
                meta.update(json.load(f))
            break
    return meta


def describe_decode(meta):
    """Short human label for a cfg's decode params, e.g. 'T=0.9 top_p=0.95 +pm'."""
    parts = []
    if meta.get("temperature") is not None:
        parts.append(f"T={meta['temperature']:g}")
    if meta.get("top_k"):
        parts.append(f"top_k={meta['top_k']}")
    if meta.get("top_p") and meta["top_p"] < 1.0:
        parts.append(f"top_p={meta['top_p']:g}")
    if meta.get("position_mask_used"):
        parts.append("+pm")
    return " ".join(parts)


# =============================================================================
# Scale bookkeeping / alignment
# =============================================================================


def build_scale_ids(scales, first_trainable_scale):
    """Map flat token position -> scale index.

    Returns (scale_ids (P,) int, trainable (P,) bool).
    """
    scale_ids = []
    for k, s in enumerate(scales):
        scale_ids.extend([k] * (int(s) * int(s)))
    scale_ids = np.array(scale_ids, dtype=np.int64)
    trainable = scale_ids >= int(first_trainable_scale)
    return scale_ids, trainable


def aligned_window(traces, explosion_t, lo, hi):
    """Re-index (N, T) absolute-time traces to tau = t - t_explode.

    Returns (N, hi-lo) float32; out-of-range slots are NaN and survived
    trajectories (t_explode >= T) are all-NaN rows.
    """
    traces = np.asarray(traces)
    N, T = traces.shape
    rel = np.arange(lo, hi)
    out = np.full((N, rel.size), np.nan, dtype=np.float32)
    for j in range(N):
        te = int(explosion_t[j])
        if te >= T:
            continue
        t_abs = te + rel
        valid = (t_abs >= 0) & (t_abs < T)
        out[j, valid] = traces[j, t_abs[valid]]
    return out


def safe_median(arr, axis):
    with np.errstate(all="ignore"):
        return np.nanmedian(arr, axis=axis)


# =============================================================================
# Smoothing / plotting
# =============================================================================


def ema(x, span, axis=-1):
    """NaN-aware exponential moving average along `axis`.

    NaN inputs hold the running mean (carried forward, not reset), so
    aligned windows with NaN borders smooth cleanly.
    """
    x = np.asarray(x, dtype=np.float32)
    if span is None or span <= 1 or x.shape[axis] < 2:
        return x
    x = np.moveaxis(x, axis, -1)
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x)
    m = x[..., 0].copy()
    out[..., 0] = m
    for t in range(1, x.shape[-1]):
        v = x[..., t]
        have_v = ~np.isnan(v)
        seed = have_v & np.isnan(m)
        m = np.where(seed, v, m)
        upd = have_v & ~seed
        m = np.where(upd, (1.0 - alpha) * m + alpha * v, m)
        out[..., t] = m
    return np.moveaxis(out, -1, axis)


def is_effectively_zero(Y, tol=1e-9):
    """True when a panel's data is identically ~0 (or all-NaN) — used to
    auto-skip dead panels instead of plotting empty axes."""
    Y = np.asarray(Y)
    if Y.size == 0 or np.all(np.isnan(Y)):
        return True
    with np.errstate(all="ignore"):
        return bool(np.nanmax(np.abs(Y)) <= tol)


def band_plot(ax, x, Y, color, label=None, individual_max=8, smooth=0,
              ls="-", lw=1.8):
    """Median + IQR band for a (N, T) trace bundle — the replacement for
    all-trajectory spaghetti.

    Individual traces are drawn (thin, low alpha) only when N <= individual_max.
    Returns the median trace (or None when Y is empty/all-NaN) so callers
    can reuse it for overlays.
    """
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float32))
    if Y.size == 0 or np.all(np.isnan(Y)):
        return None
    if smooth and smooth > 1:
        Y = ema(Y, smooth, axis=-1)
    with np.errstate(all="ignore"):
        med = np.nanmedian(Y, axis=0)
        q25 = np.nanpercentile(Y, 25, axis=0)
        q75 = np.nanpercentile(Y, 75, axis=0)
    ax.fill_between(x, q25, q75, color=color, alpha=0.18, lw=0)
    if Y.shape[0] <= individual_max:
        for row in Y:
            ax.plot(x, row, color=color, lw=0.5, alpha=0.3, ls=ls)
    n = int(np.any(~np.isnan(Y), axis=1).sum())
    ax.plot(x, med, color=color, lw=lw, ls=ls,
            label=None if label is None else f"{label} (n={n})")
    return med


def explosion_rug(ax, explosion_t, n_frames, color="C3"):
    """Mark explosion times as a rug along the bottom of the axes instead
    of N overlapping vlines."""
    et = np.asarray(explosion_t)
    et = et[et < n_frames]
    if et.size == 0:
        return
    ax.plot(et, np.zeros_like(et, dtype=np.float32), "|",
            color=color, ms=10, mew=1.2, alpha=0.8,
            transform=ax.get_xaxis_transform(), clip_on=False)


def temp_color(temperature, vmin=0.6, vmax=1.6):
    """Fixed temperature -> color mapping (coolwarm over [vmin, vmax]) so
    color means the same thing in every figure of every stage."""
    if temperature is None:
        return None
    t = (float(temperature) - vmin) / (vmax - vmin)
    return plt.cm.coolwarm(float(np.clip(t, 0.0, 1.0)))


_LINESTYLES = ["-", "--", "-.", ":"]


def assign_cfg_styles(metas):
    """Consistent {color, ls} per cfg across all figures.

    metas: {cfg: meta-dict from load_cfg_meta}. Color encodes temperature
    via temp_color; cfgs sharing a temperature (e.g. top_p variants) get
    distinct linestyles. Cfgs with unknown temperature fall back to a
    viridis ramp.
    """
    cfgs = sorted(metas.keys())
    unknown = [c for c in cfgs if metas[c].get("temperature") is None]
    fallback = plt.cm.viridis(np.linspace(0, 0.9, max(1, len(unknown))))
    styles = {}
    seen_per_temp = {}
    for cfg in cfgs:
        t = metas[cfg].get("temperature")
        if t is None:
            color = fallback[unknown.index(cfg)]
        else:
            color = temp_color(t)
        k = seen_per_temp.get(t, 0)
        seen_per_temp[t] = k + 1
        styles[cfg] = {"color": color, "ls": _LINESTYLES[k % len(_LINESTYLES)]}
    return styles


def set_diag_style():
    """Shared rcParams for all diagnostics figures. Pair with
    constrained_layout at figure creation; never tight_layout /
    bbox_inches='tight' (breaks shared-colorbar layouts)."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 130,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


def outside_legend(ax, **kwargs):
    """Legend outside the figure's right edge, with constrained_layout
    making room for it (matplotlib >= 3.7 'outside' loc). Falls back to an
    axes legend anchored past the right spine on older matplotlib."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    fig = ax.get_figure()
    try:
        return fig.legend(handles, labels, loc="outside right upper",
                          **kwargs)
    except (ValueError, TypeError):
        return ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                         borderaxespad=0.0, **kwargs)


# =============================================================================
# Wandb
# =============================================================================


def add_wandb_args(parser):
    parser.add_argument("--wandb_project", type=str,
                        default="gust2-diagnostics-bridges")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_dir", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")


def init_wandb(args, job_type, config):
    """wandb.init with the shared diagnostics conventions; returns the run
    or None (wandb missing or --no_wandb)."""
    if getattr(args, "no_wandb", False):
        print("wandb: disabled (--no_wandb)")
        return None
    if not WANDB_AVAILABLE:
        print("wandb: not installed; skipping logging")
        return None
    if args.wandb_dir is not None:
        os.makedirs(args.wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = args.wandb_dir
    kwargs = dict(project=args.wandb_project, name=args.wandb_name,
                  job_type=job_type, config=config)
    if args.wandb_group is not None:
        kwargs["group"] = args.wandb_group
    return wandb.init(**kwargs)


def _downsample(x, y, max_points=400):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    stride = max(1, int(np.ceil(x.size / max_points)))
    return x[::stride], y[::stride]


def wandb_log_figs_and_scalars(run, scalars=None, figs=None,
                               line_series=None, finish=True):
    """Log a diagnostics stage's outputs in one call.

    scalars: {key: value}
    figs:    {key: matplotlib Figure} — logged as wandb.Image
    line_series: {key: (x, {series_name: y}, xname, title)} — logged as
                 wandb.plot.line_series, downsampled to ~400 points with
                 NaN -> None so the JSON payload stays valid.
    """
    if run is None:
        return
    log = dict(scalars or {})
    for k, fig in (figs or {}).items():
        log[k] = wandb.Image(fig)
    for k, (x, ydict, xname, title) in (line_series or {}).items():
        xs, keys, ys = None, [], []
        for name, y in ydict.items():
            xd, yd = _downsample(x, y)
            xs = xd
            keys.append(name)
            ys.append([None if np.isnan(v) else float(v) for v in yd])
        if xs is None or not ys:
            continue
        log[k] = wandb.plot.line_series(
            xs=xs.tolist(), ys=ys, keys=keys, title=title, xname=xname)
    run.log(log)
    if finish:
        run.finish()
