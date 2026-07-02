"""plot_prediction_vs_outcome.py — F6.2, the M4 money plot.

Predicted temperature (a data-read) vs swept best-T (the outcome), one point
per (size, sc, arch) cell, 45-degree line, each cell's on-manifold T interval
as a vertical bar. If the prediction lands inside every cell's bar, the sweep
was never needed for SAFETY; distance to the 45-degree line is the regret
story (F6.5's job, annotated here as a median).

PROVENANCE (caption-honesty; do not blur these lines):
  * Predicted T = per-cell zero-crossing of the signed per-scale entropy drift
    dH_k(T) = H_roll,k(T) - H_data,k averaged over the FINEST-2 scales (the
    collapse-driving scales the fine-heat schedule targets). This is exactly
    solve_calib_schedule.py's per-scale inversion, sourced from the drift-sweep
    summaries (gust2-drift-*). The DATA side (H_data) is a training-set
    climate statistic (window-invariance lemma, F6.3); the CURVE side
    (H_roll(T)) is a model response measured on rollouts. The solve never
    sees EMD/spectra — it is NOT fit to the outcome it is compared against.
  * Swept best-T = EMD-argmin per cell from the N=128 sweep
    (plots/scaling_tempopt_n128/scaling_tempopt.csv).
  * On-manifold interval = swept T's with EMD <= 1.0 (the off-manifold
    explosion bar used in plot_climate_temp_band.py; judged-by-EMD-band rule —
    this bar flags explosion only, mode-B needs spectra/PDF and is why the
    interval is a SAFETY band, not an optimality band).
  * The hardcoded "a-priori T" constants floating around older scripts
    (1.0/1.7/1.8[5]) are swept-yardstick medians — deliberately NOT used here.

Inversion statuses (solve_calib_schedule semantics):
  ok           — dH crosses 0 inside the swept range; T* interpolated.
  clamped_lo   — dH > 0 already at the coldest swept T (rollout more diverse
                 than data): T* < T_min; plotted at T_min with a down-arrow.
  saturated_hi — dH < 0 at the hottest swept T (still under-diverse):
                 T* > T_max; plotted at T_max with an up-arrow.

Data sources (all local / wandb):
  plots/prediction_vs_outcome/drift_dH_cache.json   (cached wandb fetch;
      delete to refetch from gust2-drift-{small,medium,large})
  plots/scaling_tempopt_n128/scaling_tempopt.csv
  plots/scaling_tempopt_n128/n128_table.json

Outputs:
  plots/prediction_vs_outcome/prediction_vs_outcome.png
  plots/prediction_vs_outcome/prediction_vs_outcome.csv
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "plots/prediction_vs_outcome"
CACHE = os.path.join(OUT_DIR, "drift_dH_cache.json")
TEMPOPT_CSV = "plots/scaling_tempopt_n128/scaling_tempopt.csv"
N128_TABLE = "plots/scaling_tempopt_n128/n128_table.json"
EXPLODED = 1.0                     # off-manifold EMD bar (plot_climate_temp_band)
ENTITY = "bigpseud-ucsc"

SC_COLOR = {"sc341": "tab:green", "sc917": "tab:orange", "sc1941": "tab:purple"}
SIZE_MARKER = {"small": "o", "medium": "s", "large": "^"}


def fetch_drift_dH():
    """cell -> {scale: {T: [dH runs...]}} from the gust2-drift-* projects."""
    if os.path.exists(CACHE):
        raw = json.load(open(CACHE))
        return {c: {int(s): {float(t): v for t, v in d.items()}
                    for s, d in sd.items()} for c, sd in raw.items()}
    import wandb
    api = wandb.Api()
    out = defaultdict(dict)
    for size in ["small", "medium", "large"]:
        for r in api.runs(f"{ENTITY}/gust2-drift-{size}"):
            sc, T, arch = r.config.get("sc"), r.config.get("T"), r.config.get("arch")
            if sc is None or T is None:
                continue
            cell = f"{size}|{sc}|{arch}"
            for k, v in r.summary.items():
                if k.startswith("drift/dH/s"):
                    scale = int(k.split("s")[-1])
                    out[cell].setdefault(scale, {}).setdefault(
                        float(T), []).append(float(v))
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({c: {str(s): {str(t): vs for t, vs in d.items()}
                   for s, d in sd.items()} for c, sd in out.items()},
              open(CACHE, "w"))
    return {c: {s: {t: v for t, v in d.items()} for s, d in sd.items()}
            for c, sd in out.items()}


def predict_T(scale_dH):
    """Zero-crossing of mean dH over the finest-2 scales.

    Returns (T_pred, status). Linear interpolation between the bracketing
    swept temperatures; edge statuses per solve_calib_schedule.
    """
    scales = sorted(scale_dH)
    fine2 = scales[-2:]
    Ts = sorted(set.intersection(*[set(scale_dH[s]) for s in fine2]))
    dH = np.array([np.mean([np.mean(scale_dH[s][t]) for s in fine2])
                   for t in Ts])
    Ts = np.array(Ts)
    if dH[0] > 0:
        return float(Ts[0]), "clamped_lo"
    if dH[-1] < 0:
        return float(Ts[-1]), "saturated_hi"
    i = int(np.where(np.diff(np.sign(dH)) > 0)[0][0])   # first - -> + crossing
    frac = -dH[i] / (dH[i + 1] - dH[i])
    return float(Ts[i] + frac * (Ts[i + 1] - Ts[i])), "ok"


def main():
    drift = fetch_drift_dH()

    best_T = {}
    with open(TEMPOPT_CSV) as f:
        for row in csv.DictReader(f):
            best_T[f"{row['size']}|{row['sc']}|{row['arch']}"] = float(row["best_T"])

    table = {k: {float(t): v for t, v in d.items()}
             for k, d in json.load(open(N128_TABLE)).items()}

    rows = []
    for cell in sorted(best_T):
        if cell not in drift:
            print(f"[warn] no drift data for {cell} — skipped")
            continue
        T_pred, status = predict_T(drift[cell])
        emds = table[cell]
        safe = sorted(t for t, v in emds.items() if v["emd"] <= EXPLODED)
        size, sc, arch = cell.split("|")
        rows.append({
            "size": size, "sc": sc, "arch": arch,
            "best_T": best_T[cell], "T_pred": T_pred, "status": status,
            "safe_lo": safe[0] if safe else np.nan,
            "safe_hi": safe[-1] if safe else np.nan,
            "pred_in_safe": bool(safe) and (safe[0] <= T_pred <= safe[-1]),
        })

    # --- figure ---
    fig, ax = plt.subplots(figsize=(7.5, 7))
    t_all = ([r["best_T"] for r in rows] + [r["T_pred"] for r in rows]
             + [r["safe_lo"] for r in rows if np.isfinite(r["safe_lo"])]
             + [r["safe_hi"] for r in rows if np.isfinite(r["safe_hi"])])
    lims = (min(t_all) - 0.25, max(t_all) + 0.25)
    ax.plot(lims, lims, color="0.4", lw=1, ls="--", zorder=1,
            label="perfect prediction (45°)")
    rng = np.random.default_rng(0)      # small x-jitter: best_T is grid-valued
    for r in rows:
        x = r["best_T"] + rng.uniform(-0.012, 0.012)
        color = SC_COLOR[r["sc"]]
        marker = SIZE_MARKER[r["size"]]
        if np.isfinite(r["safe_lo"]):
            ax.plot([x, x], [r["safe_lo"], r["safe_hi"]], color=color,
                    lw=2.5, alpha=0.25, zorder=2,
                    solid_capstyle="butt")
        ax.scatter([x], [r["T_pred"]], color=color, marker=marker, s=48,
                   edgecolor="k", linewidth=0.5, zorder=4)
        if r["status"] == "saturated_hi":
            ax.annotate("", xy=(x, r["T_pred"] + 0.10),
                        xytext=(x, r["T_pred"] + 0.015),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
                        zorder=5)
        elif r["status"] == "clamped_lo":
            ax.annotate("", xy=(x, r["T_pred"] - 0.10),
                        xytext=(x, r["T_pred"] - 0.015),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.4),
                        zorder=5)
    handles = [plt.Line2D([], [], color="0.4", lw=1, ls="--",
                          label="perfect prediction (45°)")]
    handles += [plt.Line2D([], [], color=c, marker="o", ls="", ms=7,
                           markeredgecolor="k", markeredgewidth=0.5, label=sc)
                for sc, c in SC_COLOR.items()]
    handles += [plt.Line2D([], [], color="0.5", marker=m, ls="", ms=7,
                           markeredgecolor="k", markeredgewidth=0.5, label=size)
                for size, m in SIZE_MARKER.items()]

    n_ok = sum(r["pred_in_safe"] for r in rows)
    abs_err = [abs(r["T_pred"] - r["best_T"]) for r in rows if r["status"] == "ok"]
    n_sat = sum(r["status"] != "ok" for r in rows)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("swept best-T (EMD-argmin, N=128 sweep)", fontsize=11)
    ax.set_ylabel("predicted T (entropy-drift zero-crossing, finest-2 scales)",
                  fontsize=11)
    ax.set_title(
        "Data-read T prediction vs swept optimum, per cell\n"
        f"in on-manifold band: {n_ok}/{len(rows)} cells · median "
        f"|T_pred − best_T| = {np.median(abs_err):.2f} ({len(abs_err)} bracketed)\n"
        f"{n_sat} cells at drift-curve edge (arrows) — resolve via a drift "
        "pass over the newest rollouts",
        fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    png = os.path.join(OUT_DIR, "prediction_vs_outcome.png")
    fig.savefig(png, dpi=140)
    print(f"saved {png}")

    csv_path = os.path.join(OUT_DIR, "prediction_vs_outcome.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {csv_path}")

    print(f"\n{'cell':32s} {'best_T':>6} {'T_pred':>7} {'status':>13} "
          f"{'safe band':>12} {'in-band'}")
    for r in rows:
        print(f"{r['size']+'-'+r['sc']+'-'+r['arch']:32s} {r['best_T']:>6.2f} "
              f"{r['T_pred']:>7.2f} {r['status']:>13} "
              f"[{r['safe_lo']:.1f},{r['safe_hi']:.1f}] {r['pred_in_safe']}")


if __name__ == "__main__":
    main()
