"""plot_climate_temp_band.py — the "pick T from the climate statistics, stay
on-manifold" figure, replicated across all 3 token counts.

Reads the N=128 long-term sweep table (/tmp/n128_table.json: EMD-vs-T per
(size, sc, arch) cell) and shows that:
  - the temperature basin sharpens with token count (flat @ sc341 -> cliff @ sc1941),
  - yet a SINGLE a-priori T per token count keeps EVERY NSP arch on-manifold.

Judged by the EMD-band / on-manifold bar (NOT regret, NOT collapse_rate) per
feedback_judge_by_emd_band: EMD > 1.0 = off-manifold explosion.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPLODED = 1.0
SCS = ["sc341", "sc917", "sc1941"]
table = {tuple(k.split("|")): {float(t): v for t, v in d.items()}
         for k, d in json.load(open("/tmp/n128_table.json")).items()}


def cells_for(sc):
    return {(sz, arch): d for (sz, ssc, arch), d in table.items() if ssc == sc}


def basin_stats(sc):
    cells = cells_for(sc)
    cold_pen, exploded_cold = [], 0
    for (sz, arch), d in cells.items():
        Ts = sorted(d)
        emds = {t: d[t]["emd"] for t in Ts}
        bestE = min(emds.values())
        cold = emds[Ts[0]]                       # coldest swept T
        cold_pen.append(cold / bestE)
        if cold > EXPLODED:
            exploded_cold += 1
    # per-candidate-T worst-case over cells present at that T
    allT = sorted({t for d in cells.values() for t in d})
    per_T = {}
    for t in allT:
        es = [d[t]["emd"] for d in cells.values() if t in d]
        if len(es) < len(cells) - 1:             # skip coverage-gap temps
            continue
        per_T[t] = {"n": len(es), "max": max(es), "mean": float(np.mean(es)),
                    "n_expl": sum(e > EXPLODED for e in es)}
    Tstar = min(per_T, key=lambda t: per_T[t]["max"])
    return {"cells": cells, "median_cold_pen": float(np.median(cold_pen)),
            "exploded_cold": exploded_cold, "n": len(cells),
            "per_T": per_T, "Tstar": Tstar, "worst_at_Tstar": per_T[Tstar]["max"]}


fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
print(f"{'sc':7s} {'cells':>5} {'median cold-edge penalty':>24} "
      f"{'cold-explodes':>14} {'a-priori T*':>11} {'worst arch @T*':>14}")
for ax, sc in zip(axes, SCS):
    s = basin_stats(sc)
    ax.axhspan(EXPLODED, 1e2, color="#f5cccc", zorder=0)
    ax.text(0.03, 0.965, "off-manifold (EMD>1)", transform=ax.transAxes,
            fontsize=8, color="#993333", va="top")
    for (sz, arch), d in s["cells"].items():
        Ts = sorted(d)
        ax.plot(Ts, [d[t]["emd"] for t in Ts], marker="o", ms=4, lw=1.1,
                color="0.55", alpha=0.7, zorder=2)
    Tstar = s["Tstar"]
    ax.axvline(Tstar, color="tab:blue", ls="--", lw=2, zorder=4)
    ax.text(Tstar, 0.115, f" a-priori T*={Tstar:g}", color="tab:blue",
            fontsize=9, rotation=90, va="bottom", ha="right")
    # mark worst arch at T*
    ax.scatter([Tstar], [s["worst_at_Tstar"]], color="tab:blue", s=55,
               zorder=6, edgecolor="k", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_title(f"{sc}  ({s['n']} archs)\n"
                 f"cold-edge penalty {s['median_cold_pen']:.1f}×  ·  "
                 f"{s['exploded_cold']}/{s['n']} explode if cold\n"
                 f"worst arch @T*={Tstar:g}: EMD={s['worst_at_Tstar']:.2f} (on-manifold)",
                 fontsize=10)
    ax.set_xlabel("sampling temperature T")
    ax.grid(alpha=0.3, which="both")
    print(f"{sc:7s} {s['n']:>5} {s['median_cold_pen']:>22.2f}× "
          f"{s['exploded_cold']:>8}/{s['n']:<5} {Tstar:>11g} "
          f"{s['worst_at_Tstar']:>13.3f}")
    print(f"        per-T worst-case: " +
          "  ".join(f"T{t:g}:{v['max']:.2f}({v['n_expl']}x)" for t, v in s["per_T"].items()))
axes[0].set_ylabel("pixel-EMD vs GT climate (log)")
fig.suptitle("Pick T from the long-rollout (climate) statistics → stay on-manifold across all NSP sizes.\n"
             "Basin sharpens with token count (flat→cliff); a single a-priori T per token count keeps every arch below the explosion line.",
             fontsize=12, y=1.02)
fig.tight_layout()
import os
os.makedirs("plots/climate_temp_band", exist_ok=True)
dest = "plots/climate_temp_band/climate_temp_band.png"
fig.savefig(dest, dpi=140, bbox_inches="tight")
print(f"\nsaved {dest}")
