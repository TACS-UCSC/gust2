"""analyze_calibration_gate.py — GO/NO-GO for the calibration-temperature sampler.

Reads the per-cell calibration JSONs written by measure_calibration_temp.py and
decides, OFFLINE, whether the per-scale calibration temperature schedule T_k
tracks the swept-temperature yardstick well enough to justify the (multi-hour)
rollout sweep. This is the gate the whole cheap pass exists to answer.

The bet's DIRECTION (CE rises with token count while H stays flat -> warmer on
harder configs) is near-certain; the MAGNITUDE is the coin-flip, set by the
model's fine-scale heat capacity. This script reads the magnitude off the real
solved T_k and applies the go/no-go bands:

  config   target T_eff band     hard FAIL
  sc341    0.9 - 1.2  (cold)     > 1.3   (over-warms the cold config, like etpool)
  sc917    1.5 - 2.0  (warm)     < 1.3 or > 2.5
  sc1941   1.6 - 2.2  (warmest)  < 1.4 (under-warm, like etmodel) or > 2.5 (over-diffuse)

  cross-config: finest-scale CE monotone sc341 < sc917 < sc1941, and
                T_eff(sc1941) > T_eff(sc917).

GO only if every present config is in-band (or at worst WARN, not FAIL),
monotone holds, and sc1941 > sc917. Any hard FAIL => NO-GO (report the negative
result; do not spend rollout compute). The yardstick best_T (1.00/1.72/1.85) is
shown alongside for the "does it reproduce the sweep" read.

Usage:
    python analyze_calibration_gate.py --gate_dir <dir with *-calib.json>
    python analyze_calibration_gate.py --gate_dir ... --yardstick plots/scaling_tempopt_n128/scaling_tempopt.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics

# Per-config gate bands: (lo_ok, hi_ok, fail_lo, fail_hi). None = no hard bound.
# Calibrated to the measured per-cell swept best_T spread (scaling_tempopt.csv):
#   sc341 best_T 0.8-1.2 (cold-optimal; real risk is OVER-warming like etpool),
#   sc917 1.6-2.0, sc1941 1.6-2.2 (arch spread ~0.4 = 2 grid steps).
BANDS = {
    "sc341":  (0.8, 1.25, None, 1.35),   # lenient cold (safe), strict warm
    "sc917":  (1.5, 2.10, 1.30, 2.60),
    "sc1941": (1.6, 2.20, 1.40, 2.60),
}
PROX_TOL = 0.5    # |T_eff - cell best_T| above this downgrades a PASS to WARN
SC_ORDER = ["sc341", "sc917", "sc1941"]

CELL_RE = re.compile(r"(small|medium|large)-(sc\d+)-(?:nsp-)?(s\d+)")


def classify(sc, T, best_T=None):
    band = BANDS.get(sc)
    if band is None:
        return "??", "no band defined"
    lo, hi, flo, fhi = band
    if flo is not None and T < flo:
        return "FAIL", f"under-warm (< {flo}): under-warm collapse risk"
    if fhi is not None and T > fhi:
        return "FAIL", f"over-warm (> {fhi}): over-diffusive risk"
    if lo <= T <= hi:
        if best_T is not None and abs(T - best_T) > PROX_TOL:
            return "WARN", f"in band but |T-best_T|={abs(T-best_T):.2f} > {PROX_TOL}"
        return "PASS", "in band"
    return "WARN", f"outside [{lo},{hi}] but not a hard fail"


def load_yardstick(path):
    """(size, sc) -> {arch: best_T}."""
    table = {}
    if not path or not os.path.exists(path):
        return table
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                bt = float(row["best_T"])
            except (KeyError, ValueError):
                continue
            table.setdefault((row["size"], row["sc"]), {})[row["arch"]] = bt
    return table


def best_T_for(yard, size, sc, arch):
    archs = yard.get((size, sc), {})
    if not archs:
        return None, "n/a"
    if arch in archs:
        return archs[arch], arch
    med = statistics.median(archs.values())
    return med, f"median/{len(archs)}arch"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate_dir", required=True,
                    help="Directory of *-calib.json from measure_calibration_temp.py")
    ap.add_argument("--yardstick", default="plots/scaling_tempopt_n128/scaling_tempopt.csv",
                    help="Swept-T yardstick CSV for the best_T comparison.")
    ap.add_argument("--gate_stat", default="token_weighted",
                    choices=["token_weighted", "finest"],
                    help="Which effective-T to gate on (default token-weighted).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.gate_dir, "*-calib.json")))
    if not files:
        raise SystemExit(f"No *-calib.json in {args.gate_dir}")
    yard = load_yardstick(args.yardstick)

    # cells[size][sc] = dict(arch, T_eff, finest_T, finest_ce, T_k, scales, best_T, best_src)
    cells = {}
    for path in files:
        with open(path) as f:
            a = json.load(f)
        m = CELL_RE.search(a.get("checkpoint_dir", "") or os.path.basename(path))
        if not m:
            print(f"[warn] cannot parse cell from {path}; skipping")
            continue
        size, sc, arch = m.group(1), m.group(2), m.group(3)
        agg = a.get("aggregate", {})
        T_eff = (agg.get("T_eff_token_weighted") if args.gate_stat == "token_weighted"
                 else agg.get("finest_scale_T"))
        bt, bsrc = best_T_for(yard, size, sc, arch)
        cells.setdefault(size, {})[sc] = {
            "arch": arch,
            "T_eff": T_eff,
            "finest_T": agg.get("finest_scale_T"),
            "finest_ce": agg.get("finest_scale_ce_nats"),
            "best_T": bt, "best_src": bsrc,
            "per_scale": a.get("per_scale", {}),
        }

    overall_fail = False
    overall_incomplete = False
    print(f"\nGate statistic: T_eff = {args.gate_stat}")
    for size in sorted(cells):
        print(f"\n=== VQ size: {size} ===")
        print(f"  {'config':>7} {'arch':>5} {'finestCE':>9} {'finestT':>8} "
              f"{'T_eff':>7} {'best_T':>7} {'verdict':>6}  note")
        row_by_sc = cells[size]
        for sc in SC_ORDER:
            if sc not in row_by_sc:
                print(f"  {sc:>7} {'--':>5} {'--':>9} {'--':>8} {'--':>7} "
                      f"{'--':>7} {'MISS':>6}  (no gate JSON)")
                overall_incomplete = True
                continue
            r = row_by_sc[sc]
            T = r["T_eff"]
            verdict, note = classify(sc, T, r["best_T"]) if T is not None else ("ERR", "no T_eff")
            if verdict == "FAIL":
                overall_fail = True
            bt = r["best_T"]
            print(f"  {sc:>7} {r['arch']:>5} "
                  f"{(r['finest_ce'] if r['finest_ce'] is not None else float('nan')):>9.3f} "
                  f"{(r['finest_T'] if r['finest_T'] is not None else float('nan')):>8.3f} "
                  f"{(T if T is not None else float('nan')):>7.3f} "
                  f"{(bt if bt is not None else float('nan')):>7.3f} "
                  f"{verdict:>6}  {note}"
                  + (f" [best_T={r['best_src']}]" if bt is not None else ""))

        # Cross-config checks (need all three).
        present = [sc for sc in SC_ORDER if sc in row_by_sc and row_by_sc[sc]["T_eff"] is not None]
        if len(present) == 3:
            ces = [row_by_sc[sc]["finest_ce"] for sc in SC_ORDER]
            Ts = [row_by_sc[sc]["T_eff"] for sc in SC_ORDER]
            mono_ce = (ces[0] is not None and ces[1] is not None and ces[2] is not None
                       and ces[0] < ces[1] < ces[2])
            warm_order = Ts[2] > Ts[1]
            print(f"  cross-config: finest-CE monotone(sc341<sc917<sc1941)={mono_ce}; "
                  f"T_eff(sc1941)>T_eff(sc917)={warm_order}")
            if not mono_ce:
                print("    [!] finest-scale CE does NOT rise monotonically — the "
                      "config-adaptive warmth signal is broken.")
                overall_fail = True
            if not warm_order:
                print("    [!] sc1941 does NOT warm past sc917 — the warmest config "
                      "under-warms (etmodel failure mode).")
                overall_fail = True
        else:
            print(f"  cross-config: INCOMPLETE (have {present}); rerun the gate with all three configs.")
            overall_incomplete = True

    print("\n================= VERDICT =================")
    if overall_fail:
        print("  NO-GO. At least one config hard-fails or the warmth ordering is broken.")
        print("  Do NOT spend rollout compute. This is a clean NEGATIVE RESULT: the")
        print("  strongest single-step calibration anchor (cross-entropy / data entropy")
        print("  rate) does not reproduce the required rollout warmth -> the drift wall")
        print("  is a horizon property absent from ANY teacher-forced single-step")
        print("  statistic. Report it; consider the alpha*CE / CE-floor fallback only if")
        print("  the ORDERING was right and only the MAGNITUDE was off.")
    elif overall_incomplete:
        print("  INCOMPLETE. No hard fail yet, but not all configs are present.")
        print("  Rerun the gate over small (or medium) x {sc341, sc917, sc1941}.")
    else:
        print("  GO. The calibration schedule tracks the yardstick within tolerance.")
        print("  Recommended: run ONE confirmation rollout on the historically worst")
        print("  cell first (small-sc1941-s139, etmodel was 73x floor there) with")
        print("  --sampler per_scale_temp; require its EMD to beat 2x its swept best-T")
        print("  EMD (judge by EMD/PDF/per-position OOD, NEVER TKE RSE) before the full")
        print("  per_scale_temp arm across the grid.")
    print("===========================================")


if __name__ == "__main__":
    main()
