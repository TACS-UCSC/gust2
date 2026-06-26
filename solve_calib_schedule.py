"""solve_calib_schedule.py — invert the rollout-marginal curve onto the data target.

The horizon-marginal calibration solve (paper section B). Given the DATA target
marginals `H_data` (from measure_tokenizer_entropy.py) and a CURVE of rollout
marginals `H_roll(T)` measured at a few global temperatures (from
measure_rollout_marginal.py on the reused global-T rollouts), solve the FIXED
temperature schedule whose rollout marginal matches the data:

    per-scale arm:  T_k*  solves  H_roll,k(T_k)   = H_data,k     (pooled marginal)
    per-cell arm:   T_ij* solves  H_roll,ij(T_ij) = H_data,ij    (per-position)

This is the calibration gate (measure_calibration_temp.py) done on the RIGHT
statistic: the inversion machinery is identical (monotone entropy-vs-T, np.interp,
clamped_lo / saturated_hi statuses), but the curve is the HORIZON emitted-token
marginal of the free-running rollout, not the single-step teacher-forced
conditional — so the solved temperature is free to exceed 1 (the single-step
gate provably could not). H_roll(T) is monotone increasing in T.

Status semantics (drive the bracketing protocol — the swept-T curve only spans
warm T in [1.4, 2.2] for sc1941):
  * ok           — target bracketed; T* interpolated.
  * clamped_lo   — data marginal BELOW H_roll at the coldest swept T => T*<T_min
                   needed (the rollout is already MORE diverse than the data
                   here; cool it). Run supplementary cooler rollouts (e.g. 1.0,1.2).
  * saturated_hi — data marginal ABOVE H_roll at the hottest swept T => T*>T_max
                   needed. Run supplementary warmer rollouts (e.g. 2.5, 3.0).
  * flat_fallback— the curve is ~flat in T at this scale/cell (ill-conditioned
                   inversion, e.g. near-degenerate coarse scale) => T*=1.0.

CAVEAT: matching the marginal ENTROPY is the knob, not the success metric —
judge the resulting rollout on EMD/PDF/OOD/spectra vs the swept-T yardstick.

Usage:
    python solve_calib_schedule.py \
        --data inference_anchors/small-sc1941-data.json \
        --point 1.4 marg/T1p4.json --point 1.6 marg/T1p6.json ... \
        --out_per_scale anchors/sc1941-calib-perscale.json \
        --out_per_cell  anchors/sc1941-calib-percell.json
    python solve_calib_schedule.py --selftest
"""

import argparse
import json

import numpy as np

FLAT_EPS = 0.02      # nats; curve range below this => ill-conditioned => T=1 fallback


def invert_curve(target: float, temps: np.ndarray, H_curve: np.ndarray,
                 flat_eps: float = FLAT_EPS):
    """Invert a monotone H_roll(T) curve at `target`. temps ascending.

    Returns (T_star, status). Mirrors measure_calibration_temp.solve_Tk plus a
    flat-curve guard (risk 5).
    """
    temps = np.asarray(temps, dtype=np.float64)
    Hg = np.maximum.accumulate(np.asarray(H_curve, dtype=np.float64))  # de-noise monotone
    if Hg[-1] - Hg[0] < flat_eps:
        return 1.0, "flat_fallback"
    if target <= Hg[0]:
        return float(temps[0]), "clamped_lo"
    if target >= Hg[-1]:
        return float(temps[-1]), "saturated_hi"
    return float(np.interp(target, Hg, temps)), "ok"


def _load_curve(points):
    """points: list of [T_str, json_path]. Returns (temps_sorted, marginals_sorted)."""
    pairs = [(float(t), json.load(open(p))) for t, p in points]
    pairs.sort(key=lambda x: x[0])
    return np.array([t for t, _ in pairs]), [m for _, m in pairs]


def _check_consistency(data, marginals):
    ds = [int(s) for s in data["scales"]]
    for m in marginals:
        assert [int(s) for s in m["scales"]] == ds, "scales mismatch data vs curve"
        assert int(m["effective_vocab_size"]) == int(data["effective_vocab_size"]), \
            "effective_vocab_size mismatch"


def solve_per_scale(data, temps, marginals, target_key):
    """T_k matched to the per-scale data marginal (`target_key`). Returns the
    anchor dict consumable by load_anchor_array / --sampler per_scale_temp."""
    scales = [int(s) for s in data["scales"]]
    trainable = list(data["trainable_scale_indices"])
    per_scale, Tk_list, statuses = {}, [], []
    for idx in trainable:
        s = scales[idx]
        H_curve = [m["per_scale"][str(s)][target_key] for m in marginals]
        H_data = data["per_scale"][str(s)][target_key]
        T_star, status = invert_curve(H_data, temps, H_curve)
        Tk_list.append(T_star)
        statuses.append(status)
        per_scale[str(s)] = {
            "H_data_nats": float(H_data),
            "H_roll_curve_nats": [float(h) for h in H_curve],
            "T_k": float(T_star),
            "solve_status": status,
        }
    return {
        "source": "horizon_marginal_calib_per_scale",
        "per_scale_target": target_key,
        "bias_correct": data.get("bias_correct", False),
        "scales": scales,
        "first_trainable_scale": int(data["first_trainable_scale"]),
        "trainable_scale_indices": trainable,
        "curve_temps": [float(t) for t in temps],
        "per_scale": per_scale,
        # both standard slots hold T_k -> per_scale_temp resolves either stat.
        "per_trainable_pooled_marginal_nats": [float(x) for x in Tk_list],
        "per_trainable_mean_per_position_nats": [float(x) for x in Tk_list],
    }, statuses


def solve_per_cell(data, temps, marginals):
    """T_ij matched to the per-cell data marginal (`per_position_entropies`).
    Returns (anchor dict with percell_temperature length P, per-scale summary)."""
    scales = [int(s) for s in data["scales"]]
    first_trainable = int(data["first_trainable_scale"])
    cps = [s * s for s in scales]
    bnd = np.concatenate([[0], np.cumsum(cps)]).astype(int)
    P = int(bnd[-1])
    tau = np.ones(P, dtype=np.float64)               # seed/non-trainable positions = 1.0
    summary = {}
    for idx in range(first_trainable, len(scales)):
        s = scales[idx]
        a, b = bnd[idx], bnd[idx + 1]
        # (Npos, G) per-cell curve
        Hcell = np.stack([np.asarray(m["per_scale"][str(s)]["per_position_entropies"],
                                     dtype=np.float64) for m in marginals], axis=1)
        Hdata = np.asarray(data["per_scale"][str(s)]["per_position_entropies"],
                           dtype=np.float64)
        assert Hcell.shape[0] == (b - a) == Hdata.shape[0], f"cell count mismatch scale {s}"
        st_counts = {"ok": 0, "clamped_lo": 0, "saturated_hi": 0, "flat_fallback": 0}
        for c in range(b - a):
            T_star, status = invert_curve(Hdata[c], temps, Hcell[c])
            tau[a + c] = T_star
            st_counts[status] += 1
        block = tau[a:b]
        summary[str(s)] = {
            "T_min": float(block.min()), "T_median": float(np.median(block)),
            "T_max": float(block.max()), "n_cells": int(b - a), **st_counts,
        }
    return {
        "source": "horizon_marginal_calib_per_cell",
        "bias_correct": data.get("bias_correct", False),
        "scales": scales,
        "first_trainable_scale": first_trainable,
        "effective_vocab_size": int(data["effective_vocab_size"]),
        "tokens_per_frame": P,
        "curve_temps": [float(t) for t in temps],
        "percell_temperature": [float(x) for x in tau],
        "per_scale_summary": summary,
    }, summary


def _supplementary_hint(statuses):
    lo = sum(1 for s in statuses if s == "clamped_lo")
    hi = sum(1 for s in statuses if s == "saturated_hi")
    msgs = []
    if lo:
        msgs.append(f"{lo} clamped_lo -> run supplementary COOLER global-T rollouts "
                    f"(e.g. 1.0, 1.2) and re-solve")
    if hi:
        msgs.append(f"{hi} saturated_hi -> run supplementary WARMER global-T rollouts "
                    f"(e.g. 2.5, 3.0) and re-solve")
    return msgs


def _selftest():
    # invert_curve: linear curve H=[1,2,3] at temps [1,2,3].
    temps = np.array([1.0, 2.0, 3.0])
    assert invert_curve(2.5, temps, [1.0, 2.0, 3.0]) == (2.5, "ok")
    assert invert_curve(0.5, temps, [1.0, 2.0, 3.0]) == (1.0, "clamped_lo")
    assert invert_curve(3.5, temps, [1.0, 2.0, 3.0]) == (3.0, "saturated_hi")
    assert invert_curve(2.0, temps, [2.0, 2.0, 2.001])[1] == "flat_fallback"
    # non-monotone noise is de-noised by maximum.accumulate.
    T, st = invert_curve(2.5, temps, [1.0, 2.6, 2.4])
    assert st == "ok" and 1.0 < T < 2.0, (T, st)

    # end-to-end with files: 1 trainable scale (s=2, 4 cells), 2 temps.
    data = {
        "scales": [1, 2], "first_trainable_scale": 1, "trainable_scale_indices": [1],
        "effective_vocab_size": 6, "bias_correct": False,
        "per_scale": {
            "1": {"pooled_marginal_nats": 0.0, "mean_per_position_nats": 0.0,
                  "per_position_entropies": [0.0]},
            "2": {"pooled_marginal_nats": 1.5, "mean_per_position_nats": 1.4,
                  "per_position_entropies": [1.0, 1.4, 1.8, 2.2]},
        },
    }
    m_lo = {"scales": [1, 2], "effective_vocab_size": 6,
            "per_scale": {"1": {"pooled_marginal_nats": 0.0, "mean_per_position_nats": 0.0,
                                "per_position_entropies": [0.0]},
                          "2": {"pooled_marginal_nats": 1.0, "mean_per_position_nats": 0.9,
                                "per_position_entropies": [0.8, 1.0, 1.2, 1.4]}}}
    m_hi = {"scales": [1, 2], "effective_vocab_size": 6,
            "per_scale": {"1": {"pooled_marginal_nats": 0.0, "mean_per_position_nats": 0.0,
                                "per_position_entropies": [0.0]},
                          "2": {"pooled_marginal_nats": 2.0, "mean_per_position_nats": 1.9,
                                "per_position_entropies": [1.6, 2.0, 2.4, 2.8]}}}
    json.dump(data, open("/tmp/_sc_data.json", "w"))
    json.dump(m_lo, open("/tmp/_sc_lo.json", "w"))
    json.dump(m_hi, open("/tmp/_sc_hi.json", "w"))
    temps2, marg = _load_curve([["1.4", "/tmp/_sc_lo.json"], ["2.2", "/tmp/_sc_hi.json"]])
    _check_consistency(data, marg)
    ps, st = solve_per_scale(data, temps2, marg, "pooled_marginal_nats")
    # pooled target 1.5 between 1.0@1.4 and 2.0@2.2 -> T ~ 1.8
    assert st == ["ok"] and abs(ps["per_trainable_pooled_marginal_nats"][0] - 1.8) < 1e-6
    pc, summ = solve_per_cell(data, temps2, marg)
    assert pc["tokens_per_frame"] == 5
    assert pc["percell_temperature"][0] == 1.0          # seed scale untouched
    # cell 0: data 1.0 between 0.8@1.4 and 1.6@2.2 -> T=1.4 + 0.8*0.2... interp
    tau_c0 = pc["percell_temperature"][1]
    assert 1.4 < tau_c0 < 2.2, tau_c0
    print("[ok] solve_calib_schedule self-test passed "
          "(invert ok/clamped/saturated/flat, per-scale + per-cell end-to-end)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data-target JSON (measure_tokenizer_entropy.py)")
    ap.add_argument("--point", nargs=2, action="append", metavar=("T", "JSON"),
                    help="a curve point: global temperature + its "
                         "measure_rollout_marginal.py JSON. Repeatable.")
    ap.add_argument("--per_scale_target",
                    choices=["pooled_marginal_nats", "mean_per_position_nats"],
                    default="pooled_marginal_nats",
                    help="which data statistic the PER-SCALE arm matches "
                         "(per-cell arm always uses per_position_entropies).")
    ap.add_argument("--out_per_scale", help="output per-scale anchor JSON")
    ap.add_argument("--out_per_cell", help="output per-cell anchor JSON")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.data or not args.point:
        ap.error("--data and at least one --point are required (or --selftest)")
    if not (args.out_per_scale or args.out_per_cell):
        ap.error("at least one of --out_per_scale / --out_per_cell required")

    data = json.load(open(args.data))
    temps, marginals = _load_curve(args.point)
    _check_consistency(data, marginals)
    print(f"data target: {args.data}  bias_correct={data.get('bias_correct')}")
    print(f"curve temps: {[float(t) for t in temps]}  ({len(marginals)} points)")

    all_status = []
    if args.out_per_scale:
        ps, statuses = solve_per_scale(data, temps, marginals, args.per_scale_target)
        json.dump(ps, open(args.out_per_scale, "w"), indent=2)
        all_status += statuses
        print(f"\nwrote {args.out_per_scale}  (per-scale, target={args.per_scale_target})")
        print(f"  {'scale':>6} {'H_data':>8} | " +
              " ".join(f"T{t:g}".rjust(7) for t in temps) + f" | {'T_k':>7}  status")
        for s in [data['scales'][i] for i in data['trainable_scale_indices']]:
            e = ps["per_scale"][str(s)]
            curve = " ".join(f"{h:>7.3f}" for h in e["H_roll_curve_nats"])
            print(f"  {s:>6} {e['H_data_nats']:>8.3f} | {curve} | "
                  f"{e['T_k']:>7.3f}  {e['solve_status']}")

    if args.out_per_cell:
        pc, summary = solve_per_cell(data, temps, marginals)
        json.dump(pc, open(args.out_per_cell, "w"), indent=2)
        for v in summary.values():
            all_status += (["clamped_lo"] * v["clamped_lo"]
                           + ["saturated_hi"] * v["saturated_hi"])
        print(f"\nwrote {args.out_per_cell}  (per-cell, P={pc['tokens_per_frame']})")
        print(f"  {'scale':>6} {'cells':>6} {'Tmin':>6} {'Tmed':>6} {'Tmax':>6} "
              f"{'ok':>5} {'lo':>5} {'hi':>5} {'flat':>5}")
        for s, v in summary.items():
            print(f"  {s:>6} {v['n_cells']:>6} {v['T_min']:>6.2f} {v['T_median']:>6.2f} "
                  f"{v['T_max']:>6.2f} {v['ok']:>5} {v['clamped_lo']:>5} "
                  f"{v['saturated_hi']:>5} {v['flat_fallback']:>5}")

    hints = _supplementary_hint(all_status)
    if hints:
        print("\nBRACKETING:")
        for h in hints:
            print(f"  - {h}")
    else:
        print("\nBRACKETING: all targets bracketed by the swept-T curve.")


if __name__ == "__main__":
    main()
