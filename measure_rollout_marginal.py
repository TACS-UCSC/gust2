"""measure_rollout_marginal.py — the HORIZON emitted-token marginal of a rollout.

The keystone of the horizon-marginal temperature calibration (paper section B).
Every prior section-B anchor (calibration gate, etmodel/etpos, data_mix,
drift_warm) targeted a SINGLE-STEP teacher-forced statistic — the per-step
predictive/conditional entropy `H(softmax(z_k))` — which the gate proved solves
to T<=1 at every scale. THIS quantity is different in kind: the entropy of the
EMITTED TOKEN at a (scale, cell), aggregated over the whole rollout HORIZON

    H_roll(X_k) = H( { token the rollout emitted at scale k, step t } over t )

a distribution over TIME, not over the vocab at a fixed step. At coarse scales
the per-step conditional entropy is ~0 (the model is correctly near-certain
given the previous frame) yet the time-marginal is large because the large-scale
state visits many configurations over the trajectory. Mean-field collapse IS the
rollout time-marginal `H_roll` shrinking below the data marginal `H_data`; the
single-step conditional is structurally blind to it. A FIXED per-scale/per-cell
temperature (samplers.py `per_scale_temp`) does not force each step diffuse (the
etpos failure) — it lets the small per-step entropy accumulate over the horizon
until `H_roll` matches `H_data`. This tool measures `H_roll(T)`; the curve over
a few global-T rollouts is inverted by `solve_calib_schedule.py` to hit the data
target. Same per-scale estimator as `measure_tokenizer_entropy.py` (reused
verbatim) so `H_roll` and `H_data` are computed identically and diff directly.

CAVEAT (drives judging): matching the marginal ENTROPY does not match the
marginal DISTRIBUTION — a warmed rollout can hit the target entropy with a
wrong-shape over-diffuse marginal (the TKE-RSE gaming trap restated). Calibration
success is judged on EMD / PDF / per-position OOD / spectra vs the swept-T
yardstick, NEVER on entropy agreement. Entropy is the knob, not the metric.

Usage:
    python measure_rollout_marginal.py --rollout ROLLOUT_TOKENS.npz --output OUT.json
    python measure_rollout_marginal.py --rollout R.npz --data_ref DATA_TARGET.json
    python measure_rollout_marginal.py --rollout R.npz --use gt --output OUT.json   # sanity
    python measure_rollout_marginal.py --selftest
"""

import argparse
import json

import numpy as np

from measure_tokenizer_entropy import per_scale_entropies


def _predicted_frames(npz, which: str) -> np.ndarray:
    """Flat (M, tokens_per_frame) token matrix of the rollout's predicted (or GT)
    frames, dropping frame 0 (the GT seed copied from data, rollout_nsp.py:620)."""
    key = "rollout_indices" if which == "pred" else "gt_indices"
    arr = np.asarray(npz[key])
    if arr.ndim == 2:                      # N==1 was squeezed to (T+1, P)
        arr = arr[None]
    # arr: (N, T+1, P) -> drop seed frame -> (N, T, P) -> (N*T, P)
    pred = arr[:, 1:, :]
    return pred.reshape(-1, pred.shape[-1])


def measure(rollout_path: str, which: str = "pred",
            bias_correct: bool = False) -> dict:
    d = np.load(rollout_path, allow_pickle=True)
    scales = [int(s) for s in np.asarray(d["scales"]).ravel()]
    V = int(d["effective_vocab_size"])
    flat = _predicted_frames(d, which)                     # (M, tokens_per_frame)
    M = int(flat.shape[0])

    per_scale, _ = per_scale_entropies(flat, scales, V, bias_correct=bias_correct)

    # trainable scales: prefer an explicit key, else the data convention (leading
    # near-deterministic scales are the seed). Detect via 1 unique pooled code.
    if "first_trainable_scale" in d.files:
        first_trainable = int(d["first_trainable_scale"])
    else:
        first_trainable = 0
        for k, s in enumerate(scales):
            if per_scale[s]["n_unique_codes_pooled"] <= 1:
                first_trainable = k + 1
            else:
                break
    trainable_idx = list(range(first_trainable, len(scales)))
    pooled_arr = [per_scale[scales[i]]["pooled_marginal_nats"] for i in trainable_idx]
    pp_arr = [per_scale[scales[i]]["mean_per_position_nats"] for i in trainable_idx]

    return {
        "rollout_path": rollout_path,
        "which": which,
        "n_predicted_frames": M,
        "effective_vocab_size": V,
        "scales": scales,
        "first_trainable_scale": first_trainable,
        "trainable_scale_indices": trainable_idx,
        "bias_correct": bool(bias_correct),
        "per_scale": {str(s): per_scale[s] for s in scales},
        "per_trainable_pooled_marginal_nats": pooled_arr,
        "per_trainable_mean_per_position_nats": pp_arr,
    }


def _print_gap_table(roll: dict, data_ref_path: str):
    """scale | H_data | H_roll | gap, for both pooled and mean-per-position."""
    ref = json.load(open(data_ref_path))
    rps, dps = roll["per_scale"], ref["per_scale"]
    if roll.get("bias_correct") != ref.get("bias_correct"):
        print(f"  [warn] bias_correct mismatch: rollout={roll.get('bias_correct')} "
              f"data={ref.get('bias_correct')} — gaps include an estimator-bias "
              f"differential; rebuild one side to match.")
    print(f"  H_roll vs H_data (nats)   rollout M={roll['n_predicted_frames']}  "
          f"which={roll['which']}")
    print(f"  {'scale':>6} | {'Hd_pool':>8} {'Hr_pool':>8} {'gap':>7} | "
          f"{'Hd_pos':>8} {'Hr_pos':>8} {'gap':>7}")
    for s in roll["scales"]:
        ss = str(s)
        if ss not in dps:
            continue
        dp, rp = dps[ss], rps[ss]
        gp = rp["pooled_marginal_nats"] - dp["pooled_marginal_nats"]
        gpp = rp["mean_per_position_nats"] - dp["mean_per_position_nats"]
        flag = "" if abs(gp) < 0.05 else ("  COLLAPSED" if gp < 0 else "  over-diffuse")
        print(f"  {s:>6} | {dp['pooled_marginal_nats']:>8.4f} "
              f"{rp['pooled_marginal_nats']:>8.4f} {gp:>+7.3f} | "
              f"{dp['mean_per_position_nats']:>8.4f} "
              f"{rp['mean_per_position_nats']:>8.4f} {gpp:>+7.3f}{flag}")
    print("  (gap<0 => rollout under-diverse vs data at this scale => warm it; "
          "gap>0 => cool it. Bracketing: a target outside the swept-T curve "
          "will clamp in solve_calib_schedule.)")


def _selftest():
    rng = np.random.default_rng(0)
    scales = np.array([1, 2])                          # P = 1 + 4 = 5
    V = 6
    N, T = 4, 50
    # frame 0 = seed (single code 0); predicted frames 1..T: scale1 uniform {2,3}.
    arr = np.zeros((N, T + 1, 5), dtype=np.int32)
    arr[:, :, 0] = 0
    arr[:, 1:, 1:] = rng.integers(2, 4, size=(N, T, 4))
    # frame 0 deliberately polluted to a different code -> must be dropped.
    arr[:, 0, 1:] = 5
    np.savez("/tmp/_rm_self.npz", rollout_indices=arr, gt_indices=arr,
             scales=scales, effective_vocab_size=V)
    r = measure("/tmp/_rm_self.npz", which="pred")
    # seed dropped => scale 2 marginal is uniform over {2,3} => H = log 2.
    assert abs(r["per_scale"]["2"]["pooled_marginal_nats"] - np.log(2)) < 1e-2, r
    assert abs(r["per_scale"]["2"]["mean_per_position_nats"] - np.log(2)) < 1e-2, r
    assert r["per_scale"]["1"]["pooled_marginal_nats"] < 1e-9             # seed=const
    assert r["n_predicted_frames"] == N * T
    assert r["first_trainable_scale"] == 1 and r["trainable_scale_indices"] == [1]
    assert len(r["per_scale"]["2"]["per_position_entropies"]) == 4
    # if frame 0 (code 5) had leaked in, the marginal would exceed log 2.
    assert r["per_scale"]["2"]["n_unique_codes_pooled"] == 2, "seed frame leaked!"
    # MM raises the estimate.
    rmm = measure("/tmp/_rm_self.npz", which="pred", bias_correct=True)
    assert rmm["per_scale"]["2"]["mean_per_position_nats"] >= \
        r["per_scale"]["2"]["mean_per_position_nats"]
    print("[ok] measure_rollout_marginal self-test passed "
          "(seed dropped, H=log2 marginal, MM monotone, alignment)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollout", type=str, help="rollout_tokens.npz")
    ap.add_argument("--output", type=str, help="output marginal .json")
    ap.add_argument("--use", choices=["pred", "gt"], default="pred",
                    help="pred = predicted tokens (default); gt = ground-truth "
                         "continuation (sanity: should reproduce the data target).")
    ap.add_argument("--data_ref", type=str, default=None,
                    help="data-target JSON (from measure_tokenizer_entropy.py) — "
                         "print the per-scale H_roll - H_data gap table.")
    ap.add_argument("--bias_correct", choices=["none", "miller_madow"],
                    default="none",
                    help="MUST match the data-target build (per-cell arm).")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.rollout:
        ap.error("--rollout is required (or use --selftest)")

    roll = measure(args.rollout, which=args.use,
                   bias_correct=(args.bias_correct == "miller_madow"))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(roll, f, indent=2)
        print(f"wrote {args.output}")
    print(f"  {args.rollout}")
    print(f"  predicted frames M={roll['n_predicted_frames']}  V={roll['effective_vocab_size']}"
          f"  scales={roll['scales']}  trainable={roll['trainable_scale_indices']}"
          f"  bias_correct={roll['bias_correct']}")
    print(f"  {'scale':>6} {'pooled':>10} {'mean/pos':>10} {'npos':>6} {'uniq':>6}")
    for s in roll["scales"]:
        e = roll["per_scale"][str(s)]
        print(f"  {s:>6} {e['pooled_marginal_nats']:>10.4f} "
              f"{e['mean_per_position_nats']:>10.4f} {e['n_positions']:>6} "
              f"{e['n_unique_codes_pooled']:>6}")
    if args.data_ref:
        print()
        _print_gap_table(roll, args.data_ref)


if __name__ == "__main__":
    main()
