"""measure_calibration_temp.py — per-scale CALIBRATION temperature anchor (gate).

The decisive cheap gate for the calibration-temperature sampler (paper section
B, "Recipe 1(ii)"). It is the offline sibling of the `etmodel` anchor, but it
targets a STRICTLY DIFFERENT quantity:

  * `measure_model_entropy.py` (etmodel) anchors to the model's PREDICTIVE
    entropy  H_k = mean_pos H(softmax(z_k)).  H is, by construction, the T=1
    entropy of the teacher-forced logits, so an entropy homeostat aimed at H
    yields T_k ~= 1 everywhere and CANNOT warm — it nails cold sc341 but
    under-warms (collapses) the warm configs.

  * THIS tool anchors to the per-scale CROSS-ENTROPY / validation NLL
    CE_k = E[-log p_model(y_true)] = H_k + KL(p_data || p_model). CE is strictly
    larger than H by the calibration gap (the model's overconfidence, which the
    entropy-calibration literature names as the cause of degenerate free-running
    generation). The gap is DATA-MEASURED to grow with token count, so solving

        H(softmax(z_k / T_k)) = CE_k

    for a per-scale temperature T_k lands COLD on the calibrated easy config and
    progressively WARMER on the miscalibrated hard configs, with NO per-config
    knob. T_k is the principled, parameter-free temperature schedule.

THE GATE. Before spending any multi-hour rollout compute, run this (minutes:
one teacher-forced pass over 256 (t0,t1) val pairs, NO rollout) for the matched
small+medium x {sc341, sc917, sc1941} cells and check whether the solved T_k
schedule tracks the swept-temperature yardstick (sc341~1.0, sc917~1.72,
sc1941~1.85) — see analyze_calibration_gate.py for the go/no-go. The T_k solve
is done by inverting the entropy-vs-temperature curve on the REAL teacher-forced
logits (NOT a Gaussian/first-order proxy: the entropy->T map is logit-tail-shape
sensitive, 1.5x-9x across logit families, so the real-logit solve is the entire
point).

The forward is `eval_single_step.make_predict_and_loss` with `temp_grid` set, so
there is no new forward logic to get wrong; CE_k = per_scale_ce (the exact,
OOD-masked, log-weighting-matched validation NLL) and the grid curve come from
the same validated logits.

Output JSON reuses measure_tokenizer_entropy.py's schema so the rollout's
`--entropy_anchor_path` loader needs zero changes — BUT the two per_trainable
slots hold the per-scale TEMPERATURE T_k (not an entropy), consumed by
`--sampler per_scale_temp`.

Usage:
    python measure_calibration_temp.py \
        --checkpoint_dir experiments/ar-robust-scaling/small-sc917-nsp-s74 \
        --tokens_path experiments/tokens/small-sc917-val.npz \
        --output inference_anchors/small-sc917-calib.json \
        --max_pairs 256

    python measure_calibration_temp.py --selftest   # offline inversion test
"""

import argparse
import json
import os

import jax
jax.config.update("jax_threefry_partitionable", False)
import jax.numpy as jnp
import equinox as eqx
import numpy as np


# Temperature grid for the H(softmax(z/T)) = CE inversion. Geometric so the
# resolution is densest in the cold region; spans well past the "over-diffusive"
# kill threshold (T >> 2.5) so a saturating solve is visible rather than clipped
# silently.
TEMP_GRID_LO = 0.2
TEMP_GRID_HI = 8.0
TEMP_GRID_N = 96


def _temp_grid():
    return np.geomspace(TEMP_GRID_LO, TEMP_GRID_HI, TEMP_GRID_N).astype(np.float64)


def solve_Tk(ce, grid, H_grid):
    """Invert mean-entropy(T) = CE per scale on the REAL grid curve.

    ce:     (n_trainable,)            per-scale cross-entropy targets (nats)
    grid:   (G,)                      temperatures (ascending)
    H_grid: (n_trainable, G)          mean predictive entropy at each grid temp

    Returns (T_k, status) where status in {"ok","saturated_hi","clamped_lo"}:
      * H is monotone increasing in T, so np.interp inverts it directly.
      * CE above the achievable ceiling (small support) -> T_k = grid hi,
        "saturated_hi" (a FAIL signal: the target wants more entropy than the
        scale's support can supply -> would over-diffuse).
      * CE below the floor (coarse, near-deterministic scales) -> T_k = grid lo,
        "clamped_lo" (cold; expected at the coarsest scales).
    """
    grid = np.asarray(grid, dtype=np.float64)
    ce = np.asarray(ce, dtype=np.float64)
    H_grid = np.asarray(H_grid, dtype=np.float64)
    n = ce.shape[0]
    T_k = np.empty(n, dtype=np.float64)
    status = []
    for j in range(n):
        Hg = np.maximum.accumulate(H_grid[j])   # enforce monotone vs float noise
        tgt = ce[j]
        if tgt <= Hg[0]:
            T_k[j] = grid[0]
            status.append("clamped_lo")
        elif tgt >= Hg[-1]:
            T_k[j] = grid[-1]
            status.append("saturated_hi")
        else:
            T_k[j] = float(np.interp(tgt, Hg, grid))
            status.append("ok")
    return T_k, status


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint_dir", help="NSP checkpoint dir")
    ap.add_argument("--tokens_path",
                    help="Tokenized .npz (val tokens — the transitions we forecast)")
    ap.add_argument("--output", help="Output anchor .json (T_k schedule)")
    ap.add_argument("--max_pairs", type=int, default=256,
                    help="Number of consecutive (t0,t1) pairs to average over.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true",
                    help="Run the offline inversion self-test (no model) and exit.")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    for req in ("checkpoint_dir", "tokens_path", "output"):
        if getattr(args, req) is None:
            ap.error(f"--{req} is required (unless --selftest)")

    # Imports deferred so --selftest needs no JAX model stack / checkpoint.
    from nsp_model import NSPConfig, create_nsp_model, build_teacher_forced_mask
    from eval_single_step import make_predict_and_loss
    from tokenizer import load_tokenized_data

    # --- Load tokens ---
    token_data = load_tokenized_data(args.tokens_path)
    indices = token_data["indices_flat"]
    scales_int = [int(s) for s in token_data["scales"]]
    scale_masks = jnp.array(token_data["scale_masks"])
    first_trainable = int(token_data.get("first_trainable_scale", 0))

    # --- Load NSP model ---
    state_path = os.path.join(args.checkpoint_dir, "training_state.json")
    with open(state_path) as f:
        arch = json.load(f)["arch_config"]
    config = NSPConfig(
        n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
        dropout=0.0, rope_theta=arch["rope_theta"],
        n_refine_layers=arch["n_refine_layers"],
    )
    key = jax.random.PRNGKey(args.seed)
    model, exp_heads = create_nsp_model(token_data, config, key)
    model = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "model.eqx"), model)
    exp_heads = eqx.tree_deserialise_leaves(
        os.path.join(args.checkpoint_dir, "exp_heads.eqx"), exp_heads)
    model = eqx.nn.inference_mode(model)
    exp_heads = eqx.nn.inference_mode(exp_heads)

    # --- Build the teacher-forced forward (reused) WITH the temperature grid ---
    trainable_indices = config.trainable_scale_indices
    scales_t0 = config.scales
    scales_t1 = config.scales[:-1]
    tokens_t0 = sum(h * w for h, w in scales_t0)
    tokens_t1 = sum(h * w for h, w in scales_t1)
    padded_t0 = ((tokens_t0 + 127) // 128) * 128
    padded_t1 = ((tokens_t1 + 127) // 128) * 128
    attn_bias = build_teacher_forced_mask(scales_t0, padded_t0, scales_t1, padded_t1)
    grid = _temp_grid()
    predict_and_loss = make_predict_and_loss(
        config, scales_t0, padded_t0, scales_t1, padded_t1,
        attn_bias, scale_masks, trainable_indices,
        temperature=0.0, temp_grid=grid)

    # --- Average per-scale CE, H, and the grid-entropy curve over pairs ---
    n_pairs = min(len(indices) - 1, args.max_pairs)
    keys = jax.random.split(key, n_pairs)
    n_tr = len(trainable_indices)
    acc_ce = np.zeros(n_tr, dtype=np.float64)
    acc_H = np.zeros(n_tr, dtype=np.float64)
    acc_ood = np.zeros(n_tr, dtype=np.float64)
    acc_grid = np.zeros((n_tr, len(grid)), dtype=np.float64)
    for i in range(n_pairs):
        t0 = jnp.array(indices[i])
        t1 = jnp.array(indices[i + 1])
        (_, _, per_scale_ce, per_scale_ood,
         per_scale_entropy, per_scale_grid_entropy) = predict_and_loss(
            model, exp_heads, t0, t1, keys[i])
        acc_ce += np.asarray(per_scale_ce, dtype=np.float64)
        acc_H += np.asarray(per_scale_entropy, dtype=np.float64)
        acc_ood += np.asarray(per_scale_ood, dtype=np.float64)
        acc_grid += np.asarray(per_scale_grid_entropy, dtype=np.float64)
    ce = acc_ce / n_pairs
    H = acc_H / n_pairs
    ood = acc_ood / n_pairs
    H_grid = acc_grid / n_pairs

    # --- Solve the per-scale calibration temperature on the real curve ---
    T_k, status = solve_Tk(ce, grid, H_grid)
    gap = ce - H

    # --- Aggregate effective temperatures (summaries for the gate) ---
    trainable_scale_indices = list(range(first_trainable, len(scales_int)))
    assert len(trainable_scale_indices) == n_tr, (trainable_scale_indices, n_tr)
    token_counts = np.array(
        [scales_int[idx] * scales_int[idx] for idx in trainable_scale_indices],
        dtype=np.float64)
    # token-count weighted: finest scales (which dominate the field and need the
    # warmth) dominate the summary; matches where collapse originates.
    T_eff_weighted = float(np.sum(T_k * token_counts) / np.sum(token_counts))
    T_eff_unweighted = float(np.mean(T_k))
    finest_scale = scales_int[trainable_scale_indices[-1]]
    finest_T = float(T_k[-1])
    finest_ce = float(ce[-1])

    # --- Assemble JSON (schema-compatible with the rollout anchor loader) ---
    per_scale = {
        str(scales_int[idx]): {
            "ce_nats": float(ce[j]),
            "model_entropy_nats": float(H[j]),
            "gap_nats": float(gap[j]),
            "ood_rate": float(ood[j]),
            "T_k_calibration": float(T_k[j]),
            "solve_status": status[j],
        }
        for j, idx in enumerate(trainable_scale_indices)
    }
    Tk_list = [float(x) for x in T_k]
    anchor = {
        "source": "model_calibration_temp",
        "checkpoint_dir": args.checkpoint_dir,
        "tokens_path": args.tokens_path,
        "n_pairs": int(n_pairs),
        "scales": scales_int,
        "first_trainable_scale": first_trainable,
        "trainable_scale_indices": trainable_scale_indices,
        "temp_grid": [float(x) for x in (TEMP_GRID_LO, TEMP_GRID_HI, TEMP_GRID_N)],
        "per_scale": per_scale,
        "aggregate": {
            "T_eff_token_weighted": T_eff_weighted,
            "T_eff_unweighted": T_eff_unweighted,
            "finest_scale": finest_scale,
            "finest_scale_ce_nats": finest_ce,
            "finest_scale_T": finest_T,
        },
        # Both standard anchor slots hold the per-scale TEMPERATURE T_k (not an
        # entropy), so --sampler per_scale_temp resolves either --anchor_stat to
        # the schedule with no loader change.
        "per_trainable_pooled_marginal_nats": Tk_list,
        "per_trainable_mean_per_position_nats": Tk_list,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(anchor, f, indent=2)

    # --- Report ---
    print(f"wrote {args.output}  (n_pairs={n_pairs})")
    print(f"  {'scale':>6} {'CE':>8} {'H':>8} {'gap':>8} {'T_k':>8}  status")
    for j, idx in enumerate(trainable_scale_indices):
        print(f"  {scales_int[idx]:>6} {ce[j]:>8.4f} {H[j]:>8.4f} "
              f"{gap[j]:>8.4f} {T_k[j]:>8.4f}  {status[j]}")
    print(f"  T_eff(token-weighted)={T_eff_weighted:.4f}  "
          f"T_eff(unweighted)={T_eff_unweighted:.4f}  "
          f"finest(scale {finest_scale}) T={finest_T:.4f}")


def _selftest():
    """Offline test of the entropy->T inversion (no model needed)."""
    grid = _temp_grid()

    # Build a synthetic monotone entropy curve per scale from a known logit
    # vector: H(softmax(z / T)) at each grid T. Then pick a target CE that is the
    # entropy at a KNOWN T_true and confirm the inversion recovers it.
    def H_of(logits_row, T):
        z = logits_row / T
        z = z - z.max()
        p = np.exp(z); p = p / p.sum()
        nz = p > 0
        return float(-(p[nz] * np.log(p[nz])).sum())

    rng = np.random.default_rng(0)
    rows = [rng.normal(size=12) * s for s in (3.0, 2.0, 1.0)]   # sharp->soft
    H_grid = np.array([[H_of(r, T) for T in grid] for r in rows])

    # monotone increasing in T
    for r in range(len(rows)):
        d = np.diff(H_grid[r])
        assert np.all(d >= -1e-9), f"H not monotone in T (row {r})"

    T_true = np.array([0.7, 1.5, 3.0])
    ce = np.array([H_of(rows[r], T_true[r]) for r in range(len(rows))])
    T_k, status = solve_Tk(ce, grid, H_grid)
    err = np.max(np.abs(T_k - T_true))
    assert err < 0.05, f"inversion error {err:.4f} too large: {T_k} vs {T_true}"
    assert all(s == "ok" for s in status), status
    print(f"[ok] inversion recovers known T_true within {err:.4f}: {T_k.round(3)}")

    # CE above the achievable ceiling -> saturated_hi at grid top.
    ceiling = H_grid[:, -1]
    T_sat, st_sat = solve_Tk(ceiling + 1.0, grid, H_grid)
    assert all(s == "saturated_hi" for s in st_sat), st_sat
    assert np.allclose(T_sat, grid[-1]), T_sat
    print(f"[ok] CE above support ceiling saturates at T={grid[-1]:.2f} (FAIL signal)")

    # CE below the floor -> clamped_lo at grid bottom.
    T_lo, st_lo = solve_Tk(np.zeros(len(rows)), grid, H_grid)
    assert all(s == "clamped_lo" for s in st_lo), st_lo
    assert np.allclose(T_lo, grid[0]), T_lo
    print(f"[ok] CE below floor clamps at T={grid[0]:.2f} (cold)")

    print("\nALL CALIBRATION-INVERSION SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
