"""scale_distribution_drift.py — per-scale token-DISTRIBUTION divergence.

A refinement of pixel-EMD / per-scale entropy for measuring off-manifold drift.
Entropy (measure_tokenizer_entropy / measure_rollout_marginal) is a scalar per
scale; a warmed rollout can hit the target entropy with a WRONG-SHAPE marginal
(that tool's own caveat: "Entropy is the knob, not the metric"). This compares
the full per-scale code DISTRIBUTION:

  p_k = pooled code histogram at scale k (over all positions+frames)
  divergences: TV(p,q) in [0,1] (mass misplaced), JS(p,q) in [0,1] bits
               (symmetric, bounded), signed entropy gap H(q)-H(p)
               (<0 collapse / code-starvation, >0 over-spread / noise)

Native to the model's generative space, localizes drift coarse-vs-fine, and is
direction-resolved. Reuses the exact pooled binning of per_scale_entropies.

Modes:
  windows : stationarity / "can the target come from training data?" — compares
            a config's TRAIN tokens vs VAL tokens per scale, with a within-train
            sampling-noise floor (two disjoint same-size train windows) so the
            train-vs-val divergence is interpretable (== floor => indistinguishable).
  drift   : rollout pred per-scale dist vs a data reference, per scale (+ optional
            time-resolved over the rollout horizon).

  python scale_distribution_drift.py --mode windows --config medium-sc1941
  python scale_distribution_drift.py --mode windows --all
  python scale_distribution_drift.py --mode drift --rollout R.npz --data_ref TOK.npz
  python scale_distribution_drift.py --selftest
"""
import argparse
import glob
import os

import numpy as np


def _scale_boundaries(scales):
    cps = [int(s) * int(s) for s in scales]
    return np.concatenate([[0], np.cumsum(cps)]).astype(int)


def per_scale_pooled_dist(indices_flat, scales, V):
    """list of (V,) pooled code probability vectors, one per scale."""
    b = _scale_boundaries(scales)
    dists = []
    for k in range(len(scales)):
        block = indices_flat[:, b[k]:b[k + 1]].astype(np.int64).ravel()
        h = np.bincount(block, minlength=V).astype(np.float64)
        dists.append(h / max(h.sum(), 1.0))
    return dists


def _H_bits(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def tv(p, q):
    return float(0.5 * np.abs(p - q).sum())


def js_bits(p, q):
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return float((a[mask] * (np.log2(a[mask]) - np.log2(b[mask]))).sum())
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _load(path):
    d = np.load(path, allow_pickle=True)
    return (np.asarray(d["indices_flat"]),
            [int(s) for s in np.asarray(d["scales"]).ravel()],
            int(d["effective_vocab_size"]))


def _divergences(idxA, idxB, scales, V):
    """per-scale dict: tv, js, dH, n_codes, n_pos, n_samples (B side)."""
    dA = per_scale_pooled_dist(idxA, scales, V)
    dB = per_scale_pooled_dist(idxB, scales, V)
    b = _scale_boundaries(scales)
    rows = []
    for k, s in enumerate(scales):
        npos = int(b[k + 1] - b[k])
        rows.append({
            "scale": s, "n_pos": npos,
            "tv": tv(dA[k], dB[k]), "js": js_bits(dA[k], dB[k]),
            "dH": _H_bits(dB[k]) - _H_bits(dA[k]),
            "H_ref": _H_bits(dA[k]), "n_codes_ref": int((dA[k] > 0).sum()),
        })
    return rows


def run_windows(config, tok_dir, win=None):
    train_p = os.path.join(tok_dir, f"{config}.npz")
    val_p = os.path.join(tok_dir, f"{config}-val.npz")
    if not (os.path.exists(train_p) and os.path.exists(val_p)):
        print(f"[skip] {config}: missing train/val npz"); return None
    tr, scales, V = _load(train_p)
    va, _, _ = _load(val_p)
    nval = va.shape[0]
    w = win or nval                                   # match val window size
    # need 2 disjoint train windows of size w for the noise floor
    if tr.shape[0] < 2 * w:
        w = tr.shape[0] // 2
    trA, trB = tr[:w], tr[w:2 * w]
    floor = _divergences(trA, trB, scales, V)         # same data, two windows
    tv_v = _divergences(trA, va[:w], scales, V)       # train window vs val window
    print(f"\n=== {config}  (V={V}, train {tr.shape[0]}f, val {nval}f, window {w}f) ===")
    print(f"  {'scale':>5} {'n_pos':>6} {'H_data(bits)':>12} "
          f"{'JS floor':>9} {'JS tr-vs-val':>12} {'TV tr-vs-val':>12} {'excess':>8}")
    excess = []
    for fr, vv in zip(floor, tv_v):
        ex = vv["js"] - fr["js"]
        excess.append(ex)
        print(f"  {fr['scale']:>5} {fr['n_pos']:>6} {fr['H_ref']:>12.3f} "
              f"{fr['js']:>9.2e} {vv['js']:>12.2e} {vv['tv']:>12.2e} {ex:>8.1e}")
    print(f"  -> max excess JS over scales: {max(excess):.2e} "
          f"({'INDISTINGUISHABLE from sampling noise' if max(excess) < 5e-3 else 'CHECK'})")
    return {"config": config, "max_excess_js": float(max(excess))}


def compute_drift(rollout, data_ref=None):
    """per-scale drift rows of a rollout vs its data reference (no printing).

    Default reference = the rollout's OWN gt_indices (same tokenizer/compact
    space by construction). Pass data_ref ONLY if it shares the exact mapping.
    """
    d = np.load(rollout, allow_pickle=True)
    arr = np.asarray(d["rollout_indices"])
    if arr.ndim == 2:
        arr = arr[None]
    pred = arr[:, 1:, :]                               # drop seed frame
    scales = [int(s) for s in np.asarray(d["scales"]).ravel()]
    flat = pred.reshape(-1, pred.shape[-1])
    if data_ref:
        ref_idx, ref_scales, V = _load(data_ref)      # external token file
        scales = ref_scales
    else:
        gt = np.asarray(d["gt_indices"])
        if gt.ndim == 2:
            gt = gt[None]
        ref_idx = gt[:, 1:, :].reshape(-1, gt.shape[-1])
        V = int(d["effective_vocab_size"])
    return _divergences(ref_idx, flat, scales, V)      # ref = data, B = rollout


def run_drift(rollout, data_ref=None, n_time=0):
    rows = compute_drift(rollout, data_ref)
    ref_name = os.path.basename(data_ref) if data_ref else "own gt_indices"
    print(f"\n=== drift: {os.path.basename(rollout)} vs {ref_name} ===")
    print(f"  {'scale':>5} {'JS':>9} {'TV':>9} {'dH(bits)':>10}  signature")
    for r in rows:
        # JS = magnitude of drift; dH = direction. Entropy (dH) alone is blind to
        # a same-spread shift onto disjoint codes (dH~0 yet JS~1) -> key off JS.
        if r["js"] < 0.03:
            sig = "on-distribution"
        elif r["dH"] < -0.15:
            sig = "collapse(code-starved)"
        elif r["dH"] > 0.15:
            sig = "over-spread(noise)"
        else:
            sig = "shifted(disjoint codes)"
        print(f"  {r['scale']:>5} {r['js']:>9.3f} {r['tv']:>9.3f} {r['dH']:>10.3f}  {sig}")
    return rows


def _selftest():
    rng = np.random.default_rng(0)
    scales, V = [1, 2], 4
    # scale0 (1 pos) const code 0; scale1 (4 pos) uniform over {1,2}
    N = 20000
    s0 = np.zeros((N, 1), int)
    s1 = rng.integers(1, 3, size=(N, 4))
    idx = np.concatenate([s0, s1], axis=1)
    d = per_scale_pooled_dist(idx, scales, V)
    assert abs(_H_bits(d[0]) - 0.0) < 1e-9
    assert abs(_H_bits(d[1]) - 1.0) < 0.02, _H_bits(d[1])
    # identical sets -> ~0 divergence; shifted -> >0
    idx2 = np.concatenate([s0, rng.integers(1, 3, size=(N, 4))], axis=1)
    r = _divergences(idx, idx2, scales, V)
    assert r[1]["js"] < 1e-3 and r[1]["tv"] < 0.05
    idx3 = np.concatenate([s0, np.full((N, 4), 1)], axis=1)   # collapsed to code 1
    r3 = _divergences(idx, idx3, scales, V)
    assert r3[1]["dH"] < -0.9 and r3[1]["js"] > 0.2
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["windows", "drift"])
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--tok_dir", default="codebook_artifacts/tokens")
    ap.add_argument("--rollout")
    ap.add_argument("--data_ref")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); return
    if a.mode == "windows":
        if a.all:
            configs = sorted({os.path.basename(p)[:-4] for p in
                              glob.glob(os.path.join(a.tok_dir, "*.npz"))
                              if not p.endswith("-val.npz")})
            res = [run_windows(c, a.tok_dir) for c in configs]
            res = [r for r in res if r]
            print("\n==== SUMMARY (max excess JS over scales, train-vs-val above noise floor) ====")
            for r in sorted(res, key=lambda x: -x["max_excess_js"]):
                print(f"  {r['config']:18s} {r['max_excess_js']:.2e}")
        else:
            run_windows(a.config, a.tok_dir)
    elif a.mode == "drift":
        run_drift(a.rollout, a.data_ref)


if __name__ == "__main__":
    main()
