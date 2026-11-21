"""calibrate_schedule.py — token-only temperature calibration for a per-scale schedule family,
with the whole probe walk in one process (the model is loaded and compiled once; the schedule is a
traced argument, so a new temperature does not recompile).

A schedule family is a weight per trainable scale, w_k, and one scalar T:
        T_k = 1 + (T - 1) * w_k
    --rule equal     w_k = 1                     (every scale at T; the paper's global read)
    --rule fine2     w_k = 1 on the finest two scales, 0 elsewhere      (fine-only)
    --rule renewal   w_k = f_k / f_finest, f_k = fraction of training tokens at scale k that differ
                     between consecutive frames  (heat in proportion to how often the data renews
                     the token at that scale)

The scalar is solved from probes scored on their own emitted tokens only (no ground truth):
    gap_k(T) = H_roll_k(T) - H_data_k           (plug-in pooled marginal entropy, nats)
    --criterion mean2   zero of the mean gap over the finest two scales          (paper, eq. globalread)
    --criterion ceil3   zero of the LARGEST gap over the finest three scales     (ceiling read: stop when
                        the first of those scales reaches its training entropy)
    --target full       H_data_k from the whole training record (inference_anchors convention)
    --target window     H_data_k = mean plug-in entropy over consecutive training windows of the probe's
                        length (same estimator at the same sample size as the probe)
Search (unchanged from the paper): first probe at --t0; the sign of the criterion sets the direction;
step by --step until the sign changes; linear interpolation inside the bracket.

The deployed protocol is ONE trajectory (seed --seed) of --probe_steps steps. --extra_seeds adds
trajectories that are scored and reported for spread but never enter the solve.

After the solve, one long probe (--long_steps) runs at the solved schedule and reports, per scale,
D_k = H(all steps) - H(first --probe_steps steps): a token-only stationarity check.

Writes <out_dir>/<cell>/<tag>/{calibration.json, anchor.json}. anchor.json is a per_scale_temp anchor
for `rollout_nsp.py --sampler per_scale_temp --entropy_anchor_path`.

    python calibrate_schedule.py --cell small-sc1941-nsp-s48 --root /opt/dlami/nvme/gust2 \
        --rule renewal --criterion ceil3 --target window
"""
import argparse
import json
import os
import re
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nsp_model import (NSPConfig, build_teacher_forced_mask, create_nsp_model,
                       generate_t1_frame_cached)
from samplers import SamplerConfig
from tokenizer import load_tokenized_data

CELL_RE = re.compile(r"^(small|medium|large)-(sc\d+)-nsp-(s\d+)$")


def plug_in_H(x, V):
    h = np.bincount(np.asarray(x).ravel(), minlength=V)
    p = h[h > 0] / h.sum()
    return float(-(p * np.log(p)).sum())


def per_scale_H(tok, bnd, V):
    """tok: (steps, P) or (N, steps, P) -> list of plug-in pooled entropies, one per scale."""
    return [plug_in_H(tok[..., bnd[k]:bnd[k + 1]], V) for k in range(len(bnd) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--root", required=True, help="folder holding experiments/{ar-robust-scaling,tokens}")
    ap.add_argument("--out_dir", default=None, help="default <root>/experiments/calib-schedule")
    ap.add_argument("--rule", choices=["equal", "fine2", "renewal"], required=True)
    ap.add_argument("--criterion", choices=["mean2", "ceil3"], default="ceil3")
    ap.add_argument("--target", choices=["full", "window"], default="window")
    ap.add_argument("--tag", default=None, help="default <rule>-<criterion>-<target>")
    ap.add_argument("--probe_steps", type=int, default=250)
    ap.add_argument("--long_steps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--extra_seeds", type=int, default=3, help="extra trajectories, reported only")
    ap.add_argument("--t0", type=float, default=1.0)
    ap.add_argument("--step", type=float, default=0.2)
    ap.add_argument("--tmin", type=float, default=0.4)
    ap.add_argument("--tmax", type=float, default=4.4)
    ap.add_argument("--max_probes", type=int, default=24)
    ap.add_argument("--probe_T", type=float, nargs="*", default=None,
                    help="probe exactly these temperatures and exit (no walk, no long probe)")
    ap.add_argument("--dump_tokens", action="store_true", help="save probe tokens of the deployed seed")
    a = ap.parse_args()

    m = CELL_RE.match(a.cell)
    assert m, a.cell
    size, sc, _ = m.groups()
    vq = f"{size}-{sc}"
    E = f"{a.root}/experiments"
    ckpt = f"{E}/ar-robust-scaling/{a.cell}"
    tag = a.tag or f"{a.rule}-{a.criterion}-{a.target}"
    out = os.path.join(a.out_dir or f"{E}/calib-schedule", a.cell, tag)
    os.makedirs(out, exist_ok=True)

    # ---- data, masks, model (same construction as rollout_nsp.py) ----
    td = load_tokenized_data(f"{E}/tokens/{vq}-val.npz")
    indices = td["indices_flat"]
    scale_masks = jnp.array(td["scale_masks"])
    tr = np.load(f"{E}/tokens/{vq}.npz")
    train_idx = np.asarray(tr["indices_flat"])
    V = int(tr["effective_vocab_size"])
    assert V == int(td["effective_vocab_size"]) and np.array_equal(tr["new_to_old"], td["new_to_old"])
    F_t, P = train_idx.shape
    M = np.zeros((P, V), dtype=bool)
    M[np.broadcast_to(np.arange(P), (F_t, P)).ravel(), train_idx.ravel().astype(np.int64)] = True
    position_mask = jnp.array(M)

    arch = json.load(open(os.path.join(ckpt, "training_state.json")))["arch_config"]
    config = NSPConfig(n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
                       dropout=0.0, rope_theta=arch["rope_theta"],
                       n_refine_layers=arch["n_refine_layers"])
    model, heads = create_nsp_model(td, config, jax.random.PRNGKey(a.seed))
    model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(os.path.join(ckpt, "model.eqx"), model))
    heads = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(os.path.join(ckpt, "exp_heads.eqx"), heads))

    tri = [int(i) for i in config.trainable_scale_indices]
    s0, s1 = config.scales, config.scales[:-1]
    n0, n1 = sum(h * w for h, w in s0), sum(h * w for h, w in s1)
    p0, p1 = ((n0 + 127) // 128) * 128, ((n1 + 127) // 128) * 128
    bias = build_teacher_forced_mask(s0, p0, s1, p1)
    scales = [int(h) for h, _ in config.scales]
    bnd = np.asarray(config.scale_boundaries).astype(int)
    assert P == bnd[-1]

    # ---- training-token statistics: targets and flip rates ----
    H_full = per_scale_H(train_idx, bnd, V)
    W = a.probe_steps
    wins = [train_idx[s:s + W] for s in range(0, F_t - W + 1, W)]
    Hw = np.array([per_scale_H(w, bnd, V) for w in wins])
    H_win, H_win_sd = Hw.mean(0).tolist(), Hw.std(0).tolist()
    flip = [float((train_idx[1:, bnd[k]:bnd[k + 1]] != train_idx[:-1, bnd[k]:bnd[k + 1]]).mean())
            for k in range(len(scales))]
    H_data = H_win if a.target == "window" else H_full

    # ---- schedule family ----
    nt = len(tri)
    if a.rule == "equal":
        w = np.ones(nt)
    elif a.rule == "fine2":
        w = np.array([0.0] * (nt - 2) + [1.0, 1.0])
    else:
        w = np.array([flip[i] for i in tri]) / flip[tri[-1]]

    def schedule(T):
        return (1.0 + (float(T) - 1.0) * w).astype(np.float32)

    read_scales = tri[-2:] if a.criterion == "mean2" else tri[-3:]

    def criterion(H):
        g = [H[k] - H_data[k] for k in read_scales]
        return float(np.mean(g)) if a.criterion == "mean2" else float(np.max(g))

    # ---- generation: N trajectories, schedule passed as a traced array ----
    cfg = SamplerConfig(method="per_scale_temp", temperature=1.0)

    def one(t0, key, anchor):
        return generate_t1_frame_cached(model, heads, config, t0, s0, p0, s1, p1, bias, scale_masks, tri,
                                        key, 1.0, 0, 1.0, 0, position_mask=position_mask,
                                        sampler_cfg=cfg, anchor_array=anchor)

    step_fn = jax.jit(jax.vmap(one, in_axes=(0, 0, None)))
    seeds = [a.seed + i for i in range(1 + a.extra_seeds)]

    def rollout(T, n_steps):
        """Free-running rollout from validation frame 0 at schedule(T). Returns (N, n_steps, P) tokens.
        Keys follow rollout_nsp.py: trajectory i uses split(PRNGKey(seed + i), n_steps)."""
        n_steps = min(n_steps, len(indices) - 1)
        anchor = jnp.asarray(schedule(T))
        keys = jnp.stack([jax.random.split(jax.random.PRNGKey(s), n_steps) for s in seeds])
        t = jnp.asarray(np.repeat(indices[0][None], len(seeds), 0), dtype=jnp.int32)
        toks = []
        for i in range(n_steps):
            t = step_fn(t, keys[:, i, :], anchor)
            toks.append(np.asarray(t))
        return np.stack(toks, axis=1)

    probes = {}

    def probe(T):
        t_start = time.time()
        tok = rollout(T, a.probe_steps)
        Hs = [per_scale_H(tok[i], bnd, V) for i in range(len(seeds))]
        c = [criterion(h) for h in Hs]
        probes[round(float(T), 3)] = dict(
            schedule=[round(float(x), 4) for x in schedule(T)], criterion=c[0],
            gap={str(scales[k]): Hs[0][k] - H_data[k] for k in tri},
            H={str(scales[k]): Hs[0][k] for k in range(len(scales))},
            criterion_extra_seeds=c[1:], seconds=time.time() - t_start)
        if a.dump_tokens:
            np.savez_compressed(os.path.join(out, f"probe_T{T:.3f}.npz".replace(".", "p", 1)),
                                tokens=tok[0].astype(np.int16), T=T, schedule=schedule(T))
        print(f"  probe T={T:.3f} sched {np.round(schedule(T), 2).tolist()}: criterion {c[0]:+.4f} "
              f"(other seeds {' '.join(f'{x:+.3f}' for x in c[1:])}) gaps "
              + " ".join(f"s{scales[k]}={Hs[0][k] - H_data[k]:+.3f}" for k in tri)
              + f"  [{time.time() - t_start:.0f}s]", flush=True)
        return c[0]

    print(f"cell {a.cell}  rule {a.rule}  criterion {a.criterion}  target {a.target}  weights "
          f"{np.round(w, 3).tolist()}  read scales {[scales[k] for k in read_scales]}", flush=True)
    print("  targets " + " ".join(f"s{scales[k]}={H_data[k]:.3f}" for k in tri)
          + "   flip " + " ".join(f"s{scales[k]}={flip[k]:.2f}" for k in tri), flush=True)

    base = dict(cell=a.cell, vqvae=vq, rule=a.rule, criterion=a.criterion, target=a.target,
                scales=scales, trainable_scale_indices=tri, weights=[float(x) for x in w],
                read_scales=[scales[k] for k in read_scales], probe_steps=a.probe_steps,
                seed=a.seed, extra_seeds=seeds[1:], t0=a.t0, step=a.step,
                H_data_full={str(scales[k]): H_full[k] for k in range(len(scales))},
                H_data_window={str(scales[k]): H_win[k] for k in range(len(scales))},
                H_data_window_sd={str(scales[k]): H_win_sd[k] for k in range(len(scales))},
                flip_rate={str(scales[k]): flip[k] for k in range(len(scales))})

    if a.probe_T is not None:
        for T in a.probe_T:
            probe(T)
        base["probes"] = {f"{t:.3f}": v for t, v in sorted(probes.items())}
        json.dump(base, open(os.path.join(out, "probes_only.json"), "w"), indent=1)
        return

    # ---- walk (paper's search rule) ----
    T = a.t0
    g = probe(T)
    status, bracket = "ok", None
    if g == 0.0:
        T_star, bracket = T, [T, T]
    else:
        direction = +1 if g < 0 else -1
        T_prev, g_prev = T, g
        for _ in range(a.max_probes - 1):
            T = round(T_prev + direction * a.step, 3)
            if T < a.tmin - 1e-9 or T > a.tmax + 1e-9:
                status = "saturated_lo" if direction < 0 else "saturated_hi"
                break
            g = probe(T)
            if np.sign(g) != np.sign(g_prev):
                bracket = sorted([(T_prev, g_prev), (T, g)])
                break
            T_prev, g_prev = T, g
        if bracket is None:
            status = status if status != "ok" else "max_probes"
            T_star, bracket = T_prev, [T_prev, T_prev]
        else:
            (Tl, gl), (Th, gh) = bracket
            T_star = Tl + (Th - Tl) * (0.0 - gl) / (gh - gl)
            bracket = [Tl, Th]
    T_star = round(float(T_star), 3)
    sched = [round(float(x), 4) for x in schedule(T_star)]
    print(f"SOLVED {a.cell} [{tag}]: T*={T_star} [{status}] bracket {bracket}, {len(probes)} probes, "
          f"schedule {sched}", flush=True)

    # ---- long probe at the solved schedule: stationarity per scale ----
    drift = None
    if a.long_steps > a.probe_steps:
        t_start = time.time()
        tok = rollout(T_star, a.long_steps)
        drift = {}
        for k in tri:
            blk = tok[:, :, bnd[k]:bnd[k + 1]]
            d = [plug_in_H(blk[i], V) - plug_in_H(blk[i, :a.probe_steps], V) for i in range(len(seeds))]
            drift[str(scales[k])] = dict(
                D=d[0], D_extra_seeds=d[1:], H_all=plug_in_H(blk[0], V),
                H_first=plug_in_H(blk[0, :a.probe_steps], V),
                gap_all=plug_in_H(blk[0], V) - H_data[k])
        print(f"  long probe ({tok.shape[1]} steps, {time.time() - t_start:.0f}s): D = "
              + " ".join(f"s{s}={v['D']:+.3f}" for s, v in drift.items())
              + f" | third-finest scale, other seeds: "
              + " ".join(f"{x:+.3f}" for x in drift[str(scales[tri[-3]])]["D_extra_seeds"]), flush=True)

    rep = dict(base, T_star=T_star, status=status, bracket=bracket, n_probes=len(probes),
               schedule=sched, probes={f"{t:.3f}": v for t, v in sorted(probes.items())},
               long_steps=a.long_steps, drift=drift)
    json.dump(rep, open(os.path.join(out, "calibration.json"), "w"), indent=1)
    json.dump({"source": f"calibrate_schedule:{tag}", "T_star": T_star, "scales": scales,
               "first_trainable_scale": tri[0], "trainable_scale_indices": tri,
               "per_trainable_pooled_marginal_nats": sched,
               "per_trainable_mean_per_position_nats": sched},
              open(os.path.join(out, "anchor.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
