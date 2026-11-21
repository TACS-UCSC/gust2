"""test_generate_cached.py — equivalence of nsp_model.generate_t1_frame_cached with
generate_t1_frame on a real checkpoint.

Greedy decoding with log_topk makes both paths deterministic and exposes the pre-sampling
logits, so the comparison involves no sampling noise:
  * emitted tokens must agree (argmax ties aside);
  * the top-K logits must agree to float tolerance wherever the top-K indices agree.
A sampled check (same keys) is also reported; it is expected to agree on almost every token
of a step, not on all of them, because a float-level logit difference can flip a draw.

    python test_generate_cached.py --checkpoint_dir <ar>/<cell> --tokens_path <val.npz> \
        --train_tokens_path <train.npz> [--frames 0 500 1500] [--temperature 2.0]
"""
import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from nsp_model import (NSPConfig, build_teacher_forced_mask, create_nsp_model,
                       generate_t1_frame, generate_t1_frame_cached)
from samplers import SamplerConfig
from tokenizer import load_tokenized_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True)
    ap.add_argument("--tokens_path", required=True)
    ap.add_argument("--train_tokens_path", required=True)
    ap.add_argument("--frames", type=int, nargs="+", default=[0, 500, 1500])
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--x64", action="store_true",
                    help="run both paths with float64 weights and activations; a difference that "
                         "is only float rounding shrinks by orders of magnitude, a real one does not")
    a = ap.parse_args()

    td = load_tokenized_data(a.tokens_path)
    indices = td["indices_flat"]
    scale_masks = jnp.array(td["scale_masks"])
    tr = np.load(a.train_tokens_path)
    P, V = tr["indices_flat"].shape[1], int(tr["effective_vocab_size"])
    M = np.zeros((P, V), dtype=bool)
    M[np.broadcast_to(np.arange(P), tr["indices_flat"].shape).ravel(),
      tr["indices_flat"].ravel().astype(np.int64)] = True
    position_mask = jnp.array(M)

    arch = json.load(open(os.path.join(a.checkpoint_dir, "training_state.json")))["arch_config"]
    config = NSPConfig(n_layer=arch["n_layer"], n_head=arch["n_head"], n_embd=arch["n_embd"],
                       dropout=0.0, rope_theta=arch["rope_theta"],
                       n_refine_layers=arch["n_refine_layers"])
    model, heads = create_nsp_model(td, config, jax.random.PRNGKey(0))
    model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(
        os.path.join(a.checkpoint_dir, "model.eqx"), model))
    heads = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(
        os.path.join(a.checkpoint_dir, "exp_heads.eqx"), heads))

    if a.x64:      # enable after loading: the checkpoint on disk is float32
        jax.config.update("jax_enable_x64", True)
        up = lambda t: jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if eqx.is_inexact_array(x) else x, t)
        model, heads = up(model), up(heads)
    tri = config.trainable_scale_indices
    s0, s1 = config.scales, config.scales[:-1]
    n0, n1 = sum(h * w for h, w in s0), sum(h * w for h, w in s1)
    p0, p1 = ((n0 + 127) // 128) * 128, ((n1 + 127) // 128) * 128
    bias = build_teacher_forced_mask(s0, p0, s1, p1)
    bnd = config.scale_boundaries

    def make(fn, cfg, topk):
        @jax.jit
        def f(t0, key):
            return fn(model, heads, config, t0, s0, p0, s1, p1, bias, scale_masks, tri, key,
                      0.0, 0, 1.0, topk, position_mask=position_mask, sampler_cfg=cfg)
        return f

    greedy = SamplerConfig(method="ancestral", temperature=0.0, top_k=0, top_p=1.0)
    hot = SamplerConfig(method="ancestral", temperature=a.temperature, top_k=0, top_p=1.0)
    old_g, new_g = make(generate_t1_frame, greedy, a.topk), make(generate_t1_frame_cached, greedy, a.topk)
    old_s, new_s = make(generate_t1_frame, hot, 0), make(generate_t1_frame_cached, hot, 0)

    worst = 0.0
    for fr in a.frames:
        t0 = jnp.asarray(indices[fr], dtype=jnp.int32)
        key = jax.random.PRNGKey(1000 + fr)
        to, lo, io = old_g(t0, key)
        tn, ln, i_n = new_g(t0, key)
        to, tn, lo, ln, io, i_n = map(np.asarray, (to, tn, lo, ln, io, i_n))
        print(f"frame {fr}: greedy tokens agree {np.mean(to == tn):.6f} ({int((to != tn).sum())} differ of {to.size})")
        for k in tri:
            sl = slice(bnd[k], bnd[k + 1])
            same = io[sl] == i_n[sl]
            # only compare logits of real (unmasked) candidates
            real = (lo[sl] > -1e8) & (ln[sl] > -1e8) & same
            d = np.abs(lo[sl] - ln[sl])[real]
            h, w = config.scales[k]
            mx = float(d.max()) if d.size else float("nan")
            worst = max(worst, mx if d.size else 0.0)
            print(f"   scale {h:2d}x{w:<2d}: tokens agree {np.mean(to[sl] == tn[sl]):.4f}, top-{a.topk} "
                  f"index agree {same.mean():.4f}, max |dlogit| {mx:.2e}, logit scale {np.abs(lo[sl][real]).mean():.2f}")
        so, sn = np.asarray(old_s(t0, key)), np.asarray(new_s(t0, key))
        print(f"   sampled T={a.temperature}: tokens agree {np.mean(so == sn):.4f}; per scale "
              + ", ".join(f"{np.mean(so[bnd[k]:bnd[k+1]] == sn[bnd[k]:bnd[k+1]]):.3f}" for k in tri))

    t0 = jnp.asarray(indices[a.frames[0]], dtype=jnp.int32)
    key = jax.random.PRNGKey(0)
    for name, f in (("old", old_s), ("cached", new_s)):
        f(t0, key).block_until_ready()
        t = time.time()
        for _ in range(5):
            f(t0, key).block_until_ready()
        print(f"single-trajectory step time, {name}: {(time.time() - t) / 5 * 1e3:.1f} ms")
    print(f"WORST max |dlogit| {worst:.3e}")


if __name__ == "__main__":
    main()
