"""
Inference-time samplers for the NSP (next-scale prediction) rollout.

A single, jit-safe dispatch (`sample_scale`) over a frozen `SamplerConfig`.
The config is captured as a static closure at trace time (exactly like the
legacy `temperature` closure in `rollout_nsp.py`), so every branch on
`cfg.method` / `cfg.top_h_mode` / scalar hyperparameters resolves before jit
and only the selected branch is traced. The only *traced* per-call input that
varies with the sampler is `anchor` — a per-scale target entropy (in nats)
read off the frozen tokenizer (see `measure_tokenizer_entropy.py`).

All methods operate per-position over a fixed `(n_tokens_k, effective_vocab)`
logit block that arrives ALREADY support-masked (invalid entries == -1e9).
Set-selection methods build a boolean `keep` mask over the vocab and then call
`jax.random.categorical(key, where(keep, logits, -1e9))`: Gumbel-max sampling
is invariant to additive constants, so this is identical to sampling the
renormalized kept set, and support-masked tokens (logit -1e9) stay rejected
automatically.

Why these samplers (paper context):
    The NSP rollout collapses in two non-OOD ways. GREEDY / too-little
    randomness -> a low-variance "mean field". TOO-MUCH randomness -> an
    over-diffusive field (which spuriously flatters TKE spectral error, so
    EMD/PDF/OOD are the real metrics). A per-config temperature sweep finds the
    sweet spot between them; the goal here is a parameter-free, data-anchored
    sampler that lands in the non-collapsed range out of the box.

    Faithful Top-H truncation (arXiv:2509.02510) can only *lower* sampling
    entropy, so it guards the over-diffusive side but cannot escape mean-field
    collapse. The headline `entropy_target` homeostat solves a per-position
    temperature to hit a per-scale entropy target and so reaches the sweet spot
    from EITHER side. `typical` (arXiv:2202.00666) is a cheap baseline; `min_p`
    (arXiv:2407.01082) and `inverted_edt` are entropy-shrinking negative
    controls expected to deepen mean-field collapse.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# Large-negative sentinel matching the support-mask value used upstream
# (nsp_model.py applies `where(mask, logits, -1e9)`).
NEG = jnp.float32(-1e9)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class SamplerConfig:
    """Static sampler configuration.

    `frozen=True` makes instances hashable so JAX treats a closed-over config
    as a static argument. NEVER put arrays/lists here (they break hashing); the
    per-scale entropy anchor stays out of the config and is passed as a traced
    argument to `sample_scale`.
    """

    method: str = "ancestral"          # ancestral|entropy_target|top_h|typical|min_p|inverted_edt|data_mix

    # -- ancestral (legacy temperature / top_k / top_p; back-compat) --
    temperature: float = 0.0           # 0.0 => greedy argmax
    top_k: int = 0
    top_p: float = 1.0

    # -- top_h (arXiv:2509.02510) --
    base_temperature: float = 1.0      # optional pre-warm before the cap (warm->cap variant)
    top_h_alpha: float = 0.4           # relative bound: H(q) <= alpha * H(p)
    top_h_mode: str = "relative"       # "relative" (alpha*H(p)) | "absolute" (anchor)

    # -- entropy_target homeostat --
    entropy_tau_lo: float = 1e-2
    entropy_tau_hi: float = 1e2
    entropy_bisect_iters: int = 24

    # -- typical (arXiv:2202.00666) --
    typical_tau: float = 0.95

    # -- min_p (arXiv:2407.01082) --
    min_p: float = 0.05

    # -- inverted_edt (negative control) --
    inverted_edt_strength: float = 0.5
    inverted_edt_tau_min: float = 0.1

    # -- data_mix (convex mixture toward the per-scale data prior) --
    mix_gain: float = 1.0              # single global knob: gain on the confidence-deficit gate
    mix_lam_max: float = 1.0           # hard cap on injected fraction (1.0 = parameter-free)
    mix_eps: float = 1e-30


# =============================================================================
# Numerical helpers
# =============================================================================


def _safe_xlogx(p: jax.Array) -> jax.Array:
    """p * log(p) with the convention 0 * log(0) := 0 (no NaN)."""
    return jnp.where(p > 0, p * jnp.log(jnp.clip(p, 1e-30, 1.0)), 0.0)


def _entropy_from_probs(p: jax.Array, axis: int = -1) -> jax.Array:
    """Shannon entropy H(p) = -sum p log p, in nats."""
    return -jnp.sum(_safe_xlogx(p), axis=axis)


def _sample_kept(logits: jax.Array, keep: jax.Array, key: jax.Array) -> jax.Array:
    """Categorical sample restricted to `keep` (renormalized-equivalent)."""
    masked = jnp.where(keep, logits, NEG)
    return jax.random.categorical(key, masked, axis=-1).astype(jnp.int32)


# =============================================================================
# Methods
# =============================================================================


def _ancestral(logits: jax.Array, key: jax.Array, cfg: SamplerConfig) -> jax.Array:
    """Legacy path: greedy (T==0) else temperature + optional top_k / top_p.

    Behavior-preserving move of the original nsp_model.py sampling block, so
    existing greedy/warm rollouts reproduce bit-for-bit (same ops, same key).
    """
    if cfg.temperature == 0.0:                      # static branch
        return jnp.argmax(logits, axis=-1).astype(jnp.int32)

    if cfg.top_k > 0:                               # static branch
        top_vals, _ = jax.lax.top_k(logits, cfg.top_k)
        kth = top_vals[..., -1:]
        logits = jnp.where(logits < kth, NEG, logits)

    if cfg.top_p < 1.0:                             # static branch
        sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
        sorted_probs = jax.nn.softmax(sorted_logits / cfg.temperature, axis=-1)
        cumprobs = jnp.cumsum(sorted_probs, axis=-1)
        shifted = jnp.concatenate(
            [jnp.zeros_like(cumprobs[..., :1]), cumprobs[..., :-1]], axis=-1)
        keep = shifted < cfg.top_p
        kept = jnp.where(keep, sorted_logits, jnp.inf)
        threshold = jnp.min(kept, axis=-1, keepdims=True)
        logits = jnp.where(logits < threshold, NEG, logits)

    return jax.random.categorical(
        key, logits / cfg.temperature, axis=-1).astype(jnp.int32)


def _solve_temperature(logits: jax.Array, target: jax.Array,
                       cfg: SamplerConfig) -> jax.Array:
    """Per-position temperature tau s.t. H(softmax(logits/tau)) == target.

    Entropy is monotonically increasing in tau (tau->0 one-hot, H->0;
    tau->inf uniform-over-support, H->log|support|), so bisection on
    [tau_lo, tau_hi] is valid. Out-of-range targets self-clamp: a target above
    the achievable ceiling drives tau->tau_hi (warmest); below the floor drives
    tau->tau_lo (near-greedy).

    logits: (T, V) masked.  target: (T,).  returns tau: (T,).
    """
    T = logits.shape[0]
    target = jnp.broadcast_to(target, (T,))
    lo = jnp.full((T,), cfg.entropy_tau_lo, dtype=jnp.float32)
    hi = jnp.full((T,), cfg.entropy_tau_hi, dtype=jnp.float32)

    def _H_at(tau):
        p = jax.nn.softmax(logits / tau[:, None], axis=-1)
        return _entropy_from_probs(p, axis=-1)

    def body(_, carry):
        lo, hi = carry
        mid = 0.5 * (lo + hi)
        too_cold = _H_at(mid) < target            # entropy too low -> warm up
        lo = jnp.where(too_cold, mid, lo)
        hi = jnp.where(too_cold, hi, mid)
        return (lo, hi)

    lo, hi = jax.lax.fori_loop(0, cfg.entropy_bisect_iters, body, (lo, hi))
    return 0.5 * (lo + hi)


def _entropy_target(logits: jax.Array, key: jax.Array, anchor: jax.Array,
                    cfg: SamplerConfig) -> jax.Array:
    """Homeostat: warm OR cool each position to the per-scale target entropy."""
    assert anchor is not None, "entropy_target requires an anchor (per-scale target entropy)"
    tau = _solve_temperature(logits, jnp.asarray(anchor, jnp.float32), cfg)
    return jax.random.categorical(
        key, logits / tau[:, None], axis=-1).astype(jnp.int32)


def _top_h_keep(logits: jax.Array, bound: jax.Array) -> jax.Array:
    """Top-H keep mask: largest leading prefix (probs desc) with renormalized
    entropy <= bound. Returns a (T, V) boolean mask in ORIGINAL vocab order.

    Prefix-j renormalized entropy (verified): with M_j = sum_{<=j} p,
    S_j = sum_{<=j} p log p, the renormalized distribution q = p/M_j has
        H_j = log(M_j) - S_j / M_j.
    H_j is NOT monotone in j (a low-prob tail token can transiently lower it),
    so we keep the contiguous LEADING prefix via cumprod, not a bare threshold.
    """
    p = jax.nn.softmax(logits, axis=-1)                 # (T, V)
    order = jnp.argsort(-p, axis=-1)                    # descending
    ps = jnp.take_along_axis(p, order, axis=-1)
    M = jnp.cumsum(ps, axis=-1)
    S = jnp.cumsum(_safe_xlogx(ps), axis=-1)
    Mc = jnp.clip(M, 1e-30, 1.0)
    H_pref = jnp.log(Mc) - S / Mc                       # (T, V)

    ok = H_pref <= bound[:, None]
    ok = ok.at[:, 0].set(True)                          # always keep top-1
    cum_ok = jnp.cumprod(ok.astype(jnp.int32), axis=-1)  # 1 until first False
    n_keep = jnp.sum(cum_ok, axis=-1)                   # (T,) >= 1
    ranks = jnp.argsort(order, axis=-1)                 # vocab -> descending rank
    keep = (ranks < n_keep[:, None]) & (p > 0)
    return keep


def _top_h(logits: jax.Array, key: jax.Array, anchor: jax.Array,
           cfg: SamplerConfig) -> jax.Array:
    """Top-H bounded-entropy truncation (arXiv:2509.02510).

    Optional pre-warm by `base_temperature` (>1 warms) realizes the warm->cap
    variant: warm the distribution, then cap its entropy at the bound.
    """
    if cfg.base_temperature != 1.0:                     # static branch
        logits = logits / cfg.base_temperature
    p = jax.nn.softmax(logits, axis=-1)
    H_full = _entropy_from_probs(p, axis=-1)            # (T,)
    if cfg.top_h_mode == "absolute":                    # static branch
        assert anchor is not None, "top_h --top_h_mode absolute requires an anchor"
        bound = jnp.broadcast_to(jnp.asarray(anchor, jnp.float32), H_full.shape)
    else:                                               # relative
        bound = cfg.top_h_alpha * H_full
    keep = _top_h_keep(logits, bound)
    return _sample_kept(logits, keep, key)


def _typical(logits: jax.Array, key: jax.Array, cfg: SamplerConfig) -> jax.Array:
    """Locally Typical Sampling (arXiv:2202.00666).

    Keep tokens whose surprisal -log p is closest to the conditional entropy
    H(p), accumulating original mass until >= typical_tau.
    """
    p = jax.nn.softmax(logits, axis=-1)
    logp = jnp.log(jnp.clip(p, 1e-30, 1.0))
    H = _entropy_from_probs(p, axis=-1)                 # (T,)
    score = jnp.abs(-logp - H[:, None])                 # (T, V) ascending = typical
    order = jnp.argsort(score, axis=-1)
    ps = jnp.take_along_axis(p, order, axis=-1)
    cum = jnp.cumsum(ps, axis=-1)
    shifted = jnp.concatenate(
        [jnp.zeros_like(cum[:, :1]), cum[:, :-1]], axis=-1)
    keep_sorted = shifted < cfg.typical_tau
    keep_sorted = keep_sorted.at[:, 0].set(True)        # always keep most-typical
    ranks = jnp.argsort(order, axis=-1)
    keep = jnp.take_along_axis(keep_sorted, ranks, axis=-1) & (p > 0)
    return _sample_kept(logits, keep, key)


def _min_p(logits: jax.Array, key: jax.Array, cfg: SamplerConfig) -> jax.Array:
    """min-p sampling (arXiv:2407.01082) — negative control (confidence-concentrating)."""
    p = jax.nn.softmax(logits, axis=-1)
    pmax = jnp.max(p, axis=-1, keepdims=True)
    keep = p >= (cfg.min_p * pmax)                      # argmax always kept (min_p<=1)
    return _sample_kept(logits, keep, key)


def _inverted_edt(logits: jax.Array, key: jax.Array, cfg: SamplerConfig) -> jax.Array:
    """Inverted-polarity entropy-dynamic temperature — negative control.

    tau decreases with the position's entropy, so high-entropy positions are
    SHARPENED (entropy-shrinking) — drives toward the overconfident mean-field
    mode. Opposite polarity of EDT (arXiv:2403.14541).
    """
    V = logits.shape[-1]
    p = jax.nn.softmax(logits, axis=-1)
    H = _entropy_from_probs(p, axis=-1)                 # (T,)
    Hmax = jnp.log(jnp.float32(V))
    frac = jnp.clip(H / Hmax, 0.0, 1.0)
    tau = jnp.clip(1.0 - cfg.inverted_edt_strength * frac,
                   cfg.inverted_edt_tau_min, 1.0)       # (T,)
    return jax.random.categorical(
        key, logits / tau[:, None], axis=-1).astype(jnp.int32)


def _data_mix(logits, key, d_block, anchor, cfg):
    """Convex mixture toward the per-scale data prior, confidence-deficit gated.

    p_sample = (1 - lambda) * p_model + lambda * d_support,  where
      * d_support = the per-scale pooled data prior d_block restricted to THIS
        position's support and renormalized (ON-MANIFOLD by construction -- the
        injected codes are real codes the data uses at this scale, never the
        model's warmed tail), and
      * lambda = mix_gain * relu(H_cond - H_model) / H_cond, clipped to
        [0, mix_lam_max], where H_cond = anchor = the MODEL's teacher-forced
        per-scale CONDITIONAL entropy (the etmodel anchor, ~1 nat). The gate
        fires only where the rollout model has over-collapsed BELOW its own
        teacher-forced confidence -- so on a calibrated/cold config (sc341)
        H_model ~ H_cond -> lambda ~ 0 -> pure model (the known cold optimum),
        and it injects on-manifold diversity only where the model drifts into
        the confidence trap.

    Arithmetic (convex) mixing is deliberate: it is the unique blend whose
    entropy dials monotonically from the model conditional up to the data
    marginal (geometric/product blends saturate well below the warm target).
    """
    assert anchor is not None, "data_mix requires an anchor (per-scale conditional entropy)"
    assert d_block is not None, "data_mix requires d_block (per-scale data prior)"
    q = jax.nn.softmax(logits, axis=-1)                         # (T,V) p_model, masked
    # Restrict the per-scale prior to this position's applied support, renorm.
    smask = logits > (NEG * 0.5)                                # (T,V) bool support
    dr = jnp.where(smask, d_block, 0.0)
    dr = dr / jnp.clip(jnp.sum(dr, axis=-1, keepdims=True), cfg.mix_eps, None)
    Hq = _entropy_from_probs(q, axis=-1)                        # (T,) model entropy
    Htar = jnp.broadcast_to(jnp.asarray(anchor, jnp.float32), Hq.shape)
    excess = jax.nn.relu(Htar - Hq) / jnp.clip(Htar, cfg.mix_eps, None)   # (T,) in [0,1)
    lam = jnp.clip(cfg.mix_gain * excess, 0.0, cfg.mix_lam_max)[:, None]  # (T,1)
    ps = (1.0 - lam) * q + lam * dr                            # convex, on-support
    return jax.random.categorical(
        key, jnp.log(jnp.clip(ps, cfg.mix_eps, 1.0)), axis=-1).astype(jnp.int32)


# =============================================================================
# Dispatch
# =============================================================================


def sample_scale(logits: jax.Array, key: jax.Array, anchor, cfg: SamplerConfig,
                 d_block: jax.Array = None) -> jax.Array:
    """Sample one scale's tokens.

    Args:
        logits: (n_tokens_k, effective_vocab) float32, already support-masked
            (invalid entries == -1e9).
        key: PRNGKey, or None only when method=='ancestral' and temperature==0.
        anchor: scalar per-scale target entropy (nats) for entropy_target /
            top_h-absolute / data_mix (gate target); None otherwise.
        cfg: SamplerConfig (static).
        d_block: (n_tokens_k, effective_vocab) per-scale data prior for
            data_mix (the scale's pooled empirical code distribution, broadcast
            across the scale's rows); None otherwise.

    Returns:
        (n_tokens_k,) int32 predicted token ids — always in support.
    """
    m = cfg.method
    if m == "ancestral":
        return _ancestral(logits, key, cfg)
    if m == "entropy_target":
        return _entropy_target(logits, key, anchor, cfg)
    if m == "top_h":
        return _top_h(logits, key, anchor, cfg)
    if m == "typical":
        return _typical(logits, key, cfg)
    if m == "min_p":
        return _min_p(logits, key, cfg)
    if m == "inverted_edt":
        return _inverted_edt(logits, key, cfg)
    if m == "data_mix":
        return _data_mix(logits, key, d_block, anchor, cfg)
    raise ValueError(f"unknown sampler method: {m!r}")


# =============================================================================
# Self-test (run: python samplers.py)
# =============================================================================


def _selftest():
    import numpy as np

    key0 = jax.random.PRNGKey(0)
    T, V = 5, 8

    # Build random logits and mask a random in-support subset per row (>=2 kept).
    rng = np.random.default_rng(1)
    base = rng.normal(size=(T, V)).astype(np.float32)
    support = np.zeros((T, V), dtype=bool)
    for r in range(T):
        n_keep = rng.integers(2, V + 1)
        cols = rng.choice(V, size=n_keep, replace=False)
        support[r, cols] = True
    logits = jnp.asarray(np.where(support, base, -1e9))
    support_j = jnp.asarray(support)

    def assert_in_support(pred, name):
        ok = support_j[jnp.arange(T), pred]
        assert bool(jnp.all(ok)), f"{name}: produced out-of-support token: {pred}"

    anchor = jnp.float32(1.0)

    # (a) every method returns in-support tokens, across several keys.
    methods = {
        "ancestral_T0": SamplerConfig(method="ancestral", temperature=0.0),
        "ancestral_T1": SamplerConfig(method="ancestral", temperature=1.0),
        "ancestral_tkp": SamplerConfig(method="ancestral", temperature=1.0, top_k=4, top_p=0.9),
        "entropy_target": SamplerConfig(method="entropy_target"),
        "top_h_rel": SamplerConfig(method="top_h", top_h_mode="relative", top_h_alpha=0.4),
        "top_h_abs": SamplerConfig(method="top_h", top_h_mode="absolute"),
        "top_h_warmcap": SamplerConfig(method="top_h", top_h_mode="absolute", base_temperature=1.5),
        "typical": SamplerConfig(method="typical", typical_tau=0.9),
        "min_p": SamplerConfig(method="min_p", min_p=0.1),
        "inverted_edt": SamplerConfig(method="inverted_edt", inverted_edt_strength=0.7),
    }
    for name, cfg in methods.items():
        for s in range(4):
            k = jax.random.fold_in(key0, s)
            a = anchor if cfg.method in ("entropy_target",) or (
                cfg.method == "top_h" and cfg.top_h_mode == "absolute") else None
            pred = sample_scale(logits, k, a, cfg)
            assert_in_support(pred, name)
    print("[ok] all methods stay in support")

    # (b) ancestral T==0 == argmax; T==1 matches a hand-rolled reference.
    cfg0 = SamplerConfig(method="ancestral", temperature=0.0)
    assert bool(jnp.all(sample_scale(logits, None, None, cfg0)
                        == jnp.argmax(logits, axis=-1))), "greedy != argmax"
    cfg1 = SamplerConfig(method="ancestral", temperature=1.0)
    k = jax.random.fold_in(key0, 99)
    ref = jax.random.categorical(k, logits / 1.0, axis=-1).astype(jnp.int32)
    got = sample_scale(logits, k, None, cfg1)
    assert bool(jnp.all(ref == got)), "ancestral T=1 not byte-identical to reference"
    print("[ok] ancestral back-compat (argmax + reference categorical)")

    # (c) entropy_target hits the target within tolerance for in-range targets,
    #     and clamps gracefully out of range. Use smooth (unmasked) logits.
    smooth = jnp.asarray(rng.normal(size=(T, V)).astype(np.float32))
    cfg_et = SamplerConfig(method="entropy_target", entropy_bisect_iters=40)
    Hmax = float(np.log(V))
    for tgt in (0.3, 0.8, 1.5):
        tau = _solve_temperature(smooth, jnp.full((T,), tgt), cfg_et)
        Hsolved = _entropy_from_probs(jax.nn.softmax(smooth / tau[:, None], axis=-1), axis=-1)
        err = float(jnp.max(jnp.abs(Hsolved - tgt)))
        assert err < 1e-2, f"entropy_target miss: target={tgt} maxerr={err}"
    # over-large target -> tau ~ tau_hi, H ~ ceiling (< target)
    tau_hi = _solve_temperature(smooth, jnp.full((T,), Hmax + 5.0), cfg_et)
    assert float(jnp.min(tau_hi)) > 10.0, "over-large target did not push tau high"
    print(f"[ok] entropy_target hits target (<1e-2) and clamps; Hmax={Hmax:.3f}")

    # (d) top_h prefix-entropy formula matches a numpy brute force; kept-set
    #     entropy <= alpha*H(p).
    p = np.asarray(jax.nn.softmax(logits, axis=-1))
    alpha = 0.4
    bound = jnp.asarray(alpha * _entropy_from_probs(jnp.asarray(p), axis=-1))
    keep = np.asarray(_top_h_keep(logits, bound))
    for r in range(T):
        order = np.argsort(-p[r])
        # brute-force largest leading prefix with renorm entropy <= bound
        bnd = float(bound[r])
        best = 1
        for j in range(1, V + 1):
            sub = p[r][order[:j]]
            q = sub / sub.sum()
            q = q[q > 0]
            Hj = float(-(q * np.log(q)).sum())
            if Hj <= bnd + 1e-7:
                best = j
            else:
                break
        kept_cols = set(int(c) for c in order[:best] if p[r][int(c)] > 0)
        got_cols = set(int(c) for c in np.where(keep[r])[0])
        assert kept_cols == got_cols, f"top_h keep mismatch row {r}: {kept_cols} vs {got_cols}"
        # renormalized kept entropy <= bound
        sub = p[r][list(got_cols)]
        q = sub / sub.sum()
        Hkept = float(-(q * np.log(q)).sum())
        assert Hkept <= bnd + 1e-6, f"top_h kept entropy {Hkept} > bound {bnd}"
    print("[ok] top_h prefix-entropy matches brute force; kept entropy <= alpha*H(p)")

    # (e) jit (static cfg) + vmap consistency.
    cfg_v = SamplerConfig(method="entropy_target")
    batch_logits = jnp.stack([logits, logits * 0.5])
    keys = jax.random.split(key0, 2)

    @jax.jit
    def run(bl, ks):
        return jax.vmap(lambda l, k: sample_scale(l, k, anchor, cfg_v))(bl, ks)

    out_jv = run(batch_logits, keys)
    out_ref = jnp.stack([
        sample_scale(batch_logits[0], keys[0], anchor, cfg_v),
        sample_scale(batch_logits[1], keys[1], anchor, cfg_v),
    ])
    assert bool(jnp.all(out_jv == out_ref)), "jit+vmap differs from unjitted"
    for r in range(2):
        ok = support_j[jnp.arange(T), out_jv[r]]
        assert bool(jnp.all(ok)), "jit+vmap produced OOS token"
    print("[ok] jit + vmap consistent with unjitted path")

    # (f) data_mix: in-support; gate off (Hq>=Htar) => pure model; warm anchor
    #     => injects the data prior; lam grows as the model sharpens.
    d_block = jnp.where(support_j, 1.0, 0.0)
    d_block = d_block / jnp.sum(d_block, axis=-1, keepdims=True)   # uniform-on-support prior
    cfg_dm = SamplerConfig(method="data_mix", mix_gain=1.0)
    for s in range(4):
        k = jax.random.fold_in(key0, 700 + s)
        pred = sample_scale(logits, k, jnp.float32(0.5), cfg_dm, d_block=d_block)
        assert_in_support(pred, "data_mix")
    # gate-off: a target below the model entropy everywhere => lam==0 => matches
    # a pure-model categorical on the same key.
    low_anchor = jnp.float32(1e-6)
    k = jax.random.fold_in(key0, 4242)
    dm_off = sample_scale(logits, k, low_anchor, cfg_dm, d_block=d_block)
    model_only = jax.random.categorical(k, logits, axis=-1).astype(jnp.int32)
    assert bool(jnp.all(dm_off == model_only)), "data_mix lam==0 != pure model"
    # internal: lam is larger for a sharper (lower-entropy) row at a fixed warm anchor
    sharp = jnp.asarray(np.where(support, base * 8.0, -1e9), dtype=jnp.float32)
    Hsharp = _entropy_from_probs(jax.nn.softmax(sharp, -1), -1)
    Hsoft = _entropy_from_probs(jax.nn.softmax(logits, -1), -1)
    assert float(jnp.mean(Hsharp)) < float(jnp.mean(Hsoft)), "sanity: sharp< soft entropy"
    print("[ok] data_mix in-support; lam==0 recovers pure model; on-manifold by support")

    print("\nALL SAMPLER SELF-TESTS PASSED")


if __name__ == "__main__":
    _selftest()
