"""Profile NSP memory for arbitrary scale config without needing a tokens file.

Synthesizes a minimal token_data dict with the right shapes so the model
factory can build the NSP without us having to actually tokenize a dataset.
Useful for comparing memory across scale configs (sc341/sc917/sc1941).

Usage:
    python scripts/profile_nsp_memory_mock.py \
        --scales 1,2,4,8,16,24,32 --effective_vocab 2500 \
        --n_layer 4 --n_embd 1024 --n_refine 2 --batch_size 16
"""
import argparse
import os
import sys
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import psutil
import jax
import jax.numpy as jnp
import equinox as eqx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scales", required=True,
                   help="Comma-separated scale sizes, e.g. 1,2,4,8,16,24,32")
    p.add_argument("--effective_vocab", type=int, default=2500)
    p.add_argument("--codebook_dim", type=int, default=512)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=1024)
    p.add_argument("--n_refine", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--context_drop_rate", type=float, default=0.25)
    p.add_argument("--measure_poll_ms", type=int, default=20)
    return p.parse_args()


class PeakRSSMonitor:
    def __init__(self, poll_ms: int = 20):
        self.poll_s = poll_ms / 1000.0
        self.peak = 0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process()

    def __enter__(self):
        self.peak = self._proc.memory_info().rss
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _run(self):
        while not self._stop.is_set():
            rss = self._proc.memory_info().rss
            if rss > self.peak:
                self.peak = rss
            time.sleep(self.poll_s)


def gb(n: int) -> float:
    return n / (1024 ** 3)


def main():
    args = parse_args()
    scales = [int(s) for s in args.scales.split(",")]
    n_scales = len(scales)
    tokens_per_frame = sum(s * s for s in scales)
    eff = args.effective_vocab
    cd = args.codebook_dim

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    print(f"[rss] startup:      {gb(rss0):.2f} GB")
    print(f"Scales: {scales}, tokens/frame: {tokens_per_frame}, "
          f"eff_vocab={eff}, codebook_dim={cd}")

    # Synthesize token_data with right shapes
    rng = np.random.default_rng(0)
    token_data = {
        "indices_flat": rng.integers(
            0, eff, size=(args.batch_size + 2, tokens_per_frame),
            dtype=np.int32),
        "scales": np.array(scales, dtype=np.int64),
        "effective_vocab_size": eff,
        "codebook_dim": cd,
        "codebook": rng.standard_normal((eff, cd)).astype(np.float32),
        "scale_masks": np.ones((n_scales, eff), dtype=bool),
        "first_trainable_scale": 1,
        "vocab_size": 4096,
        "old_to_new": np.arange(4096, dtype=np.int32),
        "new_to_old": np.arange(eff, dtype=np.int32),
    }

    from nsp_model import NSPConfig, create_nsp_model, build_teacher_forced_mask
    from train_nsp import make_compute_loss, make_train_step
    import optax

    config = NSPConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        n_refine_layers=args.n_refine,
    )
    key = jax.random.PRNGKey(0)
    model, exp_heads = create_nsp_model(token_data, config, key)
    print(f"rope_theta: {config.rope_theta}, refine: {config.n_refine_layers}")

    scales_t0 = config.scales
    scales_t1 = config.scales[:-1]
    tok_t0 = sum(h * w for h, w in scales_t0)
    tok_t1 = sum(h * w for h, w in scales_t1)
    pad0 = ((tok_t0 + 127) // 128) * 128
    pad1 = ((tok_t1 + 127) // 128) * 128
    print(f"Sequence: t0={tok_t0}->{pad0}, t1_trunc={tok_t1}->{pad1}, "
          f"total={pad0+pad1}")
    attn_bias = build_teacher_forced_mask(scales_t0, pad0, scales_t1, pad1)

    trainable = config.trainable_scale_indices
    import math as _math
    counts = [config.scales[i][0] * config.scales[i][1] for i in trainable]
    raw_w = [1.0 / _math.log(c + 1.0) for c in counts]
    mean_w = sum(raw_w) / len(raw_w)
    sw = {idx: w / mean_w for idx, w in zip(trainable, raw_w)}

    scale_masks = jnp.array(token_data["scale_masks"])
    compute_loss = make_compute_loss(
        config, scales_t0, pad0, scales_t1, pad1,
        attn_bias, sw, trainable, scale_masks,
        context_drop_rate=args.context_drop_rate,
    )
    train_step = make_train_step(compute_loss)

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(1e-4, weight_decay=1e-4),
    )
    params = eqx.filter((model, exp_heads), eqx.is_inexact_array)
    opt_state = optimizer.init(params)

    B = args.batch_size
    indices = token_data["indices_flat"]
    t0 = indices[:B]
    t1 = indices[1:B + 1]
    batch = jnp.array(np.concatenate([t0, t1], axis=1))
    step_key = jax.random.PRNGKey(1)

    rss_init = proc.memory_info().rss
    print(f"[rss] after init:   {gb(rss_init):.2f} GB")

    print("Compiling train step...")
    t_compile = time.time()
    with PeakRSSMonitor(args.measure_poll_ms) as mon:
        model, exp_heads, opt_state, loss, metrics = train_step(
            model, exp_heads, opt_state, batch, optimizer, step_key)
        loss.block_until_ready()
    print(f"Compile:    {time.time()-t_compile:.1f}s, "
          f"peak RSS {gb(mon.peak):.2f} GB")

    rss_pre = proc.memory_info().rss
    print(f"[rss] before step:  {gb(rss_pre):.2f} GB")
    t_step = time.time()
    with PeakRSSMonitor(args.measure_poll_ms) as mon:
        model, exp_heads, opt_state, loss, metrics = train_step(
            model, exp_heads, opt_state, batch, optimizer,
            jax.random.fold_in(step_key, 1))
        loss.block_until_ready()
    step_peak = mon.peak
    print(f"Step:       {time.time()-t_step:.1f}s, "
          f"peak RSS {gb(step_peak):.2f} GB")
    print(f"Step delta: {gb(step_peak - rss_pre):.2f} GB above pre-step")

    print("\n=== SUMMARY ===")
    print(f"scales={scales} tokens/frame={tokens_per_frame}")
    print(f"n_layer={args.n_layer} n_embd={args.n_embd} "
          f"n_refine={args.n_refine} batch={args.batch_size}")
    print(f"compile_peak_gb={gb(mon.peak):.3f}")
    print(f"step_peak_gb={gb(step_peak):.3f}")


if __name__ == "__main__":
    main()
