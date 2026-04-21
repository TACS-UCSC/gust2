"""Profile peak memory of one NSP training step on CPU.

Runs a single (n_layer, n_embd, n_refine, batch_size) config through one
compile + one measured train step on JAX CPU, and reports peak RSS.

Usage (direct):
    python scripts/profile_nsp_memory.py --n_layer 4 --n_embd 1024 \
        --n_refine 2 --batch_size 32 --tokens_path experiments/tokens/small-sc917.npz

Run per-config via subprocess so memory baselines are clean; see the
__main__ block of profile_nsp_sweep.py for the driver pattern.
"""
import argparse
import os
import sys
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

# Run from repo root; add it to sys.path for imports
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
    p.add_argument("--tokens_path", required=True)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--n_embd", type=int, default=1024)
    p.add_argument("--n_refine", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--context_drop_rate", type=float, default=0.1)
    p.add_argument("--measure_poll_ms", type=int, default=20)
    return p.parse_args()


class PeakRSSMonitor:
    """Background thread that polls Process.memory_info().rss."""

    def __init__(self, poll_ms: int = 20):
        self.poll_s = poll_ms / 1000.0
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
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
    return n / (1024**3)


def main():
    args = parse_args()

    # Deferred imports so JAX_PLATFORMS env is honored
    from nsp_model import NSPConfig, create_nsp_model, build_teacher_forced_mask
    from train_nsp import make_compute_loss, make_train_step
    from tokenizer import load_tokenized_data

    import optax

    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    print(f"[rss] startup:      {gb(rss0):.2f} GB")

    token_data = load_tokenized_data(args.tokens_path)
    # Drop the fully-expanded vectors_flat (4992 × 917 × codebook_dim float32 ≈ 9 GB
    # for sc917) — training only needs token indices. Also trim indices_flat to
    # the minimum the profiler needs so we don't hold the full dataset.
    token_data.pop("vectors_flat", None)
    keep = args.batch_size + 2
    token_data["indices_flat"] = np.ascontiguousarray(
        token_data["indices_flat"][:keep])
    import gc; gc.collect()
    indices = token_data["indices_flat"]
    scale_masks = jnp.array(token_data["scale_masks"])

    config = NSPConfig(
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        n_refine_layers=args.n_refine,
    )
    key = jax.random.PRNGKey(0)
    model, exp_heads = create_nsp_model(token_data, config, key)
    print(f"Model:      {model.get_num_params()/1e6:.2f} M")
    print(f"ExpHeads:   {exp_heads.get_num_params()/1e6:.2f} M")
    print(f"rope_theta: {config.rope_theta}, refine: {config.n_refine_layers}")

    scales_t0 = config.scales
    scales_t1 = config.scales[:-1]
    tok_t0 = sum(h * w for h, w in scales_t0)
    tok_t1 = sum(h * w for h, w in scales_t1)
    pad0 = ((tok_t0 + 127) // 128) * 128
    pad1 = ((tok_t1 + 127) // 128) * 128
    attn_bias = build_teacher_forced_mask(scales_t0, pad0, scales_t1, pad1)

    trainable = config.trainable_scale_indices
    import math as _math
    token_counts = [config.scales[i][0] * config.scales[i][1] for i in trainable]
    raw_w = [1.0 / _math.log(c + 1.0) for c in token_counts]
    mean_w = sum(raw_w) / len(raw_w)
    scale_weights = {idx: w / mean_w for idx, w in zip(trainable, raw_w)}

    compute_loss = make_compute_loss(
        config, scales_t0, pad0, scales_t1, pad1,
        attn_bias, scale_weights, trainable, scale_masks,
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
    t0 = indices[:B]
    t1 = indices[1:B + 1]
    batch = jnp.array(np.concatenate([t0, t1], axis=1))
    step_key = jax.random.PRNGKey(1)

    rss_after_init = proc.memory_info().rss
    print(f"[rss] after init:   {gb(rss_after_init):.2f} GB "
          f"(+{gb(rss_after_init - rss0):.2f} from startup)")

    # Warmup (compile)
    print("Compiling train step...")
    t_compile = time.time()
    with PeakRSSMonitor(args.measure_poll_ms) as mon:
        model, exp_heads, opt_state, loss, metrics = train_step(
            model, exp_heads, opt_state, batch, optimizer, step_key)
        loss.block_until_ready()
    compile_peak = mon.peak
    compile_s = time.time() - t_compile
    print(f"Compile:    {compile_s:.1f}s, peak RSS {gb(compile_peak):.2f} GB")

    # Steady-state step
    rss_before = proc.memory_info().rss
    print(f"[rss] before step:  {gb(rss_before):.2f} GB")
    t_step = time.time()
    with PeakRSSMonitor(args.measure_poll_ms) as mon:
        model, exp_heads, opt_state, loss, metrics = train_step(
            model, exp_heads, opt_state, batch, optimizer,
            jax.random.fold_in(step_key, 1))
        loss.block_until_ready()
    step_peak = mon.peak
    step_s = time.time() - t_step
    print(f"Step:       {step_s:.1f}s, peak RSS {gb(step_peak):.2f} GB")
    print(f"Step delta: {gb(step_peak - rss_before):.2f} GB above pre-step baseline")

    print("\n=== SUMMARY ===")
    print(f"n_layer={args.n_layer} n_embd={args.n_embd} "
          f"n_refine={args.n_refine} batch={args.batch_size}")
    print(f"loss={float(loss):.4f}")
    print(f"compile_peak_gb={gb(compile_peak):.3f}")
    print(f"step_peak_gb={gb(step_peak):.3f}")
    print(f"compile_s={compile_s:.1f} step_s={step_s:.1f}")


if __name__ == "__main__":
    main()
