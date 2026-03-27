# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-scale VQ-VAE (Vector Quantized Variational Autoencoder) for 256x256 single-channel images, built on JAX and Equinox. Uses a Vision Transformer backbone with VAR-style multi-scale residual vector quantization. Data is loaded from HDF5 files containing scientific fields (e.g. vorticity "omega").

## Dependencies

```
uv pip install 'jax[cuda13]' equinox optax h5py matplotlib wandb
```

## Deployment: Bridges2 (PSC)

- **Account**: `mth260004p`
- **Code**: `/ocean/projects/mth260004p/sambamur/gust` (alias: `gust`)
- **Data**: `/ocean/projects/mth260004p/sambamur/data_lowres/output.h5`
- **All storage**: under `/ocean/projects/mth260004p/sambamur/` — see Experiments layout below
- **GPUs**: H100-80GB. GPU-shared partition (1-4 GPUs), GPU partition (8 GPUs).
- **Scheduler**: Slurm. Max walltime 2 days. See `scripts/` for job templates.

## Experiments Layout

All experiment artifacts live on Ocean storage. Naming uses token count (sc341/sc917/sc1941) for unambiguous identification across the pipeline.

```
/ocean/projects/mth260004p/sambamur/
├── gust/                              # code repo
├── data_lowres/output.h5              # raw turbulence data (20k × 256×256)
├── wandb/                             # wandb logs (all projects)
├── sweeps/codebook/                   # codebook hyperparam sweep (Phase 1)
└── experiments/
    ├── vqvae/                         # VQ-VAE checkpoints
    │   ├── small-sc341/               # D=5, scales 1,2,4,8,16
    │   ├── small-sc917/               # D=5, scales 1,2,4,8,16,24
    │   ├── small-sc1941/              # D=5, scales 1,2,4,8,16,24,32
    │   ├── medium-sc*/                # D=10 (future)
    │   └── large-sc*/                 # D=20 (future)
    ├── tokens/                        # tokenized datasets (.npz)
    │   ├── small-sc341.npz
    │   ├── small-sc917.npz
    │   └── small-sc1941.npz
    └── ar/                            # AR pushforward models (future)
        ├── small-sc341-ar*/
        └── ...
```

### Scale Configurations

| Config | Scales | Tokens/sample |
|--------|--------|---------------|
| sc341 | 1, 2, 4, 8, 16 | 341 |
| sc917 | 1, 2, 4, 8, 16, 24 | 917 |
| sc1941 | 1, 2, 4, 8, 16, 24, 32 | 1,941 |

### Wandb Projects

- `gust2-vqvae`: codebook hyperparameter sweeps
- `gust2-experiments`: final training runs (groups: small, medium, large)

### Best Codebook Config (from sweep)

`codebook_dim=512, codebook_size=4096, beta=0.25, ema_decay=0.90`

## Mixed Precision

Master weights in f32. Forward/backward in bf16 via `_cast_to_half()` in `train.py` (encoder+decoder cast to bf16, VQ stays f32). RMSNorm always upcasts to f32. Attention receives bf16 q/k/v for cuDNN flash attention dispatch.

## Model Sizes (d=512, mlp=1024, heads=8, batch=64)

| Model | enc/dec depth | Params | GPUs | Peak/GPU |
|-------|--------------|--------|------|----------|
| Small | 5 / 5 | 31M | 1 | ~52 GB |
| Medium | 10 / 10 | 57M | 2 | ~51 GB |
| Large | 20 / 20 | 109M | 4 | ~51 GB |

## Architecture

Three modules compose the full model, passed as a `(encoder, decoder, vq)` tuple:

- **`vit_ae.py`** — ViT autoencoder. `Encoder` maps `(C, 256, 256) → (codebook_dim, 32, 32)` via a strided-conv stem (8× spatial downsample) then transformer blocks. `Decoder` reverses with transposed convs. All normalization is RMSNorm; attention uses 2D RoPE (theta=32, tuned for 32-position spatial grid). FFN is SwiGLU. All projections are bias-free.

- **`vq.py`** — Multi-scale residual VQ + loss. `MultiScaleVQ` quantizes the encoder's latent at 6 scales (1→32) against a **shared codebook passed as an argument** (not stored in the module). `EMAState` is a standalone `NamedTuple` managing codebook updates — `ema_update()` must be called **outside** the gradient boundary. Two loss functions: `vqvae_loss` (compound: decoder called at every scale) and `vqvae_loss_simple` (decoder called once on final latent).

- **`dataloader.py`** — `VQVAEDataset` reads HDF5 from `fields/<field_name>`, adds a channel dim, and yields SPMD-sharded batches via a JAX device mesh.

## Key Design Decisions

- **Single-sample API**: all modules process one sample; batch via `jax.vmap` externally.
- **Codebook is external state**: passed to `MultiScaleVQ.__call__()` and loss functions, not stored in any `eqx.Module`. This keeps it outside `filter_grad`.
- **One-hot matmul for codebook lookups**: avoids fancy indexing for sharding safety.
- **Scale loop is unrolled Python `for`** (6 iterations): `lax.fori_loop` lacks rev-mode grad support, `lax.scan` requires fixed shapes across scales.
- **Kernel 4 on transposed convolutions**: kernel 3 + stride 2 produces odd output dims.
- **Compound loss** (`vqvae_loss`): decoder runs K times on progressively refined latents — distinguishes this from the earlier "gust" implementation.
