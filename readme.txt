ViT Autoencoder (vit_ae.py)
===========================
JAX/Equinox vision transformer autoencoder for 256x256 images.

Components
----------
_channel_norm(norm, x)
    Apply RMSNorm over channel dim of a (C, H, W) tensor.
    Reshapes to (H*W, C), vmaps RMSNorm, reshapes back.

EncoderStem(d, in_channels=1)
    (C, 256, 256) -> (d, 32, 32)
    3x strided Conv2d (stride 2, kernel 3, padding 1), each followed by SiLU + RMSNorm.
    Channel progression: C -> d//4 -> d//2 -> d.

DecoderStem(d, out_channels=1)
    (d, 32, 32) -> (C, 256, 256)
    3x strided ConvTranspose2d (stride 2, kernel 4, padding 1).
    First two: SiLU + RMSNorm. Final: no activation.
    Channel progression: d -> d//2 -> d//4 -> C.

_rope_freqs(dim, theta)
    Compute RoPE inverse frequencies: 1/(theta^(2i/dim)).

_apply_rope(x, freqs, pos)
    Apply 1D rotary position embedding to (N, dim) tensor given position indices.

_apply_2d_rope(q, k, H, W, theta)
    Apply 2D RoPE to queries and keys (N, n_heads, d_head).
    First half of head dim uses row positions, second half uses column positions.
    Vmapped over heads.

ViTBlock(d, n_heads, mlp_dim, rope_theta=32.0)
    Single pre-norm transformer block. Input/output: (N, d).
    Attention: RMSNorm -> fused QKV -> shared QK RMSNorm -> 2D RoPE ->
               dot_product_attention -> out proj + residual.
    FFN: RMSNorm -> SwiGLU (gate * silu(up) -> down) + residual.
    All projections bias-free.

Transformer(d, n_heads, mlp_dim, depth, rope_theta=32.0)
    (d, H, W) -> (d, H, W)
    Reshapes to (N, d), runs depth ViTBlocks, reshapes back.

Encoder(d, n_heads, mlp_dim, depth, codebook_dim, in_channels=1, rope_theta=32.0)
    (C, 256, 256) -> (codebook_dim, 32, 32)
    EncoderStem -> Transformer -> RMSNorm -> Linear(d, codebook_dim).

Decoder(d, n_heads, mlp_dim, depth, codebook_dim, out_channels=1, rope_theta=32.0)
    (codebook_dim, 32, 32) -> (C, 256, 256)
    Linear(codebook_dim, d) -> Transformer -> DecoderStem.

Multi-Scale VQ-VAE (vq.py)
==========================
VAR-style multi-scale residual vector quantization.

Components
----------
area_downsample(x, h, w)
    (D, H, W) -> (D, h, w) via reshape + mean. Integer scale factors only.

bicubic_upsample(x, h, w)
    (D, H_in, W_in) -> (D, h, w) via jax.image.resize(method="bicubic").

quantize(z_flat, codebook)
    (N, D), (K, D) -> (z_q, indices, commit_loss)
    Nearest-neighbor with one-hot matmul (sharding-safe) and STE.

MultiScaleVQ(codebook_dim, scales=(1,2,4,8,16,32))
    Residual quantization across scales. Per-scale phi Conv2d(D,D,3,p=1)
    + shared post_quant_conv. Returns partials at each scale for compound loss.
    Codebook passed as argument, not stored.

EMAState (NamedTuple)
    cluster_sizes (K,), codebook_sums (K, D), codebook (K, D).
    Standalone state, not inside any Module.

init_ema_state(codebook_dim, codebook_size)
    Initialize codebook from unit-normalized random normal.

ema_update(state, all_indices, all_z_flat, decay, *, key)
    EMA codebook update + Laplace smoothing + dead code reinitialization.
    Called OUTSIDE the gradient boundary.

vqvae_loss(model, codebook, x_batch, lambdas, beta)
    COMPOUND reconstruction loss: decoder called K times on progressively
    better latent approximations. Total = sum_k(lambda_k * MSE) + beta * commit.
    Compatible with eqx.filter_value_and_grad(has_aux=True).

Design choices
--------------
- RoPE theta=32: tuned for 32-position spatial axes (2D grid after 8x downsample).
- Kernel 4 on transposed convolutions: kernel 3 + stride 2 gives odd output dims.
- Single-input design: vmap over batch externally.
- Compound reconstruction loss: decoder runs at every scale (gust implementation lacks this).
- One-hot matmul for codebook lookups: sharding-safe, no fancy indexing.
- Scale loop is Python for (6 iters, unrolled): fori_loop lacks rev-mode grad, scan needs fixed shapes.
