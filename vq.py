import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple


# ---------- Helpers ----------


def area_downsample(x: jax.Array, h: int, w: int) -> jax.Array:
    """Downsample (D, H, W) -> (D, h, w) via area averaging.

    Reshapes one axis at a time so automatic sharding (set_mesh) can
    propagate without hitting the multi-axis split restriction.
    """
    D, H, W = x.shape
    x = x.reshape(D, h, H // h, W).mean(axis=2)     # (D, h, W)
    x = x.reshape(D, h, w, W // w).mean(axis=3)     # (D, h, w)
    return x


def bicubic_upsample(x: jax.Array, h: int, w: int) -> jax.Array:
    """Upsample (D, H_in, W_in) -> (D, h, w) via bicubic interpolation."""
    return jax.image.resize(x, (x.shape[0], h, w), method="bicubic")


# ---------- Core Quantization ----------


def quantize(z_flat: jax.Array, codebook: jax.Array
             ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Nearest-neighbor quantize with straight-through estimator.

    Args:
        z_flat: (N, D) flattened spatial residual
        codebook: (K, D) shared codebook

    Returns:
        z_q: (N, D) quantized with STE gradient
        indices: (N,) integer codebook indices
        commit_loss: scalar
    """
    # ||z - e||² = ||z||² - 2z·e + ||e||²  (skip ||z||², doesn't affect argmin)
    dists = -2.0 * (z_flat @ codebook.T) + jnp.sum(codebook ** 2, axis=1)
    indices = jnp.argmin(dists, axis=1)

    # One-hot matmul lookup (sharding-safe, no fancy indexing)
    K = codebook.shape[0]
    z_q_raw = jax.nn.one_hot(indices, K) @ codebook

    # STE: forward uses z_q_raw, backward passes gradient straight to z_flat
    z_q = z_flat + jax.lax.stop_gradient(z_q_raw - z_flat)

    # Commitment loss: push encoder toward codebook entries
    commit_loss = jnp.mean((z_flat - jax.lax.stop_gradient(z_q_raw)) ** 2)

    return z_q, indices, commit_loss


# ---------- Multi-Scale VQ Module ----------


class MultiScaleVQ(eqx.Module):
    """Multi-scale residual vector quantization (VAR-style).

    Contains per-scale phi convolutions and a shared post-quantization conv.
    The codebook is passed as an argument, NOT stored here.
    """
    phi_convs: list
    post_quant_conv: eqx.nn.Conv2d
    scales: tuple[int, ...] = eqx.field(static=True)
    full_size: int = eqx.field(static=True)

    def __init__(self, codebook_dim: int,
                 scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
                 full_size: int = 32, *, key: jax.Array):
        keys = jax.random.split(key, len(scales) + 1)
        self.phi_convs = [
            eqx.nn.Conv2d(codebook_dim, codebook_dim, 3, padding=1, key=keys[i])
            for i in range(len(scales))
        ]
        self.post_quant_conv = eqx.nn.Conv2d(
            codebook_dim, codebook_dim, 3, padding=1, key=keys[-1]
        )
        self.scales = scales
        self.full_size = full_size

    def __call__(self, z_e: jax.Array, codebook: jax.Array):
        """Multi-scale residual quantization.

        Args:
            z_e: (D, 32, 32) encoder output
            codebook: (K, D) shared codebook

        Returns:
            z_q: (D, 32, 32) final quantized latent (for decoder)
            all_indices: list of (s_k²,) per-scale codebook indices
            partials: list of (D, 32, 32) per-scale partial reconstructions
            commit_loss: scalar total commitment loss
            all_z_flat: list of (s_k², D) pre-quantization residuals (for EMA)
        """
        D = z_e.shape[0]
        f_hat = jnp.zeros((D, 1, 1))
        partials = []
        all_indices = []
        all_z_flat = []
        commit_loss = 0.0

        for k, s in enumerate(self.scales):
            f_hat = bicubic_upsample(f_hat, s, s)
            res = area_downsample(z_e, s, s) - f_hat
            res_flat = res.reshape(D, s * s).T                   # (s², D)

            z_q_flat, indices, commit = quantize(res_flat, codebook)
            z_q = z_q_flat.T.reshape(D, s, s)
            z_q = self.phi_convs[k](z_q)
            f_hat = f_hat + z_q

            partial = bicubic_upsample(f_hat, self.full_size, self.full_size)
            partial = self.post_quant_conv(partial)
            partials.append(partial)

            all_indices.append(indices)
            all_z_flat.append(jax.lax.stop_gradient(res_flat))
            commit_loss = commit_loss + commit

        return partials[-1], all_indices, partials, commit_loss, all_z_flat


# ---------- EMA State ----------


class EMAState(NamedTuple):
    cluster_sizes: jax.Array    # (K,)
    codebook_sums: jax.Array    # (K, D)
    codebook: jax.Array         # (K, D)


def init_ema_state(codebook_dim: int, codebook_size: int,
                   *, key: jax.Array) -> EMAState:
    """Initialize codebook and EMA tracking state."""
    codebook = jax.random.normal(key, (codebook_size, codebook_dim))
    codebook = codebook / jnp.linalg.norm(codebook, axis=1, keepdims=True)
    return EMAState(
        cluster_sizes=jnp.ones(codebook_size),
        codebook_sums=codebook.copy(),
        codebook=codebook,
    )


def ema_update(state: EMAState,
               all_indices: list[jax.Array],
               all_z_flat: list[jax.Array],
               decay: float = 0.99,
               epsilon: float = 1e-5,
               dead_threshold: float = 0.25,
               *, key: jax.Array) -> EMAState:
    """EMA codebook update with Laplace smoothing and dead code reinitialization.

    Must be called OUTSIDE the gradient boundary.

    Accepts batched inputs: all_indices is a list of (..., N_k) arrays
    and all_z_flat is a list of (..., N_k, D) arrays.  Leading dims
    (e.g. batch) are reduced via sum, which is compatible with SPMD
    sharding (the reduction handles the all-reduce automatically).

    Args:
        state: current EMA state
        all_indices: list of (..., N_k) per-scale indices
        all_z_flat: list of (..., N_k, D) per-scale residuals
    """
    K, D = state.codebook.shape

    # Aggregate usage counts and vector sums across all scales.
    # Reduce spatial first (unsharded), then batch (sharded → all-reduce).
    new_counts = jnp.zeros(K)
    new_sums = jnp.zeros((K, D))
    total_vectors = 0
    for indices, z_flat in zip(all_indices, all_z_flat):
        oh = jax.nn.one_hot(indices, K)                   # (B, N_k, K)
        per_sample_counts = oh.sum(axis=-2)               # (B, K)
        new_counts = new_counts + per_sample_counts.sum(axis=0)  # (K,)
        per_sample_sums = jnp.einsum('bnk,bnd->bkd', oh, z_flat) # (B, K, D)
        new_sums = new_sums + per_sample_sums.sum(axis=0)        # (K, D)
        total_vectors = total_vectors + indices.size

    # EMA smooth
    cluster_sizes = decay * state.cluster_sizes + (1.0 - decay) * new_counts
    codebook_sums = decay * state.codebook_sums + (1.0 - decay) * new_sums

    # Laplace smoothing + normalize to get codebook vectors
    n = cluster_sizes.sum()
    sizes_smooth = (cluster_sizes + epsilon) / (n + K * epsilon) * n
    codebook = codebook_sums / sizes_smooth[:, None]

    # Dead code reinitialization (random normal, sharding-safe)
    uniform_usage = total_vectors / K
    dead_mask = cluster_sizes < dead_threshold * uniform_usage

    replacements = jax.random.normal(key, (K, D))
    replacements = replacements / jnp.linalg.norm(replacements, axis=1, keepdims=True)

    codebook = jnp.where(dead_mask[:, None], replacements, codebook)
    cluster_sizes = jnp.where(dead_mask, uniform_usage, cluster_sizes)
    codebook_sums = jnp.where(dead_mask[:, None],
                               replacements * uniform_usage, codebook_sums)

    return EMAState(cluster_sizes, codebook_sums, codebook)


# ---------- Loss Function ----------


def vqvae_loss(model: tuple, codebook: jax.Array,
               x_batch: jax.Array, lambdas: jax.Array,
               beta: float = 0.25):
    """Compound VQ-VAE loss with multi-scale reconstruction.

    The decoder is called K times — once per scale — on progressively
    better latent approximations, giving gradient signal at every scale.
    This is the key difference from the gust implementation.

    Args:
        model: (encoder, decoder, vq) tuple of eqx.Modules
        codebook: (K, D) from EMAState (no gradients flow here)
        x_batch: (B, C, 256, 256) input batch
        lambdas: (n_scales,) per-scale weights (typically increasing)
        beta: commitment loss weight

    Returns:
        total_loss: scalar
        aux: dict with recon_loss, commit_loss, all_indices, all_z_flat
    """
    encoder, decoder, vq = model

    def forward_single(x):
        z_e = encoder(x)                                        # matches model dtype
        return vq(z_e.astype(jnp.float32), codebook)            # VQ always f32

    z_q, all_indices, partials, commit, all_z_flat = jax.vmap(
        forward_single
    )(x_batch)

    # Compound reconstruction loss: decode each scale's partial
    recon_loss = 0.0
    for k in range(len(partials)):
        y_k = jax.vmap(decoder)(partials[k].astype(x_batch.dtype))
        recon_loss = recon_loss + lambdas[k] * jnp.mean(
            (x_batch.astype(jnp.float32) - y_k.astype(jnp.float32)) ** 2)

    commit_loss = jnp.mean(commit)
    total = recon_loss + beta * commit_loss

    return total, {
        'recon_loss': recon_loss,
        'commit_loss': commit_loss,
        'all_indices': all_indices,
        'all_z_flat': all_z_flat,
    }


def vqvae_loss_simple(model: tuple, codebook: jax.Array,
                      x_batch: jax.Array, beta: float = 0.25):
    """Single-reconstruction VQ-VAE loss. Decoder called once on final quantized latent.

    Args:
        model: (encoder, decoder, vq) tuple of eqx.Modules
        codebook: (K, D) from EMAState (no gradients flow here)
        x_batch: (B, C, 256, 256) input batch
        beta: commitment loss weight

    Returns:
        total_loss: scalar
        aux: dict with recon_loss, commit_loss, all_indices, all_z_flat
    """
    encoder, decoder, vq = model

    def forward_single(x):
        z_e = encoder(x)                                        # matches model dtype
        return vq(z_e.astype(jnp.float32), codebook)            # VQ always f32

    z_q, all_indices, _partials, commit, all_z_flat = jax.vmap(
        forward_single
    )(x_batch)

    y = jax.vmap(decoder)(z_q.astype(x_batch.dtype))            # decoder matches input
    recon_loss = jnp.mean((x_batch.astype(jnp.float32)
                           - y.astype(jnp.float32)) ** 2)       # loss always f32
    commit_loss = jnp.mean(commit)
    total = recon_loss + beta * commit_loss

    return total, {
        'recon_loss': recon_loss,
        'commit_loss': commit_loss,
        'all_indices': all_indices,
        'all_z_flat': all_z_flat,
    }
