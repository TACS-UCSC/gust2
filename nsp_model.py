"""
Next-Scale Prediction (NSP) model for turbulence generation.

Teacher-forced VAR-style approach: predicts tokens at t1 conditioned on
all scales at t0 (full context) + coarser scales at t1 (block-causal).
Expansion heads upsample hidden states from scale k-1 to predict scale k.

All prediction heads output logits over the full effective vocabulary,
with per-scale masks zeroing out invalid tokens. This is simpler than
the gust approach of per-scale vocabulary ranges with offsets.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import equinox as eqx

from vit_ae import _rope_freqs, _apply_rope


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class NSPConfig:
    """NSP model configuration.

    Data-derived fields (scales, effective_vocab_size, codebook_dim,
    first_trainable_scale, tokens_per_frame) are populated from the
    tokenized .npz via create_nsp_model().
    """
    # Data-derived (set from tokenized data)
    scales: tuple = ()                 # ((1,1), (2,2), (4,4), ...)
    effective_vocab_size: int = 0
    codebook_dim: int = 0
    first_trainable_scale: int = 0
    tokens_per_frame: int = 0

    # Model hyperparams
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    rope_theta: float = 16.0

    @property
    def n_scales(self) -> int:
        return len(self.scales)

    @property
    def scale_boundaries(self) -> list:
        """Cumulative token positions per scale."""
        boundaries = [0]
        for h, w in self.scales:
            boundaries.append(boundaries[-1] + h * w)
        return boundaries

    @property
    def padded_seq_len(self) -> int:
        """Round tokens_per_frame up to next multiple of 128 for flash attention."""
        return ((self.tokens_per_frame + 127) // 128) * 128

    @property
    def trainable_scale_indices(self) -> list:
        return list(range(self.first_trainable_scale, self.n_scales))


# =============================================================================
# Positional helpers
# =============================================================================


def build_rope_coords(scales: tuple, padded_len: int) -> jnp.ndarray:
    """Map each token to cell-center coordinates on the finest grid.

    For scale (h, w), token at raster position (i, j) maps to:
        row = i * (max_h / h) + (max_h / h / 2)
        col = j * (max_w / w) + (max_w / w / 2)

    Returns: (padded_len, 2) float32. Padding positions get (0, 0).
    """
    max_h = max(h for h, w in scales)
    max_w = max(w for h, w in scales)

    coords = []
    for h, w in scales:
        step_h = max_h / h
        step_w = max_w / w
        for i in range(h):
            for j in range(w):
                coords.append((i * step_h + step_h / 2,
                                j * step_w + step_w / 2))

    while len(coords) < padded_len:
        coords.append((0.0, 0.0))

    return jnp.array(coords, dtype=jnp.float32)


def get_scale_ids(scales: tuple, padded_len: int) -> jnp.ndarray:
    """Assign a scale index to each token position. Padding gets n_scales."""
    n_scales = len(scales)
    scale_ids = []
    for k, (h, w) in enumerate(scales):
        scale_ids.extend([k] * (h * w))
    scale_ids = jnp.array(scale_ids, dtype=jnp.int32)
    return jnp.pad(scale_ids, (0, padded_len - len(scale_ids)),
                   constant_values=n_scales)


# =============================================================================
# Attention mask
# =============================================================================


def build_teacher_forced_mask(scales_t0: tuple, padded_len_t0: int,
                              scales_t1: tuple, padded_len_t1: int
                              ) -> jnp.ndarray:
    """Build asymmetric attention mask for [full t0, truncated t1].

    Shape: (padded_len_t0 + padded_len_t1, padded_len_t0 + padded_len_t1)

    Quadrants:
    - t0→t0: full attention (0.0)
    - t0→t1: blocked (-1e9)
    - t1→t0: full attention (0.0)
    - t1→t1: source_scale <= target_scale (within-scale self-attn allowed)
    """
    L0 = padded_len_t0
    L1 = padded_len_t1
    total_len = L0 + L1

    full_mask = jnp.full((total_len, total_len), -1e9, dtype=jnp.float32)

    # t0→t0: full attention
    full_mask = full_mask.at[:L0, :L0].set(0.0)

    # t1→t0: full attention
    full_mask = full_mask.at[L0:, :L0].set(0.0)

    # t1→t1: source_scale <= target_scale
    t1_mask = _build_tf_t1_mask(scales_t1, padded_len_t1)
    full_mask = full_mask.at[L0:, L0:].set(t1_mask)

    # Fix padding
    total_tokens_t0 = sum(h * w for h, w in scales_t0)
    total_tokens_t1 = sum(h * w for h, w in scales_t1)

    is_padding = jnp.concatenate([
        jnp.arange(L0) >= total_tokens_t0,
        jnp.arange(L1) >= total_tokens_t1,
    ])

    diag_mask = jnp.eye(total_len, dtype=bool)

    # Padding rows: identity only
    full_mask = jnp.where(is_padding[:, None] & diag_mask, 0.0, full_mask)
    # Block attention to padding columns
    full_mask = jnp.where(is_padding[None, :], -1e9, full_mask)
    # Re-open diagonal for padding
    full_mask = jnp.where(is_padding[:, None] & diag_mask, 0.0, full_mask)

    return full_mask


def _build_tf_t1_mask(scales: tuple, padded_len: int) -> jnp.ndarray:
    """t1→t1 mask: source_scale <= target_scale."""
    total = sum(h * w for h, w in scales)
    n_scales = len(scales)

    scale_ids = []
    for k, (h, w) in enumerate(scales):
        scale_ids.extend([k] * (h * w))
    scale_ids = jnp.array(scale_ids, dtype=jnp.int32)
    scale_ids = jnp.pad(scale_ids, (0, padded_len - total),
                        constant_values=n_scales)

    target_scale = scale_ids[:, None]
    source_scale = scale_ids[None, :]

    return jnp.where(source_scale <= target_scale, 0.0, -1e9)


# =============================================================================
# 2D Axial RoPE (coord-based, for multi-scale sequences)
# =============================================================================


def _apply_2d_rope_coords(q: jax.Array, k: jax.Array,
                          coords: jax.Array, theta: float
                          ) -> tuple[jax.Array, jax.Array]:
    """Apply 2D axial RoPE using (row, col) coordinates.

    First half of head dim uses row coords, second half uses col coords.
    Reuses _rope_freqs and _apply_rope from vit_ae.py.

    Args:
        q, k: (N, n_heads, d_head)
        coords: (N, 2) — row, col positions
        theta: base frequency
    """
    d_half = q.shape[-1] // 2
    freqs = _rope_freqs(d_half, theta)
    rows = coords[:, 0]
    cols = coords[:, 1]

    def rope_head(qh, kh):
        qh = jnp.concatenate([_apply_rope(qh[:, :d_half], freqs, rows),
                              _apply_rope(qh[:, d_half:], freqs, cols)], axis=-1)
        kh = jnp.concatenate([_apply_rope(kh[:, :d_half], freqs, rows),
                              _apply_rope(kh[:, d_half:], freqs, cols)], axis=-1)
        return qh, kh

    return jax.vmap(rope_head, in_axes=(1, 1), out_axes=(1, 1))(q, k)


# =============================================================================
# Embedding
# =============================================================================


class NSPEmbedding(eqx.Module):
    """Embedding for NSP model.

    Combines codebook projection, scale embedding, and frame embedding.
    Position information is provided by 2D axial RoPE in each block.
    """
    codebook: jax.Array                # (effective_vocab, codebook_dim) — frozen
    codebook_proj: eqx.nn.Linear       # codebook_dim → n_embd
    scale_embed: eqx.nn.Embedding      # (n_scales, n_embd)
    frame_embed: eqx.nn.Embedding      # (2, n_embd)

    def __init__(self, config: NSPConfig, codebook: jax.Array, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.codebook = codebook
        self.codebook_proj = eqx.nn.Linear(
            config.codebook_dim, config.n_embd, use_bias=False, key=k1)
        self.scale_embed = eqx.nn.Embedding(config.n_scales, config.n_embd, key=k2)
        self.frame_embed = eqx.nn.Embedding(2, config.n_embd, key=k3)

    def __call__(self, tokens: jax.Array, scale_ids: jax.Array,
                 frame_ids: jax.Array,
                 token_vectors: jax.Array | None = None) -> jax.Array:
        """
        Args:
            tokens: (L,) token indices
            scale_ids: (L,) scale index per position
            frame_ids: (L,) frame index per position (0=t0, 1=t1)
            token_vectors: optional (L, codebook_dim) pre-looked-up vectors
        """
        # Token embedding (one-hot matmul for sharding safety)
        if token_vectors is not None:
            vectors = jax.lax.stop_gradient(token_vectors)
        else:
            K = self.codebook.shape[0]
            vectors = jax.lax.stop_gradient(
                jax.nn.one_hot(tokens, K) @ self.codebook)
        tok_emb = jax.vmap(self.codebook_proj)(vectors)

        # Scale + frame embeddings
        scale_ids_clamped = jnp.minimum(scale_ids, self.scale_embed.num_embeddings - 1)
        scale_emb = jax.vmap(self.scale_embed)(scale_ids_clamped)
        frame_emb = jax.vmap(self.frame_embed)(frame_ids)

        return tok_emb + scale_emb + frame_emb


# =============================================================================
# Transformer block
# =============================================================================


class NSPBlock(eqx.Module):
    """Pre-norm transformer block with block-causal attention bias.

    Uses RMSNorm, QK-norm, bias-free projections, SwiGLU FFN.
    """
    attn_norm: eqx.nn.RMSNorm
    qkv_proj: eqx.nn.Linear
    qk_norm: eqx.nn.RMSNorm
    out_proj: eqx.nn.Linear
    ffn_norm: eqx.nn.RMSNorm
    gate_proj: eqx.nn.Linear
    up_proj: eqx.nn.Linear
    down_proj: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    d_head: int = eqx.field(static=True)
    rope_theta: float = eqx.field(static=True)

    def __init__(self, config: NSPConfig, *, key: jax.Array):
        k_qkv, k_out, k_gate, k_up, k_down = jax.random.split(key, 5)
        d = config.n_embd
        self.n_heads = config.n_head
        self.d_head = d // config.n_head
        self.rope_theta = config.rope_theta

        self.attn_norm = eqx.nn.RMSNorm(d)
        self.qkv_proj = eqx.nn.Linear(d, 3 * d, use_bias=False, key=k_qkv)
        self.qk_norm = eqx.nn.RMSNorm(self.d_head)
        self.out_proj = eqx.nn.Linear(d, d, use_bias=False, key=k_out)

        # SwiGLU FFN: hidden_dim = 4*d*2/3, rounded to multiple of 64
        hidden_dim = (4 * d * 2) // 3
        hidden_dim = ((hidden_dim + 63) // 64) * 64
        self.ffn_norm = eqx.nn.RMSNorm(d)
        self.gate_proj = eqx.nn.Linear(d, hidden_dim, use_bias=False, key=k_gate)
        self.up_proj = eqx.nn.Linear(d, hidden_dim, use_bias=False, key=k_up)
        self.down_proj = eqx.nn.Linear(hidden_dim, d, use_bias=False, key=k_down)

    def __call__(self, x: jax.Array, attn_bias: jax.Array,
                 rope_coords: jax.Array) -> jax.Array:
        """
        Args:
            x: (L, d)
            attn_bias: (L, L) additive mask
            rope_coords: (L, 2) row/col coordinates for RoPE
        """
        dtype = x.dtype
        d = self.n_heads * self.d_head

        # Attention
        r = x
        x = jax.vmap(self.attn_norm)(x.astype(jnp.float32)).astype(dtype)
        qkv = jax.vmap(self.qkv_proj)(x)
        qkv = qkv.reshape(-1, 3, self.n_heads, self.d_head)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # QK-norm (upcast to f32 for stability)
        q = jax.vmap(jax.vmap(self.qk_norm))(q.astype(jnp.float32)).astype(dtype)
        k = jax.vmap(jax.vmap(self.qk_norm))(k.astype(jnp.float32)).astype(dtype)

        # 2D axial RoPE (promotes to f32 internally)
        q, k = _apply_2d_rope_coords(q, k, rope_coords, self.rope_theta)
        q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)

        # Attention with bias
        bias = attn_bias[None, None, :, :].astype(dtype)
        x = jax.nn.dot_product_attention(q, k, v, bias=bias)
        x = x.reshape(-1, d)
        x = jax.vmap(self.out_proj)(x)
        x = r + x

        # FFN (SwiGLU)
        r = x
        x = jax.vmap(self.ffn_norm)(x.astype(jnp.float32)).astype(dtype)
        x = jax.vmap(self.down_proj)(
            jax.nn.silu(jax.vmap(self.gate_proj)(x))
            * jax.vmap(self.up_proj)(x)
        )
        x = r + x

        return x


# =============================================================================
# Full model
# =============================================================================


class NSPModel(eqx.Module):
    """NSP transformer backbone.

    Returns hidden states; prediction heads are in ExpansionHeads.
    """
    _config: NSPConfig = eqx.field(static=True)
    embedding: NSPEmbedding
    blocks: list
    ln_f: eqx.nn.RMSNorm

    def __init__(self, config: NSPConfig, codebook: jax.Array, *, key: jax.Array):
        self._config = config
        k1, k2 = jax.random.split(key)

        self.embedding = NSPEmbedding(config, codebook, key=k1)
        block_keys = jax.random.split(k2, config.n_layer)
        self.blocks = [NSPBlock(config, key=bk) for bk in block_keys]
        self.ln_f = eqx.nn.RMSNorm(config.n_embd)

    def __call__(self, tokens: jax.Array, scale_ids: jax.Array,
                 frame_ids: jax.Array, attn_bias: jax.Array,
                 rope_coords: jax.Array,
                 token_vectors: jax.Array | None = None) -> jax.Array:
        """
        Args:
            tokens: (L,) token indices
            scale_ids: (L,) scale index per position
            frame_ids: (L,) frame id per position (0=t0, 1=t1)
            attn_bias: (L, L) additive attention mask
            rope_coords: (L, 2) row/col coordinates
            token_vectors: optional (L, codebook_dim) pre-looked-up vectors

        Returns:
            (L, n_embd) hidden states
        """
        x = self.embedding(tokens, scale_ids, frame_ids,
                           token_vectors=token_vectors)

        for block in self.blocks:
            x = eqx.filter_checkpoint(block)(x, attn_bias, rope_coords)

        x = jax.vmap(self.ln_f)(x.astype(jnp.float32)).astype(x.dtype)
        return x

    def get_num_params(self) -> int:
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )


# =============================================================================
# Expansion heads
# =============================================================================


class ExpansionHeads(eqx.Module):
    """Per-scale prediction heads with shared position encoding.

    Each head maps n_embd → effective_vocab. Scale masks are applied
    externally in the loss function to zero out invalid token logits.

    pos_proj maps 2D normalized coordinates → n_embd, added to upsampled
    hidden states to distinguish target positions sharing the same source.
    """
    heads: list
    pos_proj: eqx.nn.Linear

    def __init__(self, config: NSPConfig, *, key: jax.Array):
        k1, k2 = jax.random.split(key)
        head_keys = jax.random.split(k1, len(config.trainable_scale_indices))

        self.heads = [
            eqx.nn.Linear(config.n_embd, config.effective_vocab_size,
                          use_bias=False, key=head_keys[i])
            for i in range(len(config.trainable_scale_indices))
        ]
        self.pos_proj = eqx.nn.Linear(2, config.n_embd, use_bias=True, key=k2)

    def get_num_params(self) -> int:
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )


# =============================================================================
# Forward pass (teacher-forced)
# =============================================================================


def forward_teacher_forced(model: NSPModel, tokens_full: jax.Array,
                           config: NSPConfig,
                           scales_t0: tuple, padded_len_t0: int,
                           scales_t1: tuple, padded_len_t1: int,
                           attn_bias: jax.Array,
                           token_vectors: jax.Array | None = None
                           ) -> jax.Array:
    """Forward pass with asymmetric t0 (full) / t1 (truncated) lengths.

    Args:
        model: NSPModel instance
        tokens_full: (L0 + L1,) padded token indices
        config: NSPConfig
        scales_t0: full-frame scale tuples
        scales_t1: truncated scale tuples (last scale excluded)
        padded_len_t0, padded_len_t1: padded lengths
        attn_bias: (L0+L1, L0+L1) attention mask
        token_vectors: optional (L0+L1, codebook_dim) pre-looked-up vectors

    Returns:
        (L0+L1, n_embd) hidden states
    """
    # Scale IDs — asymmetric
    scale_ids_t0 = get_scale_ids(scales_t0, padded_len_t0)
    scale_ids_t1 = get_scale_ids(scales_t1, padded_len_t1)
    scale_ids = jnp.concatenate([scale_ids_t0, scale_ids_t1])

    # Frame IDs
    frame_ids = jnp.concatenate([
        jnp.zeros(padded_len_t0, dtype=jnp.int32),
        jnp.ones(padded_len_t1, dtype=jnp.int32),
    ])

    # RoPE coords — asymmetric
    coords_t0 = build_rope_coords(scales_t0, padded_len_t0)
    coords_t1 = build_rope_coords(scales_t1, padded_len_t1)
    rope_coords = jnp.concatenate([coords_t0, coords_t1], axis=0)

    return model(tokens_full, scale_ids, frame_ids, attn_bias, rope_coords,
                 token_vectors=token_vectors)


# =============================================================================
# Factory
# =============================================================================


def create_nsp_model(token_data: dict, config: NSPConfig, key: jax.Array
                     ) -> tuple[NSPModel, ExpansionHeads]:
    """Create NSP model + expansion heads from tokenized data.

    Populates data-derived config fields from token_data, then initializes.

    Args:
        token_data: dict from tokenizer.load_tokenized_data()
        config: NSPConfig with model hyperparams set
        key: JAX random key

    Returns:
        (model, exp_heads) tuple
    """
    codebook = jnp.array(token_data["codebook"])

    # Populate data-derived config fields
    scales_int = token_data["scales"]
    config.scales = tuple((s, s) for s in scales_int)
    config.effective_vocab_size = int(token_data["effective_vocab_size"])
    config.codebook_dim = int(token_data["codebook_dim"])
    config.tokens_per_frame = sum(s * s for s in scales_int)
    config.first_trainable_scale = int(token_data.get("first_trainable_scale", 0))

    k1, k2 = jax.random.split(key)
    model = NSPModel(config, codebook, key=k1)
    exp_heads = ExpansionHeads(config, key=k2)

    print(f"NSPModel: {model.get_num_params()/1e6:.2f}M parameters")
    print(f"ExpansionHeads: {exp_heads.get_num_params()/1e6:.2f}M parameters")
    print(f"Scales: {config.scales}")
    print(f"Tokens/frame: {config.tokens_per_frame}, "
          f"padded: {config.padded_seq_len}")
    print(f"Effective vocab: {config.effective_vocab_size}")
    print(f"First trainable scale: {config.first_trainable_scale}")

    return model, exp_heads
