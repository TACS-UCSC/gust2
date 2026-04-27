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
    rope_theta: float | None = None          # auto: max grid side across scales
    n_refine_layers: int = 2                 # within-scale refinement blocks

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
                 token_vectors: jax.Array | None = None,
                 drop_mask: jax.Array | None = None) -> jax.Array:
        """
        Args:
            tokens: (L,) token indices
            scale_ids: (L,) scale index per position
            frame_ids: (L,) frame id per position (0=t0, 1=t1)
            attn_bias: (L, L) additive attention mask
            rope_coords: (L, 2) row/col coordinates
            token_vectors: optional (L, codebook_dim) pre-looked-up vectors
            drop_mask: optional (L,) binary mask — 0 zeros out the embedding

        Returns:
            (L, n_embd) hidden states
        """
        x = self.embedding(tokens, scale_ids, frame_ids,
                           token_vectors=token_vectors)
        if drop_mask is not None:
            x = x * drop_mask[:, None]

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
    """Per-scale prediction heads with shared refinement + position encoding.

    Each head maps n_embd → effective_vocab. Scale masks are applied
    externally in the loss function to zero out invalid token logits.

    pos_proj maps 2D normalized coordinates → n_embd, added to upsampled
    hidden states to distinguish target positions sharing the same source.

    refinement is a shared stack of NSPBlocks run over the K×K target-scale
    hidden states with local (cell-center) RoPE coords. It lets fine-scale
    token choices coordinate across neighbors before the per-scale logit head.
    """
    heads: list
    pos_proj: eqx.nn.Linear
    refinement: list

    def __init__(self, config: NSPConfig, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        head_keys = jax.random.split(k1, len(config.trainable_scale_indices))

        self.heads = [
            eqx.nn.Linear(config.n_embd, config.effective_vocab_size,
                          use_bias=False, key=head_keys[i])
            for i in range(len(config.trainable_scale_indices))
        ]
        self.pos_proj = eqx.nn.Linear(2, config.n_embd, use_bias=True, key=k2)

        n_refine = max(int(config.n_refine_layers), 0)
        if n_refine > 0:
            ref_keys = jax.random.split(k3, n_refine)
            self.refinement = [NSPBlock(config, key=k) for k in ref_keys]
        else:
            self.refinement = []

    def expand(self, h: jax.Array, head_idx: int,
               local_coords: jax.Array) -> jax.Array:
        """Refinement → per-scale logit head.

        Args:
            h: (K*K, n_embd) positioned hidden states for one sample
            head_idx: index into self.heads (position in trainable_indices)
            local_coords: (K*K, 2) cell-center coords on the K×K target grid

        Returns:
            (K*K, effective_vocab) logits
        """
        if len(self.refinement) > 0:
            n = h.shape[0]
            attn_bias = jnp.zeros((n, n), dtype=jnp.float32)
            for blk in self.refinement:
                h = blk(h, attn_bias, local_coords)
        return jax.vmap(self.heads[head_idx])(h)

    def get_num_params(self) -> int:
        return sum(
            x.size for x in jax.tree_util.tree_leaves(eqx.filter(self, eqx.is_array))
        )


def _local_cell_coords(h_k: int, w_k: int) -> jnp.ndarray:
    """Cell-center coords for a K×K grid: 0.5, 1.5, ..., K-0.5.

    Returns: (h_k * w_k, 2) float32, row-major.
    """
    rows = jnp.arange(h_k, dtype=jnp.float32) + 0.5
    cols = jnp.arange(w_k, dtype=jnp.float32) + 0.5
    grid_r, grid_c = jnp.meshgrid(rows, cols, indexing='ij')
    return jnp.stack([grid_r, grid_c], axis=-1).reshape(h_k * w_k, 2)


# =============================================================================
# Forward pass (teacher-forced)
# =============================================================================


def forward_teacher_forced(model: NSPModel, tokens_full: jax.Array,
                           config: NSPConfig,
                           scales_t0: tuple, padded_len_t0: int,
                           scales_t1: tuple, padded_len_t1: int,
                           attn_bias: jax.Array,
                           token_vectors: jax.Array | None = None,
                           drop_mask: jax.Array | None = None,
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
                 token_vectors=token_vectors, drop_mask=drop_mask)


# =============================================================================
# Generation (greedy)
# =============================================================================


def generate_t1_frame(model: NSPModel, exp_heads: ExpansionHeads,
                      config: NSPConfig, t0_tokens: jax.Array,
                      scales_t0: tuple, padded_len_t0: int,
                      scales_t1: tuple, padded_len_t1: int,
                      attn_bias: jax.Array, scale_masks: jax.Array,
                      trainable_indices: list,
                      key: jax.Array,
                      temperature: float = 0.0,
                      top_k: int = 0,
                      top_p: float = 1.0,
                      log_topk: int = 0):
    """Generate a full t1 frame from t0, scale by scale.

    Runs one forward pass per trainable scale. Each scale k is predicted
    from scale k-1's hidden states, bilinear-upsampled through the
    expansion head, with invalid tokens masked out via scale_masks.

    Decoding:
        temperature == 0.0 -> greedy argmax (deterministic, key unused)
        temperature > 0.0  -> ancestral sampling from softmax(logits / T),
            optionally with top-k and/or top-p (nucleus) truncation applied
            to the per-token logits before the categorical draw.

    top_k, top_p and log_topk are captured as Python values at trace time
    so the truncation/logging branches resolve statically.

    Args:
        model, exp_heads: in inference mode
        config: NSPConfig
        t0_tokens: (tokens_per_frame,) compact indices
        scales_t0: all scale tuples, scales_t1: scales[:-1]
        padded_len_t0, padded_len_t1: padded lengths
        attn_bias: (L0+L1, L0+L1) precomputed mask
        scale_masks: (n_scales, effective_vocab) bool
        trainable_indices: list of trainable scale indices
        key: PRNGKey used for sampling (ignored when temperature == 0)
        temperature: sampling temperature; 0 disables sampling
        top_k: keep only the top-k logits per token (0 disables)
        top_p: nucleus threshold in (0, 1]; 1.0 disables
        log_topk: if > 0, also return the top-K post-mask, pre-truncation
            logits + their token indices for every emitted token. The
            captured logits are the raw model output (after scale_mask
            but before temperature scaling and top_k/top_p truncation),
            so they reflect what the model believed before any sampling
            intervention.

    Returns:
        If log_topk == 0: (tokens_per_frame,) predicted t1 compact indices.
        If log_topk  > 0: (predicted, top_logits, top_indices) where
            top_logits / top_indices are (tokens_per_frame, log_topk)
            float32/int32 arrays; deterministic-scale slots (below
            trainable_indices[0]) are filled with zeros.
    """
    boundaries = config.scale_boundaries
    tokens_per_frame = config.tokens_per_frame
    tokens_t1_trunc = sum(h * w for h, w in scales_t1)

    boundaries_t0 = [0]
    for h, w in scales_t0:
        boundaries_t0.append(boundaries_t0[-1] + h * w)

    boundaries_t1 = [0]
    for h, w in scales_t1:
        boundaries_t1.append(boundaries_t1[-1] + h * w)

    # Initialize t1_tokens. Trainable-scale slots are zero and get
    # overwritten by the loop; deterministic-scale slots (scales below
    # trainable_indices[0]) need the actual codebook value, which is
    # identical to t0's value at the same position (by definition of
    # "deterministic": single valid code across the dataset).
    first_trainable_pos = boundaries[trainable_indices[0]]
    t1_tokens = jnp.zeros(tokens_per_frame, dtype=jnp.int32)
    t1_tokens = t1_tokens.at[:first_trainable_pos].set(
        t0_tokens[:first_trainable_pos])

    if temperature == 0.0:
        scale_keys = [None] * len(trainable_indices)
    else:
        scale_keys = list(jax.random.split(key, len(trainable_indices)))

    if log_topk > 0:
        top_logits_full = jnp.zeros(
            (tokens_per_frame, log_topk), dtype=jnp.float32)
        top_indices_full = jnp.zeros(
            (tokens_per_frame, log_topk), dtype=jnp.int32)

    for i, scale_idx in enumerate(trainable_indices):
        h_k, w_k = config.scales[scale_idx]
        n_tokens_k = h_k * w_k

        # Build input: [t0_padded, t1_truncated_padded]
        t1_trunc = t1_tokens[:tokens_t1_trunc]
        t0_pad = jnp.pad(t0_tokens, (0, padded_len_t0 - tokens_per_frame))
        t1_pad = jnp.pad(t1_trunc, (0, padded_len_t1 - tokens_t1_trunc))
        tokens_in = jnp.concatenate([t0_pad, t1_pad])

        hidden = forward_teacher_forced(
            model, tokens_in, config,
            scales_t0, padded_len_t0,
            scales_t1, padded_len_t1,
            attn_bias,
        )

        h_t0 = hidden[:padded_len_t0, :]
        h_t1 = hidden[padded_len_t0:, :]

        # Source hidden states from scale k-1
        if scale_idx == 0:
            src_start = boundaries_t0[0]
            src_end = boundaries_t0[1]
            h_src, w_src = config.scales[0]
            h_source = h_t0[src_start:src_end, :]
        else:
            src_in_t1 = scale_idx - 1
            src_start = boundaries_t1[src_in_t1]
            src_end = boundaries_t1[src_in_t1 + 1]
            h_src, w_src = scales_t1[src_in_t1]
            h_source = h_t1[src_start:src_end, :]

        # Bilinear upsample + position encoding
        h_source_2d = h_source.reshape(h_src, w_src, config.n_embd)
        h_up = jax.image.resize(
            h_source_2d, (h_k, w_k, config.n_embd), method='bilinear')

        rows = jnp.arange(h_k, dtype=jnp.float32) / max(h_k - 1, 1)
        cols = jnp.arange(w_k, dtype=jnp.float32) / max(w_k - 1, 1)
        grid_r, grid_c = jnp.meshgrid(rows, cols, indexing='ij')
        coords = jnp.stack([grid_r, grid_c], axis=-1)
        pos_emb = jax.vmap(jax.vmap(exp_heads.pos_proj))(coords)

        # Expansion (refinement + head) → masked logits → argmax or sample
        h_flat = (h_up + pos_emb).reshape(n_tokens_k, config.n_embd)
        local_coords = _local_cell_coords(h_k, w_k)
        logits = exp_heads.expand(h_flat, i, local_coords)
        logits = jnp.where(scale_masks[scale_idx][None, :], logits, -1e9)

        # Capture pre-temperature, pre-truncation top-K logits & indices.
        # Done *before* top_k/top_p modify logits so the snapshot reflects
        # the model's raw next-token belief, not the sampler's truncated view.
        if log_topk > 0:
            tgt_start = boundaries[scale_idx]
            top_vals_k, top_idx_k = jax.lax.top_k(logits, log_topk)
            top_logits_full = jax.lax.dynamic_update_slice(
                top_logits_full, top_vals_k.astype(jnp.float32),
                (tgt_start, 0))
            top_indices_full = jax.lax.dynamic_update_slice(
                top_indices_full, top_idx_k.astype(jnp.int32),
                (tgt_start, 0))

        if temperature == 0.0:
            predicted = jnp.argmax(logits, axis=-1)
        else:
            if top_k > 0:
                top_vals, _ = jax.lax.top_k(logits, top_k)
                kth = top_vals[..., -1:]
                logits = jnp.where(logits < kth, -1e9, logits)
            if top_p < 1.0:
                sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
                sorted_probs = jax.nn.softmax(
                    sorted_logits / temperature, axis=-1)
                cumprobs = jnp.cumsum(sorted_probs, axis=-1)
                shifted = jnp.concatenate(
                    [jnp.zeros_like(cumprobs[..., :1]),
                     cumprobs[..., :-1]], axis=-1)
                keep = shifted < top_p
                kept = jnp.where(keep, sorted_logits, jnp.inf)
                threshold = jnp.min(kept, axis=-1, keepdims=True)
                logits = jnp.where(logits < threshold, -1e9, logits)
            predicted = jax.random.categorical(
                scale_keys[i], logits / temperature, axis=-1)

        tgt_start = boundaries[scale_idx]
        tgt_end = boundaries[scale_idx + 1]
        t1_tokens = t1_tokens.at[tgt_start:tgt_end].set(predicted)

    if log_topk > 0:
        return t1_tokens, top_logits_full, top_indices_full
    return t1_tokens


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

    if config.rope_theta is None:
        config.rope_theta = float(max(max(h, w) for h, w in config.scales))

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
    print(f"RoPE theta: {config.rope_theta}, "
          f"refine layers: {config.n_refine_layers}")

    return model, exp_heads
