import jax
import jax.numpy as jnp
import equinox as eqx


def _channel_norm(norm: eqx.nn.RMSNorm, x: jax.Array) -> jax.Array:
    """Apply RMSNorm over channel dim of a (C, H, W) tensor.

    Upcasts to float32 for numerical stability, then casts back.
    """
    C, H, W = x.shape
    dtype = x.dtype
    x = x.reshape(C, H * W).T                          # (H*W, C)
    x = jax.vmap(norm)(x.astype(jnp.float32)).astype(dtype)
    return x.T.reshape(C, H, W)                        # (C, H, W)


class EncoderStem(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    conv3: eqx.nn.Conv2d
    norm1: eqx.nn.RMSNorm
    norm2: eqx.nn.RMSNorm
    norm3: eqx.nn.RMSNorm

    def __init__(self, d: int, in_channels: int = 1, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.Conv2d(in_channels, d // 4, 3, stride=2, padding=1, key=k1)
        self.conv2 = eqx.nn.Conv2d(d // 4, d // 2, 3, stride=2, padding=1, key=k2)
        self.conv3 = eqx.nn.Conv2d(d // 2, d, 3, stride=2, padding=1, key=k3)
        self.norm1 = eqx.nn.RMSNorm(d // 4)
        self.norm2 = eqx.nn.RMSNorm(d // 2)
        self.norm3 = eqx.nn.RMSNorm(d)

    def __call__(self, x: jax.Array) -> jax.Array:
        # (C, 256, 256) -> (d, 32, 32)
        x = _channel_norm(self.norm1, jax.nn.silu(self.conv1(x)))
        x = _channel_norm(self.norm2, jax.nn.silu(self.conv2(x)))
        x = _channel_norm(self.norm3, jax.nn.silu(self.conv3(x)))
        return x


class DecoderStem(eqx.Module):
    conv1: eqx.nn.ConvTranspose2d
    conv2: eqx.nn.ConvTranspose2d
    conv3: eqx.nn.ConvTranspose2d
    norm1: eqx.nn.RMSNorm
    norm2: eqx.nn.RMSNorm

    def __init__(self, d: int, out_channels: int = 1, *, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.conv1 = eqx.nn.ConvTranspose2d(d, d // 2, 4, stride=2, padding=1, key=k1)
        self.conv2 = eqx.nn.ConvTranspose2d(d // 2, d // 4, 4, stride=2, padding=1, key=k2)
        self.conv3 = eqx.nn.ConvTranspose2d(d // 4, out_channels, 4, stride=2, padding=1, key=k3)
        self.norm1 = eqx.nn.RMSNorm(d // 2)
        self.norm2 = eqx.nn.RMSNorm(d // 4)

    def __call__(self, x: jax.Array) -> jax.Array:
        # (d, 32, 32) -> (C, 256, 256)
        x = _channel_norm(self.norm1, jax.nn.silu(self.conv1(x)))
        x = _channel_norm(self.norm2, jax.nn.silu(self.conv2(x)))
        x = self.conv3(x)
        return x


# ---------- 2D Rotary Position Embedding ----------


def _rope_freqs(dim: int, theta: float = 10000.0) -> jax.Array:
    return 1.0 / (theta ** (jnp.arange(0, dim, 2) / dim))


def _apply_rope(x: jax.Array, freqs: jax.Array, pos: jax.Array) -> jax.Array:
    angles = pos[:, None] * freqs[None, :]          # (N, dim//2)
    cos, sin = jnp.cos(angles), jnp.sin(angles)
    x1, x2 = x[..., ::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], axis=-1).reshape(x.shape)


def _apply_2d_rope(q: jax.Array, k: jax.Array, H: int, W: int,
                   theta: float = 10000.0) -> tuple[jax.Array, jax.Array]:
    """Apply 2D RoPE: first half of head dim ← row freqs, second half ← col freqs."""
    d_half = q.shape[-1] // 2
    freqs = _rope_freqs(d_half, theta)
    rows = jnp.arange(H * W) // W
    cols = jnp.arange(H * W) % W

    def rope_head(qh, kh):                              # (N, d_head) each
        qh = jnp.concatenate([_apply_rope(qh[:, :d_half], freqs, rows),
                              _apply_rope(qh[:, d_half:], freqs, cols)], axis=-1)
        kh = jnp.concatenate([_apply_rope(kh[:, :d_half], freqs, rows),
                              _apply_rope(kh[:, d_half:], freqs, cols)], axis=-1)
        return qh, kh

    return jax.vmap(rope_head, in_axes=(1, 1), out_axes=(1, 1))(q, k)


# ---------- ViT Block ----------


class ViTBlock(eqx.Module):
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

    def __init__(self, d: int, n_heads: int, mlp_dim: int,
                 rope_theta: float = 32.0, *, key: jax.Array):
        k_qkv, k_out, k_gate, k_up, k_down = jax.random.split(key, 5)
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.rope_theta = rope_theta

        self.attn_norm = eqx.nn.RMSNorm(d)
        self.qkv_proj = eqx.nn.Linear(d, 3 * d, use_bias=False, key=k_qkv)
        self.qk_norm = eqx.nn.RMSNorm(self.d_head)
        self.out_proj = eqx.nn.Linear(d, d, use_bias=False, key=k_out)

        self.ffn_norm = eqx.nn.RMSNorm(d)
        self.gate_proj = eqx.nn.Linear(d, mlp_dim, use_bias=False, key=k_gate)
        self.up_proj = eqx.nn.Linear(d, mlp_dim, use_bias=False, key=k_up)
        self.down_proj = eqx.nn.Linear(mlp_dim, d, use_bias=False, key=k_down)

    def __call__(self, x: jax.Array, H: int, W: int) -> jax.Array:
        # x: (N, d) where N = H*W
        dtype = x.dtype
        r = x
        x = jax.vmap(self.attn_norm)(x.astype(jnp.float32)).astype(dtype)
        qkv = jax.vmap(self.qkv_proj)(x)                          # (N, 3d)
        qkv = qkv.reshape(-1, 3, self.n_heads, self.d_head)       # (N, 3, nh, dh)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = jax.vmap(jax.vmap(self.qk_norm))(q.astype(jnp.float32)).astype(dtype)
        k = jax.vmap(jax.vmap(self.qk_norm))(k.astype(jnp.float32)).astype(dtype)
        q, k = _apply_2d_rope(q, k, H, W, self.rope_theta)
        q, k, v = q.astype(dtype), k.astype(dtype), v.astype(dtype)
        x = jax.nn.dot_product_attention(q, k, v)                 # (N, nh, dh)
        x = x.reshape(-1, self.n_heads * self.d_head)             # (N, d)
        x = jax.vmap(self.out_proj)(x)
        x = r + x

        r = x
        x = jax.vmap(self.ffn_norm)(x.astype(jnp.float32)).astype(dtype)
        x = jax.vmap(self.down_proj)(
            jax.nn.silu(jax.vmap(self.gate_proj)(x))
            * jax.vmap(self.up_proj)(x)
        )                                                          # (N, d)
        x = r + x
        return x


# ---------- Transformer ----------


class Transformer(eqx.Module):
    blocks: list

    def __init__(self, d: int, n_heads: int, mlp_dim: int, depth: int,
                 rope_theta: float = 32.0, *, key: jax.Array):
        keys = jax.random.split(key, depth)
        self.blocks = [ViTBlock(d, n_heads, mlp_dim, rope_theta, key=k) for k in keys]

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (d, H, W) -> (d, H, W)
        d, H, W = x.shape
        x = x.reshape(d, H * W).T                                 # (N, d)
        for block in self.blocks:
            x = block(x, H, W)
        return x.T.reshape(d, H, W)


# ---------- Encoder ----------


class Encoder(eqx.Module):
    stem: EncoderStem
    transformer: Transformer
    norm: eqx.nn.RMSNorm
    proj: eqx.nn.Linear

    def __init__(self, d: int, n_heads: int, mlp_dim: int, depth: int,
                 codebook_dim: int, in_channels: int = 1, rope_theta: float = 32.0,
                 *, key: jax.Array):
        k_stem, k_tf, k_proj = jax.random.split(key, 3)
        self.stem = EncoderStem(d, in_channels, key=k_stem)
        self.transformer = Transformer(d, n_heads, mlp_dim, depth, rope_theta, key=k_tf)
        self.norm = eqx.nn.RMSNorm(d)
        self.proj = eqx.nn.Linear(d, codebook_dim, use_bias=False, key=k_proj)

    def __call__(self, x: jax.Array) -> jax.Array:
        # (C, 256, 256) -> (codebook_dim, 32, 32)
        x = self.stem(x)                               # (d, 32, 32)
        x = self.transformer(x)                         # (d, 32, 32)
        d, H, W = x.shape
        x = x.reshape(d, H * W).T                      # (N, d)
        x = jax.vmap(self.norm)(x.astype(jnp.float32)).astype(x.dtype)
        x = jax.vmap(self.proj)(x)                      # (N, codebook_dim)
        return x.T.reshape(-1, H, W)                       # (codebook_dim, 32, 32)


# ---------- Decoder ----------


class Decoder(eqx.Module):
    proj: eqx.nn.Linear
    transformer: Transformer
    stem: DecoderStem

    def __init__(self, d: int, n_heads: int, mlp_dim: int, depth: int,
                 codebook_dim: int, out_channels: int = 1, rope_theta: float = 32.0,
                 *, key: jax.Array):
        k_proj, k_tf, k_stem = jax.random.split(key, 3)
        self.proj = eqx.nn.Linear(codebook_dim, d, use_bias=False, key=k_proj)
        self.transformer = Transformer(d, n_heads, mlp_dim, depth, rope_theta, key=k_tf)
        self.stem = DecoderStem(d, out_channels, key=k_stem)

    def __call__(self, x: jax.Array) -> jax.Array:
        # (codebook_dim, 32, 32) -> (C, 256, 256)
        V, H, W = x.shape
        x = x.reshape(V, H * W).T                      # (N, codebook_dim)
        x = jax.vmap(self.proj)(x)                      # (N, d)
        x = x.T.reshape(-1, H, W)                       # (d, 32, 32)
        x = self.transformer(x)                          # (d, 32, 32)
        return self.stem(x)                              # (C, 256, 256)
