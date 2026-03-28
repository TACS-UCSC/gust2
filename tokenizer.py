"""Tokenization pipeline for AR model training.

Wraps a trained gust2 multi-scale VQ-VAE to prepare discrete token
sequences for autoregressive model training.  Supports both streaming
(live) and file-saving modes.

Key features:
- Load trained checkpoint (encoder, decoder, vq, codebook)
- Encode data to compact discrete indices
- Single global remapping (sparse original indices → consecutive)
- Per-scale binary masks indicating valid tokens at each scale
- Save/load tokenized data as NPZ
"""

import argparse
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from vit_ae import Encoder, Decoder
from vq import MultiScaleVQ, EMAState, bicubic_resize


# ---------- JIT helpers ----------


@eqx.filter_jit
def _vmap_encode(encoder, vq, codebook, batch):
    """JIT-compiled batched encode."""
    encoder = eqx.nn.inference_mode(encoder)
    vq = eqx.nn.inference_mode(vq)

    def encode_single(x):
        z_e = encoder(x)
        z_q, all_indices, _partials, _commit, _all_z = vq(z_e, codebook)
        return z_q, all_indices

    return jax.vmap(encode_single)(batch)


# ---------- Checkpoint loading ----------


def load_checkpoint(checkpoint_dir, key):
    """Load gust2 model from checkpoint directory.

    Args:
        checkpoint_dir: directory with encoder.eqx, decoder.eqx, vq.eqx,
                        ema_state.npz, training_state.json
        key: JAX PRNGKey for model skeleton init

    Returns:
        (encoder, decoder, vq, ema_state, arch_config)
    """
    with open(os.path.join(checkpoint_dir, "training_state.json")) as f:
        training_state = json.load(f)
    arch_config = training_state["arch_config"]
    scales = tuple(arch_config["scales"])

    k_enc, k_dec, k_vq = jax.random.split(key, 3)
    encoder = Encoder(
        arch_config["d_model"], arch_config["n_heads"],
        arch_config["mlp_dim"], arch_config["encoder_depth"],
        arch_config["codebook_dim"],
        rope_theta=arch_config.get("rope_theta", 32.0), key=k_enc,
    )
    decoder = Decoder(
        arch_config["d_model"], arch_config["n_heads"],
        arch_config["mlp_dim"], arch_config["decoder_depth"],
        arch_config["codebook_dim"],
        rope_theta=arch_config.get("rope_theta", 32.0), key=k_dec,
    )
    vq_mod = MultiScaleVQ(arch_config["codebook_dim"], scales=scales, key=k_vq)

    encoder = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "encoder.eqx"), encoder)
    decoder = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "decoder.eqx"), decoder)
    vq_mod = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "vq.eqx"), vq_mod)

    ema_data = np.load(os.path.join(checkpoint_dir, "ema_state.npz"))
    ema_state = EMAState(
        cluster_sizes=jnp.array(ema_data["cluster_sizes"]),
        codebook_sums=jnp.array(ema_data["codebook_sums"]),
        codebook=jnp.array(ema_data["codebook"]),
    )

    return encoder, decoder, vq_mod, ema_state, arch_config


# ---------- Index reconstruction ----------


def reconstruct_from_indices(indices_list, codebook, vq):
    """Reconstruct z_q from multi-scale codebook indices.

    Replays the MultiScaleVQ forward loop (phi_convs + post_quant_conv)
    without the encoder.

    Args:
        indices_list: list of (s²,) int arrays, one per scale
        codebook: (K, D) shared codebook
        vq: MultiScaleVQ module (for phi_convs and post_quant_conv)

    Returns:
        z_q: (D, 32, 32) reconstructed quantized latent
    """
    D = codebook.shape[1]
    K = codebook.shape[0]
    f_hat = jnp.zeros((D, 1, 1))

    for k, s in enumerate(vq.scales):
        f_hat = bicubic_resize(f_hat, s, s)
        z_q_flat = jax.nn.one_hot(indices_list[k], K) @ codebook  # (s², D)
        z_q_k = z_q_flat.T.reshape(D, s, s)
        z_q_k = vq.phi_convs[k](z_q_k)
        f_hat = f_hat + z_q_k

    z_q = bicubic_resize(f_hat, vq.full_size, vq.full_size)
    z_q = vq.post_quant_conv(z_q)
    return z_q


# ---------- Index flattening ----------


def flatten_multiscale_indices(indices_list):
    """Flatten multi-scale indices to a single 1D array."""
    return jnp.concatenate([idx.flatten() for idx in indices_list])


def unflatten_to_scales(flat_indices, scales):
    """Unflatten 1D indices back to per-scale list of (s²,) arrays."""
    result = []
    offset = 0
    for s in scales:
        size = s * s
        result.append(flat_indices[offset:offset + size])
        offset += size
    return result


# ---------- Tokenizer ----------


class VQVAETokenizer:
    """Tokenizer wrapping a trained gust2 multi-scale VQ-VAE.

    After fit(), provides:
    - old_to_new: (K,) mapping from original to compact index (-1 if unused)
    - new_to_old: (effective_vocab,) reverse mapping
    - effective_codebook: (effective_vocab, D)
    - scale_masks: (n_scales, effective_vocab) bool — valid tokens per scale
    """

    def __init__(self, encoder, decoder, vq, codebook, arch_config,
                 first_trainable_scale=None):
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self._codebook = codebook          # (K, D) original shared
        self.arch_config = arch_config
        self.scales = tuple(arch_config["scales"])

        self.first_trainable_scale = first_trainable_scale
        self.deterministic_scales = None

        # Populated by fit()
        self.old_to_new = None
        self.new_to_old = None
        self.effective_codebook = None
        self.scale_masks = None

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, key, first_trainable_scale=None):
        """Load tokenizer from a gust2 checkpoint directory."""
        encoder, decoder, vq, ema_state, arch_config = load_checkpoint(
            checkpoint_dir, key)
        return cls(encoder, decoder, vq, ema_state.codebook, arch_config,
                   first_trainable_scale)

    # -- Properties --

    @property
    def is_fitted(self):
        return self.old_to_new is not None

    @property
    def codebook(self):
        """Original shared codebook (K, D)."""
        return self._codebook

    @property
    def codebook_dim(self):
        return self._codebook.shape[1]

    @property
    def vocab_size(self):
        """Original codebook size K."""
        return self._codebook.shape[0]

    @property
    def effective_vocab_size(self):
        if self.effective_codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return len(self.effective_codebook)

    @property
    def remapped_codebook(self):
        """Effective codebook (effective_vocab, D)."""
        if self.effective_codebook is None:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        return self.effective_codebook

    @property
    def tokens_per_sample(self):
        return sum(s * s for s in self.scales)

    # -- Fitting --

    def fit(self, batches):
        """Collect unique indices per scale, build compact mapping + masks.

        Args:
            batches: iterable yielding (B, 1, 256, 256) arrays

        Returns:
            self
        """
        print("Fitting tokenizer: collecting unique codebook indices...")
        n_scales = len(self.scales)
        per_scale_unique = [[] for _ in range(n_scales)]
        # Track raw usage counts per original codebook index per scale
        raw_counts = np.zeros((n_scales, self.vocab_size), dtype=np.int64)
        n_processed = 0
        batch_count = 0

        for batch in batches:
            batch = jnp.asarray(batch)
            n_processed += batch.shape[0]
            batch_count += 1

            _, all_indices = _vmap_encode(
                self.encoder, self.vq, self._codebook, batch)
            for k in range(n_scales):
                flat = np.array(all_indices[k].reshape(-1))
                unique_k = np.unique(flat)
                per_scale_unique[k].append(unique_k)
                np.add.at(raw_counts[k], flat, 1)

            if batch_count % 50 == 0:
                print(f"  Processed {n_processed} samples")

        print(f"  Processed {n_processed} samples (done)")

        # Global unique set across all scales
        all_unique = np.unique(np.concatenate(
            [np.concatenate(u) for u in per_scale_unique]))
        effective_vocab = len(all_unique)

        # old_to_new: original index → compact index (-1 if unused)
        old_to_new = np.full(self.vocab_size, -1, dtype=np.int32)
        old_to_new[all_unique] = np.arange(effective_vocab, dtype=np.int32)
        self.old_to_new = jnp.array(old_to_new)

        # new_to_old: compact index → original index
        self.new_to_old = jnp.array(all_unique, dtype=jnp.int32)

        # Effective codebook
        self.effective_codebook = self._codebook[self.new_to_old]

        # Per-scale masks: (n_scales, effective_vocab)
        masks = np.zeros((n_scales, effective_vocab), dtype=bool)
        per_scale_counts = []
        for k in range(n_scales):
            combined = np.unique(np.concatenate(per_scale_unique[k]))
            compact = old_to_new[combined]
            masks[k, compact] = True
            per_scale_counts.append(len(combined))
            s = self.scales[k]
            print(f"  Scale {s}x{s}: {len(combined):4d} unique codes "
                  f"({len(combined)/effective_vocab:.0%} of effective vocab)")
        self.scale_masks = jnp.array(masks)

        # Auto-detect deterministic scales
        if self.first_trainable_scale is None:
            last_det = -1
            for k in range(n_scales):
                if per_scale_counts[k] == 1:
                    last_det = k
                else:
                    break
            self.first_trainable_scale = last_det + 1
        self.deterministic_scales = list(range(self.first_trainable_scale))

        if self.first_trainable_scale > 0:
            det_names = [f"{s}x{s}" for s in self.scales[:self.first_trainable_scale]]
            print(f"Deterministic scales: {', '.join(det_names)}")
        else:
            print("No deterministic scales detected")

        # Remap raw counts to effective vocab space: (n_scales, effective_vocab)
        self.scale_usage_counts = np.zeros(
            (n_scales, effective_vocab), dtype=np.int64)
        for k in range(n_scales):
            for orig_idx in range(self.vocab_size):
                new_idx = old_to_new[orig_idx]
                if new_idx >= 0:
                    self.scale_usage_counts[k, new_idx] = raw_counts[k, orig_idx]

        print(f"Fit complete: {effective_vocab} effective vocab "
              f"(from {self.vocab_size} original, {n_scales} scales)")
        return self

    def set_mapping(self, **kwargs):
        """Set mapping attributes manually (e.g. from loaded data)."""
        for attr in ('old_to_new', 'new_to_old', 'effective_codebook',
                     'scale_masks'):
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])

    # -- Encoding --

    def encode(self, x):
        """Encode a single sample.

        Args:
            x: (1, 256, 256) single input

        Returns:
            remapped_indices: list of (s²,) compact index arrays per scale
            z_q: (D, 32, 32) quantized latent
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        encoder = eqx.nn.inference_mode(self.encoder)
        vq = eqx.nn.inference_mode(self.vq)
        z_e = encoder(x)
        z_q, all_indices, _, _, _ = vq(z_e, self._codebook)
        remapped = [self.old_to_new[idx] for idx in all_indices]
        return remapped, z_q

    def encode_batch(self, batch):
        """Encode a batch.

        Args:
            batch: (B, 1, 256, 256)

        Returns:
            remapped_indices: list of (B, s²) compact index arrays per scale
            z_q: (B, D, 32, 32)
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        z_q, all_indices = _vmap_encode(
            self.encoder, self.vq, self._codebook, batch)
        remapped = [self.old_to_new[idx] for idx in all_indices]
        return remapped, z_q

    def encode_batch_flat(self, batch):
        """Encode a batch to flattened compact indices and vectors.

        Args:
            batch: (B, 1, 256, 256)

        Returns:
            indices_flat: (B, total_tokens) compact indices
            vectors_flat: (B, total_tokens, D) codebook vectors
        """
        remapped, _ = self.encode_batch(batch)
        B = batch.shape[0]
        indices_flat = jnp.concatenate(
            [idx.reshape(B, -1) for idx in remapped], axis=1)
        vectors_flat = self.effective_codebook[indices_flat]
        return indices_flat, vectors_flat

    # -- Decoding --

    def decode_indices(self, remapped_indices):
        """Decode from compact (remapped) indices to image.

        Args:
            remapped_indices: list of (s²,) compact index arrays per scale

        Returns:
            reconstruction: (1, 256, 256)
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        original_indices = [self.new_to_old[idx] for idx in remapped_indices]
        z_q = reconstruct_from_indices(original_indices, self._codebook, self.vq)
        decoder = eqx.nn.inference_mode(self.decoder)
        return decoder(z_q)

    def decode_flat_indices(self, flat_remapped_indices):
        """Decode from flattened compact indices.

        Args:
            flat_remapped_indices: (total_tokens,) flat compact indices

        Returns:
            reconstruction: (1, 256, 256)
        """
        indices_list = unflatten_to_scales(flat_remapped_indices, self.scales)
        return self.decode_indices(indices_list)


# ---------- Save / Load ----------


def save_tokenized_data(path, tokenizer, batches, config):
    """Tokenize and save entire dataset to NPZ.

    Args:
        path: output .npz path
        tokenizer: fitted VQVAETokenizer
        batches: iterable yielding (B, 1, 256, 256)
        config: arch config dict
    """
    if not tokenizer.is_fitted:
        raise ValueError("Tokenizer not fitted. Call fit() first.")

    print(f"Tokenizing and saving to {path}...")
    all_indices = []
    per_scale_indices = {k: [] for k in range(len(tokenizer.scales))}
    n_processed = 0
    batch_count = 0

    for batch in batches:
        batch = jnp.asarray(batch)
        n_processed += batch.shape[0]
        batch_count += 1

        remapped, _ = tokenizer.encode_batch(batch)
        B = batch.shape[0]
        indices_flat = jnp.concatenate(
            [idx.reshape(B, -1) for idx in remapped], axis=1)
        all_indices.append(np.array(indices_flat))

        for k in range(len(tokenizer.scales)):
            per_scale_indices[k].append(np.array(remapped[k]))

        if batch_count % 50 == 0:
            print(f"  Processed {n_processed} samples")

    print(f"  Processed {n_processed} samples (done)")

    all_indices = np.concatenate(all_indices, axis=0)

    save_dict = {
        "codebook": np.array(tokenizer.remapped_codebook),
        "original_codebook": np.array(tokenizer.codebook),
        "effective_vocab_size": np.array(tokenizer.effective_vocab_size),
        "vocab_size": np.array(tokenizer.vocab_size),
        "codebook_dim": np.array(tokenizer.codebook_dim),
        "indices_flat": all_indices,
        "config_json": np.array(json.dumps(config)),
        "scales": np.array(tokenizer.scales),
        "old_to_new": np.array(tokenizer.old_to_new),
        "new_to_old": np.array(tokenizer.new_to_old),
        "scale_masks": np.array(tokenizer.scale_masks),
    }

    if tokenizer.first_trainable_scale is not None:
        save_dict["first_trainable_scale"] = np.array(
            tokenizer.first_trainable_scale)

    for k, s in enumerate(tokenizer.scales):
        save_dict[f"indices_scale_{s}"] = np.concatenate(
            per_scale_indices[k], axis=0)

    np.savez_compressed(path, **save_dict)
    print(f"Saved: {all_indices.shape[0]} samples, "
          f"{tokenizer.effective_vocab_size} effective vocab, "
          f"{all_indices.shape[1]} tokens/sample")


def load_tokenized_data(path):
    """Load tokenized data for AR training.

    Returns:
        dict with indices_flat, vectors_flat, codebook, scale_masks,
        mappings, config, etc.
    """
    data = dict(np.load(path, allow_pickle=True))

    codebook = data["codebook"]
    indices_flat = data["indices_flat"]
    vectors_flat = codebook[indices_flat]

    result = {
        "indices_flat": indices_flat,
        "vectors_flat": vectors_flat,
        "codebook": codebook,
        "effective_vocab_size": int(data["effective_vocab_size"]),
        "vocab_size": int(data["vocab_size"]),
        "codebook_dim": int(data["codebook_dim"]),
        "config": json.loads(str(data["config_json"])),
        "scales": tuple(int(s) for s in data["scales"]),
        "old_to_new": data["old_to_new"],
        "new_to_old": data["new_to_old"],
        "scale_masks": data["scale_masks"],
    }

    if "original_codebook" in data:
        result["original_codebook"] = data["original_codebook"]
    if "first_trainable_scale" in data:
        result["first_trainable_scale"] = int(data["first_trainable_scale"])

    # Per-scale index arrays
    for s in result["scales"]:
        key = f"indices_scale_{s}"
        if key in data:
            result[key] = data[key]

    return result


# ---------- Info ----------


def print_tokenizer_info(tokenizer, n_samples):
    """Print tokenizer statistics."""
    print("\n" + "=" * 60)
    print("TOKENIZER INFO")
    print("=" * 60)

    print(f"\nOriginal vocab size: {tokenizer.vocab_size}")
    print(f"Effective vocab size: {tokenizer.effective_vocab_size}")
    print(f"Codebook dimension: {tokenizer.codebook_dim}")

    scale_strs = [f"{s}x{s}" for s in tokenizer.scales]
    print(f"\nScales: {', '.join(scale_strs)}")
    print(f"Tokens per scale: {[s*s for s in tokenizer.scales]}")
    print(f"Total tokens per sample: {tokenizer.tokens_per_sample}")

    if tokenizer.first_trainable_scale is not None and tokenizer.first_trainable_scale > 0:
        det = [f"{s}x{s}" for s in tokenizer.scales[:tokenizer.first_trainable_scale]]
        print(f"Deterministic scales: {', '.join(det)}")
        s = tokenizer.scales[tokenizer.first_trainable_scale]
        print(f"First trainable scale: {s}x{s} (index {tokenizer.first_trainable_scale})")

    # Per-scale mask density
    print(f"\nPer-scale mask density:")
    for k, s in enumerate(tokenizer.scales):
        n_valid = int(tokenizer.scale_masks[k].sum())
        pct = n_valid / tokenizer.effective_vocab_size
        print(f"  Scale {s:2d}x{s:<2d}: {n_valid:4d} / {tokenizer.effective_vocab_size} "
              f"valid tokens ({pct:.1%})")

    # Per-scale codebook usage
    if tokenizer.scale_usage_counts is not None:
        print(f"\nCodebook usage per scale:")
        for k, s in enumerate(tokenizer.scales):
            counts = tokenizer.scale_usage_counts[k]
            used = counts[counts > 0]
            if len(used) == 0:
                print(f"  Scale {s:2d}x{s:<2d}: no usage")
                continue
            total = counts.sum()
            print(f"  Scale {s:2d}x{s:<2d}: {len(used):4d} codes used, "
                  f"total={total:>10d}, "
                  f"min={used.min()}, max={used.max()}, "
                  f"mean={used.mean():.1f}, std={used.std():.1f}")

    # Scale overlap matrix
    n_scales = len(tokenizer.scales)
    masks = np.array(tokenizer.scale_masks)
    print(f"\nCodebook scale overlap (shared code count):")
    header = "         " + "".join(f"{s:>6d}" for s in tokenizer.scales)
    print(header)
    for i, si in enumerate(tokenizer.scales):
        row = f"  {si:2d}x{si:<2d}  "
        for j, sj in enumerate(tokenizer.scales):
            overlap = int((masks[i] & masks[j]).sum())
            row += f"{overlap:6d}"
        print(row)

    print(f"\nDataset: {n_samples} samples")
    print(f"Total tokens: {n_samples * tokenizer.tokens_per_sample}")
    print("=" * 60 + "\n")


def plot_codebook_histograms(tokenizer, save_path=None):
    """Plot per-scale codebook usage histograms.

    Args:
        tokenizer: fitted VQVAETokenizer
        save_path: if provided, save figure to this path instead of showing
    """
    if tokenizer.scale_usage_counts is None:
        raise ValueError("No usage counts — call fit() first.")

    n_scales = len(tokenizer.scales)
    fig, axes = plt.subplots(2, (n_scales + 1) // 2, figsize=(5 * ((n_scales + 1) // 2), 8))
    axes = axes.flatten()

    ev = tokenizer.effective_vocab_size

    for k, s in enumerate(tokenizer.scales):
        ax = axes[k]
        counts = tokenizer.scale_usage_counts[k]
        total = counts.sum()

        if total > 0:
            freq = counts / total
            ax.bar(np.arange(ev), freq, width=1.0, color='steelblue', edgecolor='none')
            ax.axhline(y=1 / ev, color='r', linestyle='--', alpha=0.7, label='Uniform')
            n_used = int((counts > 0).sum())
            ax.set_title(f"Scale {s}x{s} ({n_used}/{ev} codes)")
        else:
            ax.set_title(f"Scale {s}x{s} (no usage)")

        ax.set_xlabel("Code index")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for k in range(n_scales, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle("Codebook Usage per Scale", fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved codebook histograms to {save_path}")
    else:
        plt.show()

    return fig


# ---------- CLI ----------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize data using trained gust2 VQ-VAE")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint_dir", required=True,
                        help="Path to checkpoint directory")
    common.add_argument("--data_path", required=True,
                        help="Path to HDF5 data file")
    common.add_argument("--field", default="omega",
                        help="HDF5 field name under /fields/")
    common.add_argument("--batch_size", type=int, default=128)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--first_trainable_scale", type=int, default=None)

    save_parser = subparsers.add_parser("save", parents=[common],
                                        help="Tokenize and save to file")
    save_parser.add_argument("--output", required=True,
                             help="Output .npz path")

    info_parser = subparsers.add_parser("info", parents=[common],
                                        help="Show tokenizer stats without saving")
    info_parser.add_argument("--histogram", type=str, default=None,
                             help="Save histogram plot to this path (e.g. hist.png)")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command is None:
        print("Error: specify 'save' or 'info'")
        return

    from dataloader import VQVAEDataset

    key = jax.random.PRNGKey(args.seed)
    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    tokenizer = VQVAETokenizer.from_checkpoint(
        args.checkpoint_dir, key, args.first_trainable_scale)

    def make_dataset():
        return VQVAEDataset(
            path=args.data_path, field=args.field,
            batch_size=args.batch_size, shuffle=False)

    dataset = make_dataset()
    print(f"Dataset: {dataset.n_samples} samples, shape {dataset.sample_shape}")

    tokenizer.fit(dataset)

    if args.command == "info":
        print_tokenizer_info(tokenizer, dataset.n_samples)

        # Histogram
        fig = plot_codebook_histograms(tokenizer, save_path=args.histogram)
        plt.close(fig)

        # Round-trip verification
        print("Verifying round-trip encoding/decoding...")
        import h5py
        with h5py.File(args.data_path, "r") as f:
            sample = f[f"fields/{args.field}"][0:1].astype(np.float32)
        sample = jnp.array(sample[:, None, :, :][0])  # (1, H, W)

        remapped, _ = tokenizer.encode(sample)
        recon = tokenizer.decode_indices(remapped)
        mse = float(jnp.mean((sample - recon) ** 2))
        print(f"Round-trip MSE: {mse:.6f}")

    elif args.command == "save":
        save_tokenized_data(
            args.output, tokenizer, make_dataset(), tokenizer.arch_config)
        print_tokenizer_info(tokenizer, dataset.n_samples)


if __name__ == "__main__":
    main()
