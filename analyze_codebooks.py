"""Local codebook analysis across all trained VQ-VAE tokenizers.

Diagnoses what changes about the codebook as VQ-VAE size (D=5/10/20) and
scale-config (sc341/sc917/sc1941) vary. Reads only the on-disk artifacts
(codebook embeddings + tokenized splits + config), so it runs on CPU
without any JAX/equinox import.

Inputs (created by scripts/local/download_codebook_artifacts.sh):
  codebook_artifacts/vqvae/{size}-{sc}/{ema_state.npz, config.txt}
  codebook_artifacts/tokens/{size}-{sc}{,-val}.npz

Outputs:
  plots/codebook_analysis/utilization_perplexity.png
  plots/codebook_analysis/utilization_zipf.png
  plots/codebook_analysis/utilization_dead_codes.png
  plots/codebook_analysis/geometry_pca.png
  plots/codebook_analysis/geometry_distance_hist.png
  plots/codebook_analysis/residual_energy.png
  plots/codebook_analysis/codebook_summary.csv

Usage:
    ~/llm/bin/python analyze_codebooks.py
"""

import argparse
import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

SIZES = ("small", "medium", "large")
SCS = ("sc341", "sc917", "sc1941")

# Wong CVD-safe palette — one color per scale index (max 7 scales supported).
SCALE_COLORS = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#CC79A7",  # reddish purple
]
SIZE_COLORS = {"small": "#56B4E9", "medium": "#009E73", "large": "#D55E00"}
SC_LINESTYLES = {"sc341": "-", "sc917": "--", "sc1941": ":"}

ROW_TITLES = {"sc341": "sc341 (341 tok/frame)",
              "sc917": "sc917 (917 tok/frame)",
              "sc1941": "sc1941 (1941 tok/frame)"}
COL_TITLES = {"small":  "VQ small (D=5)",
              "medium": "VQ medium (D=10)",
              "large":  "VQ large (D=20)"}


# ----------------------------------------------------------------------
# Artifact loading
# ----------------------------------------------------------------------

@dataclass
class VQVAEArtifacts:
    size: str
    sc: str
    codebook: np.ndarray             # (K, D) full codebook, L2-normalized
    cluster_sizes: np.ndarray        # (K,) training-cumulative usage
    codebook_size: int               # K  (4096)
    codebook_dim: int                # D  (512)
    scales: list                     # e.g. [1, 2, 4, 8, 16]
    splits: dict = field(default_factory=dict)  # name -> token dict


def parse_config_txt(path):
    cfg = {}
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            v = v.strip()
            try:
                cfg[k.strip()] = eval(v)  # safe for our key/value format
            except Exception:
                cfg[k.strip()] = v
    return cfg


def load_artifacts(artifacts_root):
    """Walk codebook_artifacts/ and load whatever (size, sc) pairs exist."""
    vroot = Path(artifacts_root) / "vqvae"
    troot = Path(artifacts_root) / "tokens"

    artifacts = {}
    for size in SIZES:
        for sc in SCS:
            vdir = vroot / f"{size}-{sc}"
            ema_path = vdir / "ema_state.npz"
            cfg_path = vdir / "config.txt"
            if not (ema_path.exists() and cfg_path.exists()):
                continue

            ema = np.load(ema_path)
            cfg = parse_config_txt(cfg_path)

            obj = VQVAEArtifacts(
                size=size, sc=sc,
                codebook=ema["codebook"].astype(np.float32),
                cluster_sizes=ema["cluster_sizes"].astype(np.float32),
                codebook_size=int(cfg["codebook_size"]),
                codebook_dim=int(cfg["codebook_dim"]),
                scales=list(cfg["scales"]),
            )

            for split_name, suffix in (("train", ""), ("val", "-val")):
                tpath = troot / f"{size}-{sc}{suffix}.npz"
                if not tpath.exists():
                    continue
                t = dict(np.load(tpath, allow_pickle=True))
                obj.splits[split_name] = t

            if obj.splits:
                artifacts[(size, sc)] = obj
                print(f"  loaded {size}-{sc}: scales={obj.scales}, "
                      f"splits={list(obj.splits.keys())}, "
                      f"|codebook|={obj.codebook.shape}")
    return artifacts


# ----------------------------------------------------------------------
# Utilization statistics
# ----------------------------------------------------------------------

def per_scale_usage(token_dict, scale_idx, scale_value):
    """Return bincount over compact indices for one scale, masked to the
    scale's allowed codes.

    Returns:
        counts          : (n_valid_codes_for_scale,) int array of usage
        scale_mask_sum  : int, |allowed codes for this scale|
        n_tokens_total  : int, total tokens drawn at this scale
    """
    indices = token_dict[f"indices_scale_{scale_value}"]  # (N, s²)
    effective_vocab = int(token_dict["effective_vocab_size"])
    scale_masks = token_dict["scale_masks"]               # (n_scales, eff)

    bins = np.bincount(indices.ravel(), minlength=effective_vocab)
    valid_mask = scale_masks[scale_idx].astype(bool)
    valid_counts = bins[valid_mask]
    return valid_counts, int(valid_mask.sum()), int(indices.size)


def utilization_stats(counts, allowed_n):
    """Reduce a per-scale count vector to a handful of summary scalars."""
    total = counts.sum()
    if total == 0:
        return dict(perplexity=1.0, entropy_bits=0.0,
                    effective_vocab=0, dead_fraction=1.0,
                    top1_share=0.0, allowed_n=allowed_n)
    p = counts / total
    nz = p[p > 0]
    entropy_bits = float(-(nz * np.log2(nz)).sum())
    return dict(
        perplexity=float(2 ** entropy_bits),
        entropy_bits=entropy_bits,
        effective_vocab=int((counts > 0).sum()),
        dead_fraction=float(1 - (counts > 0).sum() / max(allowed_n, 1)),
        top1_share=float(counts.max() / total),
        allowed_n=allowed_n,
    )


def collect_utilization(artifacts):
    """Long-form list of dicts; one row per (size, sc, split, scale_idx)."""
    rows = []
    for (size, sc), obj in artifacts.items():
        for split_name, t in obj.splits.items():
            for k_idx, s_val in enumerate(obj.scales):
                counts, allowed, total = per_scale_usage(t, k_idx, s_val)
                stats = utilization_stats(counts, allowed)
                stats.update(
                    size=size, sc=sc, split=split_name,
                    scale_idx=k_idx, scale_value=int(s_val),
                    n_tokens=total,
                    counts=counts,   # kept for the Zipf plot
                )
                rows.append(stats)
    return rows


# ----------------------------------------------------------------------
# Geometry statistics
# ----------------------------------------------------------------------

def codebook_geometry(codebook, rng, n_pairs=200_000):
    """Norms, SVD spectrum, pairwise cosine-distance histogram."""
    K, D = codebook.shape
    norms = np.linalg.norm(codebook, axis=1)

    # Centered SVD → variance spectrum.
    centered = codebook - codebook.mean(axis=0, keepdims=True)
    s = np.linalg.svd(centered, compute_uv=False)
    var = s ** 2
    var_frac = var / var.sum()
    cumvar = np.cumsum(var_frac)
    # Effective dim: exp of entropy of the variance distribution.
    nz = var_frac[var_frac > 0]
    eff_dim = float(np.exp(-(nz * np.log(nz)).sum()))

    # Pairwise cosine distance on a random sample. EMA updates don't
    # preserve unit norm (codebook = sums / counts), so normalize explicitly.
    cb_unit = codebook / np.maximum(norms[:, None], 1e-12)
    i = rng.integers(0, K, size=n_pairs)
    j = rng.integers(0, K, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    cos_sim = (cb_unit[i] * cb_unit[j]).sum(axis=1)
    cos_dist = 1.0 - cos_sim

    return dict(
        norms=norms,
        sv=s,
        var_frac=var_frac,
        cumvar=cumvar,
        eff_dim=eff_dim,
        cos_dist=cos_dist,
    )


def per_scale_centroids(obj, split="val"):
    """Per-scale centroid of code-space embeddings, weighted by usage.

    Useful diagnostic: if scales sit at very different centroids, the
    codebook is *partitioned* between scales rather than shared.
    """
    if split not in obj.splits:
        return None
    t = obj.splits[split]
    new_to_old = t["new_to_old"]
    full_cb = obj.codebook  # (K, D), original index space
    out = []
    for k_idx, s_val in enumerate(obj.scales):
        idx = t[f"indices_scale_{s_val}"].ravel()
        raw_idx = new_to_old[idx]            # back to 0..K-1 space
        cent = full_cb[raw_idx].mean(axis=0)  # (D,)
        out.append(cent)
    return np.stack(out, axis=0)  # (n_scales, D)


# ----------------------------------------------------------------------
# Residual-energy statistics (latent-level, cheap)
# ----------------------------------------------------------------------

def residual_energy_per_scale(obj, split="val", n_frames=256):
    """Average per-scale latent L2 energy = mean ||z_q^{(k)}||² over the
    s² positions and over n_frames frames.

    In residual VQ each scale's codebook is fit to the residual of the
    previous scales' reconstruction, so this gives a direct reading of
    how much variance each scale carries.
    """
    if split not in obj.splits:
        return None
    t = obj.splits[split]
    compact_cb = t["codebook"]  # (effective_vocab, D)

    energies = []
    for k_idx, s_val in enumerate(obj.scales):
        idx = t[f"indices_scale_{s_val}"][:n_frames]  # (n_frames, s²)
        z = compact_cb[idx]                            # (n_frames, s², D)
        energy = (z ** 2).sum(axis=-1).mean()
        energies.append(float(energy))
    return np.array(energies)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def _grid_axes(present_pairs, figsize=(15, 13)):
    """Build a 3×3 fig with row=sc, col=size; greys out missing cells."""
    fig, axes = plt.subplots(len(SCS), len(SIZES), figsize=figsize,
                             squeeze=False)
    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            ax = axes[i, j]
            if i == 0:
                ax.set_title(COL_TITLES[size], fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(ROW_TITLES[sc] + "\n", fontsize=11)
            if (size, sc) not in present_pairs:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center",
                        transform=ax.transAxes, color="0.5")
                ax.set_xticks([]); ax.set_yticks([])
    return fig, axes


def plot_utilization_perplexity(rows, artifacts, out_path):
    """Per-scale perplexity bar chart per (size, sc). Val solid, train hatched."""
    present = set(artifacts.keys())
    fig, axes = _grid_axes(present, figsize=(15, 13))

    by_key = {}
    for r in rows:
        by_key.setdefault((r["size"], r["sc"], r["split"]), []).append(r)

    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            if (size, sc) not in present: continue
            ax = axes[i, j]
            obj = artifacts[(size, sc)]
            n_scales = len(obj.scales)
            x = np.arange(n_scales)
            w = 0.4

            val_rows   = sorted(by_key.get((size, sc, "val"),   []),
                                key=lambda r: r["scale_idx"])
            train_rows = sorted(by_key.get((size, sc, "train"), []),
                                key=lambda r: r["scale_idx"])

            if val_rows:
                ax.bar(x - w/2, [r["perplexity"] for r in val_rows], w,
                       color=[SCALE_COLORS[r["scale_idx"]] for r in val_rows],
                       label="val", edgecolor="black", linewidth=0.4)
            if train_rows:
                ax.bar(x + w/2, [r["perplexity"] for r in train_rows], w,
                       color=[SCALE_COLORS[r["scale_idx"]] for r in train_rows],
                       hatch="//", label="train",
                       edgecolor="black", linewidth=0.4, alpha=0.85)

            # Annotate each bar with effective_vocab / allowed.
            for r in val_rows:
                ax.text(r["scale_idx"] - w/2, r["perplexity"],
                        f"{r['effective_vocab']}/{r['allowed_n']}",
                        ha="center", va="bottom", fontsize=8, rotation=0)

            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in obj.scales])
            ax.set_yscale("log")
            ax.grid(True, axis="y", alpha=0.3)
            if i == len(SCS) - 1:
                ax.set_xlabel("scale (grid side)", fontsize=10)
            if j == 0:
                ax.set_ylabel(ROW_TITLES[sc] + "\nperplexity (log)", fontsize=10)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc="0.7", ec="black", label="val"),
        plt.Rectangle((0, 0), 1, 1, fc="0.7", ec="black", hatch="//",
                      label="train"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=11, frameon=False)
    fig.suptitle("Per-scale codebook perplexity (annotated: "
                 "effective_vocab / allowed)\n"
                 "Bars colored by scale index — same-color bars are the "
                 "same logical scale across panels",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.025, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_utilization_zipf(rows, artifacts, out_path):
    """Zipf log-rank/log-freq per scale. Use val only for clarity."""
    present = set(artifacts.keys())
    fig, axes = _grid_axes(present, figsize=(15, 13))

    by_key = {}
    for r in rows:
        if r["split"] != "val": continue
        by_key.setdefault((r["size"], r["sc"]), []).append(r)

    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            if (size, sc) not in present: continue
            ax = axes[i, j]
            obj = artifacts[(size, sc)]
            rs = sorted(by_key.get((size, sc), []),
                        key=lambda r: r["scale_idx"])
            for r in rs:
                c = r["counts"]
                c = np.sort(c)[::-1]
                c = c[c > 0]
                if len(c) == 0: continue
                rank = np.arange(1, len(c) + 1)
                ax.plot(rank, c, color=SCALE_COLORS[r["scale_idx"]],
                        linewidth=1.5, alpha=0.85,
                        label=f"scale {r['scale_value']}")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.3)
            if i == len(SCS) - 1:
                ax.set_xlabel("rank (log)", fontsize=10)
            if j == 0:
                ax.set_ylabel(ROW_TITLES[sc] + "\nfreq (log)", fontsize=10)
            ax.legend(fontsize=7, ncol=2, loc="lower left")

    fig.suptitle("Codebook usage Zipf curves (val) — one line per scale",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_utilization_dead_codes(rows, artifacts, out_path):
    present = set(artifacts.keys())
    fig, axes = _grid_axes(present, figsize=(15, 11))

    by_key = {}
    for r in rows:
        if r["split"] != "val": continue
        by_key.setdefault((r["size"], r["sc"]), []).append(r)

    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            if (size, sc) not in present: continue
            ax = axes[i, j]
            obj = artifacts[(size, sc)]
            rs = sorted(by_key.get((size, sc), []),
                        key=lambda r: r["scale_idx"])
            x = np.arange(len(rs))
            colors = [SCALE_COLORS[r["scale_idx"]] for r in rs]
            vals = [r["dead_fraction"] for r in rs]
            ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels([str(r["scale_value"]) for r in rs])
            ax.set_ylim(0, 1.0)
            ax.grid(True, axis="y", alpha=0.3)
            if i == len(SCS) - 1:
                ax.set_xlabel("scale", fontsize=10)
            if j == 0:
                ax.set_ylabel(ROW_TITLES[sc] + "\ndead-code frac", fontsize=10)
            for k, v in enumerate(vals):
                ax.text(k, v + 0.02, f"{v:.2f}", ha="center",
                        fontsize=8)

    fig.suptitle("Per-scale dead-code fraction (val) — "
                 "fraction of scale's allowed codes never used",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_geometry_pca(geom_table, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax_cv, ax_nh = axes

    for (size, sc), g in geom_table.items():
        x = np.arange(1, len(g["cumvar"]) + 1)
        ax_cv.plot(x, g["cumvar"],
                   color=SIZE_COLORS[size],
                   linestyle=SC_LINESTYLES[sc],
                   linewidth=1.8, alpha=0.9)
    ax_cv.set_xscale("log")
    ax_cv.set_xlabel("rank", fontsize=11)
    ax_cv.set_ylabel("cumulative explained variance", fontsize=11)
    ax_cv.set_title("Codebook PCA: cumulative variance vs rank",
                    fontsize=12, fontweight="bold")
    ax_cv.grid(True, which="both", alpha=0.3)
    ax_cv.set_ylim(0, 1.02)

    # Norm histogram (overlay).
    for (size, sc), g in geom_table.items():
        ax_nh.hist(g["norms"], bins=40, histtype="step",
                   color=SIZE_COLORS[size],
                   linestyle=SC_LINESTYLES[sc],
                   linewidth=1.4, alpha=0.85)
    ax_nh.set_xlabel("||codebook entry||₂", fontsize=11)
    ax_nh.set_ylabel("# codes", fontsize=11)
    ax_nh.set_title("Codebook L2-norm distribution", fontsize=12,
                    fontweight="bold")
    ax_nh.grid(True, alpha=0.3)

    handles = []
    for size in SIZES:
        for sc in SCS:
            if (size, sc) in geom_table:
                label = f"{size} / {sc} (eff dim = {geom_table[(size, sc)]['eff_dim']:.0f})"
                handles.append(Line2D([0], [0], color=SIZE_COLORS[size],
                                      linestyle=SC_LINESTYLES[sc],
                                      linewidth=2.0, label=label))
    fig.legend(handles=handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_geometry_distance_hist(geom_table, out_path):
    present = set(geom_table.keys())
    fig, axes = _grid_axes(present, figsize=(15, 11))

    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            if (size, sc) not in present: continue
            ax = axes[i, j]
            d = geom_table[(size, sc)]["cos_dist"]
            ax.hist(d, bins=80, color=SIZE_COLORS[size], alpha=0.85,
                    edgecolor="black", linewidth=0.3)
            ax.axvline(d.mean(), color="black", linestyle="--",
                       linewidth=1.2, alpha=0.7)
            ax.text(0.98, 0.95,
                    f"mean = {d.mean():.3f}\n"
                    f"eff dim = {geom_table[(size, sc)]['eff_dim']:.0f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85,
                              ec="0.6"))
            ax.grid(True, alpha=0.3)
            if i == len(SCS) - 1:
                ax.set_xlabel("cosine distance (1 − sim)", fontsize=10)

    fig.suptitle("Pairwise cosine-distance histogram (random 200k pairs)\n"
                 "Tighter / lower-mean distribution = more clustered codebook",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_residual_energy(energy_table, artifacts, out_path):
    present = set(artifacts.keys())
    fig, axes = _grid_axes(present, figsize=(15, 11))

    for i, sc in enumerate(SCS):
        for j, size in enumerate(SIZES):
            if (size, sc) not in present: continue
            obj = artifacts[(size, sc)]
            e = energy_table.get((size, sc))
            ax = axes[i, j]
            if e is None: continue
            x = np.arange(1, len(e) + 1)
            ax.bar(x, e, color=[SCALE_COLORS[k] for k in range(len(e))],
                   edgecolor="black", linewidth=0.4)
            # Cumulative line.
            cum = np.cumsum(e)
            ax2 = ax.twinx()
            ax2.plot(x, cum / cum[-1], "o-k", linewidth=1.4,
                     markersize=4, alpha=0.85)
            ax2.set_ylim(0, 1.05)
            ax2.set_ylabel("cumulative frac", fontsize=9, color="0.3")
            ax2.tick_params(axis="y", labelsize=8, colors="0.3")

            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in obj.scales])
            ax.grid(True, axis="y", alpha=0.3)
            if i == len(SCS) - 1:
                ax.set_xlabel("scale (grid side)", fontsize=10)
            if j == 0:
                ax.set_ylabel(ROW_TITLES[sc] + "\nmean ‖z_q^{(k)}‖²",
                              fontsize=10)

    fig.suptitle("Per-scale residual energy (mean ‖z_q‖² over val frames)\n"
                 "Cumulative fraction overlaid on right axis — "
                 "where each VQ spends its bits",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# CSV summary
# ----------------------------------------------------------------------

def write_summary_csv(rows, geom_table, energy_table, out_path):
    fieldnames = [
        "size", "sc", "split", "scale_idx", "scale_value",
        "allowed_n", "effective_vocab", "dead_fraction",
        "perplexity", "entropy_bits", "top1_share", "n_tokens",
        "residual_energy", "geom_eff_dim", "geom_mean_cos_dist",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            key = (r["size"], r["sc"])
            e = energy_table.get(key)
            g = geom_table.get(key)
            row = {k: r[k] for k in (
                "size", "sc", "split", "scale_idx", "scale_value",
                "allowed_n", "effective_vocab", "dead_fraction",
                "perplexity", "entropy_bits", "top1_share", "n_tokens",
            )}
            row["residual_energy"] = (float(e[r["scale_idx"]])
                                      if e is not None
                                      and r["scale_idx"] < len(e)
                                      else "")
            row["geom_eff_dim"] = float(g["eff_dim"]) if g else ""
            row["geom_mean_cos_dist"] = (float(g["cos_dist"].mean())
                                          if g else "")
            w.writerow(row)
    print(f"  saved {out_path}")


# ----------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------

def sanity_checks(artifacts):
    """Verifications listed in the plan."""
    print("\nSanity checks:")
    for (size, sc), obj in artifacts.items():
        # Codebook shape + normalization.
        K, D = obj.codebook.shape
        norms = np.linalg.norm(obj.codebook, axis=1)
        unit_frac = np.mean(np.isclose(norms, 1.0, atol=1e-3))
        # Per-split shape match.
        for split, t in obj.splits.items():
            expected = sum(s ** 2 for s in obj.scales)
            actual = int(t["indices_flat"].shape[1])
            eff_vocab_stored = int(t["effective_vocab_size"])
            # Recompute effective vocab from flat indices.
            eff_vocab_recompute = int(np.unique(t["indices_flat"]).size)
            ok = (actual == expected
                  and eff_vocab_recompute <= eff_vocab_stored)
            tag = "ok" if ok else "FAIL"
            print(f"  [{tag}] {size}-{sc}-{split}: "
                  f"Σs² {actual} (expected {expected}); "
                  f"eff_vocab stored={eff_vocab_stored} recomputed≤={eff_vocab_recompute}; "
                  f"|cb|={K}×{D}; norms-unit-frac={unit_frac:.3f}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts_dir", default="codebook_artifacts",
                        help="local mirror produced by "
                             "scripts/local/download_codebook_artifacts.sh")
    parser.add_argument("--output_dir", default="plots/codebook_analysis")
    parser.add_argument("--n_pairs", type=int, default=200_000,
                        help="pairs sampled for cosine-distance hist")
    parser.add_argument("--energy_frames", type=int, default=256,
                        help="val frames used for residual-energy estimate")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading artifacts from {args.artifacts_dir}/")
    artifacts = load_artifacts(args.artifacts_dir)
    if not artifacts:
        print("\nNo artifacts found. Run "
              "scripts/local/download_codebook_artifacts.sh first.")
        return

    sanity_checks(artifacts)

    print("\nComputing per-scale utilization...")
    rows = collect_utilization(artifacts)

    print("Computing codebook geometry...")
    geom_table = {}
    for k, obj in artifacts.items():
        geom_table[k] = codebook_geometry(obj.codebook, rng,
                                          n_pairs=args.n_pairs)

    print("Computing per-scale residual energy (val)...")
    energy_table = {}
    for k, obj in artifacts.items():
        e = residual_energy_per_scale(obj, split="val",
                                      n_frames=args.energy_frames)
        if e is not None:
            energy_table[k] = e

    print("\nRendering plots...")
    plot_utilization_perplexity(rows, artifacts,
        os.path.join(args.output_dir, "utilization_perplexity.png"))
    plot_utilization_zipf(rows, artifacts,
        os.path.join(args.output_dir, "utilization_zipf.png"))
    plot_utilization_dead_codes(rows, artifacts,
        os.path.join(args.output_dir, "utilization_dead_codes.png"))
    plot_geometry_pca(geom_table,
        os.path.join(args.output_dir, "geometry_pca.png"))
    plot_geometry_distance_hist(geom_table,
        os.path.join(args.output_dir, "geometry_distance_hist.png"))
    plot_residual_energy(energy_table, artifacts,
        os.path.join(args.output_dir, "residual_energy.png"))
    write_summary_csv(rows, geom_table, energy_table,
        os.path.join(args.output_dir, "codebook_summary.csv"))

    print("\nDone. All outputs in", args.output_dir)


if __name__ == "__main__":
    main()
