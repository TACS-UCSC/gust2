"""Pull pixel_histogram PNGs from wandb for medium-sc341 across NSP archs
and assemble them into a single 2×2 figure.

Used to visualize the "AR rollout marginal beats VQ-VAE floor" finding —
each panel overlays GT, VQ-VAE round-trip, and NSP rollout pixel
distributions for one NSP arch.

Output: plots/sc341_local/medium_sc341_histograms.png

Usage:
    ~/llm/bin/python plot_sc341_histograms.py
"""

import argparse
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import wandb

ENTITY = "bigpseud-ucsc"
PROJECT = "gust2-analysis-bridges-scaling"

ARCHS = ["s06", "s13", "s18", "s24"]
ARCH_LABELS = {
    "s06": "s06 (6M, EMD=0.325, gap=1.45×)",
    "s13": "s13 (13M, EMD=0.152, gap=0.67×) — below floor",
    "s18": "s18 (18M, EMD=0.127, gap=0.56×) — below floor",
    "s24": "s24 (24M, EMD=0.192, gap=0.85×) — below floor",
}


def fetch_histogram(api, run_name, cache_dir):
    runs = list(api.runs(f"{ENTITY}/{PROJECT}",
                         filters={"display_name": run_name}))
    runs = [r for r in runs if r.name == run_name]
    if not runs:
        return None
    files = list(runs[0].files())
    hist_files = [f for f in files
                  if "pixel_histogram" in f.name and f.name.endswith(".png")]
    if not hist_files:
        return None
    f = hist_files[0]
    f.download(root=cache_dir, replace=True)
    return os.path.join(cache_dir, f.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="plots/sc341_local")
    parser.add_argument("--cache_dir",  default="/tmp/sc341_histograms")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    api = wandb.Api()

    paths = {}
    for arch in ARCHS:
        run_name = f"medium-sc341-nsp-{arch}"
        cache_subdir = os.path.join(args.cache_dir, arch)
        path = fetch_histogram(api, run_name, cache_subdir)
        if path is None:
            print(f"  [skip] {run_name}: no pixel_histogram found")
            continue
        paths[arch] = path
        print(f"  fetched {run_name} -> {path}")

    if len(paths) < 4:
        print(f"\n  warning: only {len(paths)}/4 histograms available")

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, arch in zip(axes.flatten(), ARCHS):
        if arch not in paths:
            ax.text(0.5, 0.5, "(no data)", ha="center", va="center",
                    transform=ax.transAxes, color="0.5")
            ax.axis("off")
            continue
        img = mpimg.imread(paths[arch])
        ax.imshow(img)
        ax.set_title(ARCH_LABELS[arch], fontsize=12, fontweight="bold")
        ax.axis("off")

    fig.suptitle("medium-sc341: pixel-value histograms vs NSP capacity\n"
                 "GT (blue) / VQ-VAE round-trip (green dashed) / NSP rollout (red dotted)",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(args.output_dir, "medium_sc341_histograms.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out}")


if __name__ == "__main__":
    main()
