#!/usr/bin/env python
"""Pull NSP + VQ-VAE training-loss curves from wandb and render clean figures.

These are the one genuinely-missing wandb figure for the paper (the top-level
nsp_*.png are pre-U-shape and must not be reused). Training loss only — wandb
projects gust2-nsp / gust2-experiments log no validation curve (eval lives in
gust2-eval). Run:  ~/llm/bin/python paper/figures/pull_loss_curves.py

Outputs (paper/figures/):
  loss_curves_nsp.png      total CE vs step, faceted by sc-config, line per run
  loss_curves_vqvae.png    reconstruction loss vs step, line per VQ-VAE run
  loss_curves_nsp.csv / loss_curves_vqvae.csv   raw history (reproducibility)
"""
import os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

ENTITY = "bigpseud-ucsc"
OUT = os.path.dirname(os.path.abspath(__file__))
SC = ["sc341", "sc917", "sc1941"]
SC_COL = {"sc341": "C0", "sc917": "C1", "sc1941": "C2"}
SIZE_LS = {"small": "-", "medium": "--", "large": ":", "micro": "-.",
           "mini": (0, (1, 1)), "nano": (0, (3, 1, 1, 1))}


def sc_of(name):
    m = re.search(r"sc(341|917|1941)", name)
    return f"sc{m.group(1)}" if m else None


def fetch(project, ykey):
    api = wandb.Api(timeout=30)
    runs = api.runs(f"{ENTITY}/{project}")
    out = []
    for r in runs:
        try:
            h = r.history(keys=[ykey], samples=2000, pandas=True)
        except Exception as e:
            print(f"  [warn] {r.name}: history failed ({e})")
            continue
        if h is None or ykey not in getattr(h, "columns", []) or len(h) == 0:
            print(f"  [skip] {r.name}: no '{ykey}'")
            continue
        step = h["_step"].to_numpy() if "_step" in h.columns else np.arange(len(h))
        y = h[ykey].to_numpy()
        m = np.isfinite(y)
        out.append((r.name, step[m], y[m]))
        print(f"  [ok]   {r.name}: {m.sum()} pts")
    return out


def save_csv(path, series, ykey):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", ykey])
        for name, step, y in series:
            for s, v in zip(step, y):
                w.writerow([name, int(s), float(v)])
    print(f"  wrote {path}")


def plot_nsp(series):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, sc in zip(axes, SC):
        runs = [(n, s, y) for (n, s, y) in series if sc_of(n) == sc]
        for name, step, y in sorted(runs, key=lambda t: t[0]):
            size = next((k for k in SIZE_LS if f"-nsp-{k}" in name or name.endswith(k)), "small")
            vq = name.split("-")[0]
            ax.plot(step, y, ls=SIZE_LS.get(size, "-"), lw=1.3,
                    label=name.replace(f"-{sc}", "").replace("nsp-", ""))
        ax.set_title(f"{sc}  (n={len(runs)})")
        ax.set_xlabel("training step")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        if runs:
            ax.legend(fontsize=6, ncol=2)
    axes[0].set_ylabel("total CE loss")
    fig.suptitle("NSP training loss (gust2-nsp)", y=1.02)
    dest = os.path.join(OUT, "loss_curves_nsp.png")
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    print(f"  wrote {dest}")


def plot_vqvae(series):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, step, y in sorted(series, key=lambda t: t[0]):
        sc = sc_of(name)
        size = name.split("-")[0]
        ax.plot(step, y, color=SC_COL.get(sc, "k"),
                ls=SIZE_LS.get(size, "-"), lw=1.4, label=name)
    ax.set_xlabel("training step")
    ax.set_ylabel("reconstruction loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("VQ-VAE reconstruction loss (gust2-experiments)\n"
                 "color = sc-config, linestyle = enc/dec size")
    dest = os.path.join(OUT, "loss_curves_vqvae.png")
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    print(f"  wrote {dest}")


def main():
    print("== NSP (gust2-nsp), key=loss ==")
    nsp = fetch("gust2-nsp", "loss")
    if nsp:
        save_csv(os.path.join(OUT, "loss_curves_nsp.csv"), nsp, "loss")
        plot_nsp(nsp)
    else:
        print("  NO NSP loss series found")
    print("== VQ-VAE (gust2-experiments), key=loss/reconstruction ==")
    vq = fetch("gust2-experiments", "loss/reconstruction")
    if vq:
        save_csv(os.path.join(OUT, "loss_curves_vqvae.csv"), vq, "loss/reconstruction")
        plot_vqvae(vq)
    else:
        print("  NO VQ-VAE recon series found")


if __name__ == "__main__":
    main()
