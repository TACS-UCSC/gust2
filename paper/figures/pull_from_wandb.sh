#!/usr/bin/env bash
# Regenerate every wandb-sourced paper figure (entity bigpseud-ucsc).
# Figures are logged as *rendered* PNGs -> pulled via the wandb Python API in the
# plot_*.py scripts (the raw `wandb` CLI cannot export logged media). Run from repo root:
#     bash paper/figures/pull_from_wandb.sh
# Each block is independent; a missing project warns and continues. After pulling,
# re-curate the paper/figures/ slot copies with the cp block at the bottom (or rerun
# the storyboard's source mapping).
set -uo pipefail
cd "$(dirname "$0")/../.."          # -> repo root
PY=~/llm/bin/python
run() { echo; echo "==> $*"; "$@" || echo "   [warn] block failed (missing project / network?) — continuing"; }

# --- §4 training curves (the one genuinely-missing figure) ---
run $PY paper/figures/pull_loss_curves.py                      # gust2-nsp, gust2-experiments

# --- §6/§7/§8 greedy->recovery narrative (temp_spectra, temp_snapshots, recovery_curves) ---
run $PY plot_paper_narrative.py                                # gust2-analysis-bridges-scaling + gust2-sampling-*

# --- §6/§7 rollout spectra + snapshot grids + tables ---
run $PY plot_scaling_report_grids.py                           # gust2-analysis-bridges-scaling

# --- §7 canonical N=128 temp-optimal scaling ---
run $PY plot_scaling_tempopt.py \
        --projects gust2-scaling-tempopt-n128-small gust2-scaling-tempopt-n128-medium gust2-scaling-tempopt-n128-large \
        --output_dir plots/scaling_tempopt_n128

# --- §8 short-horizon forecast scaling ---
run $PY plot_scaling_forecast.py                               # gust2-scaling-forecast-{small,medium,large}

# --- §8 temperature x lead-time collapse grid (one cell shown in paper) ---
run $PY plot_snapshot_tempgrid.py --project gust2-scaling-tempopt-n128-medium --cell medium-sc917-nsp-s22

# --- §8 new sweeps (since 06-23) ---
run $PY plot_inference_samplers.py                             # gust2-inference-samplers-{small,medium,large}
run $PY plot_per_scale_temp.py --sc sc1941                     # gust2-inference-samplers-*
run $PY plot_scale_drift.py                                    # gust2-drift-{small,medium,large}
run $PY plot_climate_temp_band.py                              # reads N128 csv + drift

# --- §3 codebook analysis: needs the EMA-state mirror first (rsync, not wandb) ---
echo; echo "==> §3 codebook: PSC_USER=sambamur bash scripts/local/download_codebook_artifacts.sh && $PY analyze_codebooks.py"

echo; echo "Done. Re-curate slot copies for paper/figures/ if any source changed (see paper/FIGURES.md source map)."
