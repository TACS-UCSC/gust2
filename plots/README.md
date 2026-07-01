# `plots/` — figure manifest (what experiment made what)

Provenance map for every directory under `plots/`, so the figure pipeline is legible. Companion to
[`../paper/FIGURES.md`](../paper/FIGURES.md) (the paper-in-plots storyboard — which *specific* file fills
each paper figure slot) and [`../paper/OUTLINE.md`](../paper/OUTLINE.md) (the prose claim spine).

**Layout convention**
- **Top level** = curated *final* figures + *diagnostic* sets that feed the paper.
- **`_archive/`** = superseded / pre-fix / stale-param runs. Kept (not deleted) but **do not cite**.
- **`_wandb_cache/`** = raw per-run PNGs downloaded from wandb (the *source* media that the composites
  in `grids/` and `scaling_report/` are tiled from). Re-pullable; not figures themselves.

Most pullers use the wandb **Python API** (entity `bigpseud-ucsc`), not the raw `wandb` CLI — the
spectra/snapshot/histogram images are logged only as *rendered* PNGs. See `pull` commands per row.

Status legend: **final** (paper-ready) · **diagnostic** (appendix / supporting) · **superseded** · **wandb-cache**.

---

## Curated final figures (top level)

| dir / file | produced by | experiment / sweep | wandb project | paper §/claim | status |
|---|---|---|---|---|---|
| `vqvae_arch.png`, `multiscale_vq.png`, `nsp_arch.png`, `attention_mask.png` | static schematics | — (hand-drawn) | — | §3, §4 | final |
| `compare_recon.png` | VQ-VAE recon comparison (Apr) | VQ-VAE recon vs sc-config | `gust2-experiments` | §3 / C3 (floor) | final |
| `compare_commit.png`, `compare_total.png`, `compare_final_bars.png` | same family | VQ-VAE loss-term comparison | `gust2-experiments` | §3 (supporting) | diagnostic |
| `codebook_analysis/` | `analyze_codebooks.py` (local, on `codebook_artifacts/` via `scripts/local/download_codebook_artifacts.sh`) | codebook geometry / utilization across 9 VQ-VAEs | — (offline EMA state) | §3 / C3 | final |
| `scaling_report/` (top-level `spectra_*`, `snapshots_*`, `table_*`, `metrics_*`) | `plot_scaling_report_grids.py` | sc341/sc917 × size × single/multi-step rollouts | `gust2-analysis-bridges-scaling` | §6 (spectra), §7, App | final |
| `paper_narrative/` (`temp_spectra_*`, `temp_snapshots_*`, `recovery_curves.png`) | `plot_paper_narrative.py` | greedy-collapse → temperature-recovery | `gust2-analysis-bridges-scaling`, `gust2-sampling-{sc341,sc917,sc1941}` | §6, §7 (below-floor), §8 | final |
| `scaling_tempopt_n128/` | `plot_scaling_tempopt.py` **with** `--projects gust2-scaling-tempopt-n128-{small,medium,large} --output_dir plots/scaling_tempopt_n128` (defaults point at the N=4 projects) / sweep `sweep_scaling_tempopt_n128.sh` | **canonical** N=128 temp-optimal scaling | `gust2-scaling-tempopt-n128-*` | §7 / C5 | final |
| `scaling_forecast/` | `plot_scaling_forecast.py` | short-horizon forecast-skill scaling | `gust2-scaling-forecast-*` | §8 (forecast) / C5 | final |
| `snapshots_n128_tempcollapse/` | `plot_snapshot_tempgrid.py` | T × lead-time collapse grid (`medium-sc917-s22/s34`) | `gust2-scaling-tempopt-n128-medium` | §8 | final |
| `snapshots_t1/` | T=1 snapshot tiler (`scripts/bridges/submit_visualize.sh`) | per-config T=1 rollout snapshots (27 cells) | `gust2-diagnostics-bridges` | §6, App | final |
| `grids/` | **ad-hoc tiler — no committed script** (Apr; tiles the per-run PNGs now in `_wandb_cache/{analysis,eval}/`) | tiled per-run spectra + histograms + T1 snapshots | (from wandb media) | §2 (metric battery), §6 | final |
| `sc341-multitraj/` (`position_ood/`, `logits_aligned/`, `survival/`) | `analyze_position_ood.py`, `analyze_logits_aligned.py`, `multitraj_survival.py` | sc341 multi-trajectory OOD + logit diagnostics | (local from rollout `.npz`) | **§5 / C1** (two modes) | final |
| `sc341-multitraj-posmask/` (`survival/`, `survival_v2/`, `T*/`) | `multitraj_survival.py` + posmask rollout sweep | per-token-mask survival + EMD traces vs no-mask | (local from rollout `.npz`) | **§5 / C2** (per-token recipe) | final |
| **`inference_samplers/`** *(new 06-23→25)* | `plot_inference_samplers.py` | adaptive samplers vs swept-T yardstick + VQ floor | `gust2-inference-samplers-*` | §8 / less-intervention | final |
| **`per_scale_temp/`** *(new 06-26→28)* | `plot_per_scale_temp.py` | static per-scale fine-heat recipe; `target_entropy_profile.png` = **T2** anchor | `gust2-inference-samplers-*` | §8 / I1,T2 | final |
| **`climate_temp_band/`** *(new 06-29)* | `plot_climate_temp_band.py` | a-priori single T≈1.8 from climate entropy band | (reads N128 csv + drift) | §8 / C5 | final |
| **`scale_drift/`** *(new 06-29→30)* | `plot_scale_drift.py` (`scale_distribution_drift.py`) | per-scale TV/JS/ΔH drift vs T, N=128 | `gust2-drift-*` | §C4 metric | final |

## Diagnostic / appendix (top level)

| dir | produced by | what | wandb project | status |
|---|---|---|---|---|
| `sc341-report/`, `sc917-report/` | `analyze_rollout.py` (on `rollout_nsp.py` T=1.0 rollouts + `eval_single_step.py` mask-aware single-step; see each dir's own `README.md`) | full per-arch rollout + single-step snapshot grids, composites, scaling, tables | `gust2-analysis-bridges-scaling` | diagnostic (App) |

## `_archive/` — superseded, do **not** cite

| item | produced by | why superseded |
|---|---|---|
| `scaling_tempopt/` | `plot_scaling_tempopt.py` (N=4) | replaced by **N=128** `scaling_tempopt_n128/` |
| `scaling_per_temp/` | `plot_scaling.py` family | per-temp scaling, replaced by N=128 |
| `scaling_sampling/` | `plot_scaling.py` (greedy) | **greedy = broken** (project rule: evaluate at T=1.0) |
| `scaling/`, `scaling_small/` | `plot_scaling.py`, `plot_scaling_bridges.py` | stale param accounting (pre-real-param-count) |
| `sc341_local/` | `plot_sc341_*`, `plot_rollout_*`, `plot_gap_curves.py` | local exploratory sc341 metrics, superseded by `scaling_report/` |
| `paper_narrative_greedy/` (`greedy_*.png`) | `plot_paper_narrative.py` | greedy-collapse panels — kept for reference, broken as finals |
| `nsp_final_loss.png`, `nsp_training_curves.png` | early NSP run | **pre-U-shape**; use `../paper/figures/loss_curves_*.png` instead |
| `stray_presentation_build/` | stray beamer build | LaTeX aux dumped into `plots/`; not figures |

## `_wandb_cache/` — raw downloaded media (re-pullable source, not figures)

| item | source | re-pull |
|---|---|---|
| `analysis/`, `eval/` | per-run spectra/enstrophy/pixel-hist PNGs | feeds `grids/` composites; re-pull via the report/visualize sweeps |
| `paper_narrative_cache/` | per-run snapshot/spectra media | `~/llm/bin/python plot_paper_narrative.py` (omit `--skip-download`) |
| `scaling_report_cache/` | per-run media for the report grids | `~/llm/bin/python plot_scaling_report_grids.py` |

---

*Curated paper-slot copies live in [`../paper/figures/`](../paper/figures/) (survive `git clean`); the wandb
pull commands for every sourced figure are in `../paper/figures/pull_from_wandb.sh`.*
