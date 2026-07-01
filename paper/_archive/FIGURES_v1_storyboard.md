# The paper, in plots — figure storyboard

**The TMLR paper (`main.tex` / [`OUTLINE.md`](OUTLINE.md)) told as a sequence of figures.** Each section
below is a paper section; each entry is a figure *slot* filled by a **verified** image, or an explicit
**`GAP`** / **`WANDB`** / **`BROKEN`** marker. Images render from the stable copies in
[`figures/`](figures/) (frozen, survive `git clean`); the **source** path + producing script is given for
provenance (full map in [`../plots/README.md`](../plots/README.md)).

Claim tags **C1–C5** and `Experiments.md` task ids (e.g. `P0-2`, `B1`) are carried on each section so this
doubles as a submission-readiness scoreboard.

### Readiness scoreboard

| § | figure slots filled | open gaps |
|---|---|---|
| 1 Intro | 1 (teaser) | optional 2-mode montage |
| 2 Setup / metric | 3 (spectra→OOD→snap→EMD) | metric-hierarchy schematic |
| 3 Tokenizer | 5 | VQ-size↔floor bar |
| 4 NSP | 4 (incl. fresh loss curves ✅) | — |
| 5 Recipe (C1,C2) | 6 | **P0-2** clean 3-way ablation |
| 6 Spectral (C3) | 2 | **B1/B2/B3** baselines (biggest exposure) |
| 7 Scaling (C5) | 4 | **P0-4** double-beat |
| 8 Temperature | 8 (incl. 4 new sweeps ✅) | **P0-3** multi-seed |
| App | grids/tables/loss | — |

Legend: **`GAP`** = no figure exists, must run/build · **`WANDB`** = pull from a `gust2-*` project ·
**`BROKEN`** = exists but do not cite (greedy / pre-fix).

---

## §1 — Introduction  ·  *teaser, all claims*

The one-figure hook: rollout metric ÷ VQ-VAE floor across the three scale-configs — greedy collapses,
temperature recovers, sc341 sits *below* the floor.

![recovery curves](figures/fig1_recovery.png)
*Source:* `plots/paper_narrative/recovery_curves.png` ← `plot_paper_narrative.py` (`gust2-analysis-bridges-scaling` + `gust2-sampling-*`).
*Optional:* a side-by-side **two-collapse-mode** montage (off-manifold vs diffusive) would sharpen the opening — build from §5 panels.

---

## §2 — Problem Setup, Data, Evaluation Protocol  ·  *C4*

Commit to the metric hierarchy **in order of evidential weight** *before* any result — shown here in that order:

**(1) spectra — primary.**
![TKE spectra battery](figures/fig2a_spectra_battery.png)
*Source:* `plots/grids/eval_tke_spectrum.png` (enstrophy companion: `plots/grids/eval_enstrophy_spectrum.png`).

**(2) per-position OOD-emission rate — the discrete-native alarm** (the figure itself is the §5 mechanism panel):
![OOD-rate alarm](figures/fig5a_position_ood.png)
*Source:* `plots/sc341-multitraj/position_ood/overlay_relative.png` ← `analyze_position_ood.py`.

**(3) snapshots** and **(4) EMD / pixel-histogram — fidelity only:**

| | |
|---|---|
| ![T=1 snapshot battery](figures/fig2_metric_battery.png) | ![pixel histogram battery](figures/fig2b_pixhist_battery.png) |
| snapshots (`plots/grids/snapshots_t1.png`) | pixel-PDF / EMD (`plots/grids/eval_pixel_histogram.png`) |

*Battery tiles are composites over the wandb media now in `plots/_wandb_cache/{analysis,eval}/`.*
- **`GAP`** metric-hierarchy *schematic* (boxes: spectra=primary … EMD=fidelity-only) — to draw.

---

## §3 — Multi-Scale VQ Tokenizer  ·  *supporting evidence for C3 (the tokenizer floor; the C3 claim itself lands in §6)*

Architecture, then the codebook evidence that the floor is **scale-resolution + det-argmax bound, not
capacity-bound** (codebook ~59% used, 0 dead codes; size barely moves the floor).

| | |
|---|---|
| ![vqvae arch](figures/fig3a_vqvae_arch.png) | ![multiscale vq](figures/fig3b_multiscale_vq.png) |
| ViT-AE schematic (`vqvae_arch.png`) | residual multi-scale VQ (`multiscale_vq.png`) |
| ![codebook utilization](figures/fig3c_codebook_util.png) | ![residual energy](figures/fig3d_codebook_residual.png) |
| utilization / dead codes (~59%, 0 dead) | per-scale residual energy |

![recon comparison](figures/fig3e_compare_recon.png)
*Sources:* `plots/codebook_analysis/{utilization_dead_codes,residual_energy}.png` ← `analyze_codebooks.py`
(offline on `codebook_artifacts/`); `plots/compare_recon.png` (VQ-VAE recon vs sc-config, `gust2-experiments`).
Also available: `codebook_analysis/{geometry_pca,utilization_zipf,utilization_perplexity}.png`.
- **`GAP`** **VQ-size ↔ floor** bar (Small/Med/Large round-trip EMD; non-monotone) — data in
  `plots/codebook_analysis/codebook_summary.csv` (A-VQ); ~10-line plot.

---

## §4 — Next-Scale Prediction Model

Backbone + attention mask + the now-fresh training curves (the top-level `nsp_*.png` were pre-U-shape and
are archived).

| | |
|---|---|
| ![nsp arch](figures/fig4a_nsp_arch.png) | ![attention mask](figures/fig4b_attention_mask.png) |
| VAR backbone + expansion/refinement heads | t0/t1 scale-causal attention mask |
| ![nsp loss](figures/loss_curves_nsp.png) | ![vqvae loss](figures/loss_curves_vqvae.png) |
| **NSP training CE** ✅ (`gust2-nsp`, 38 runs) | **VQ-VAE recon loss** ✅ (`gust2-experiments`, 9 runs) |

*Sources:* `figures/loss_curves_{nsp,vqvae}.png` ← `figures/pull_loss_curves.py` (pulled fresh this pass;
raw histories cached as `figures/loss_curves_*.csv`).

---

## §5 — Rollout Collapse: Two Failure Modes & the Per-Token Recipe  ·  **C1, C2**  ·  *the methods anchor*

The mechanism. (a) the two collapse signatures; (b)–(c) the per-token mask takes survival 0–4% → 88–100%.

![position OOD overlay](figures/fig5a_position_ood.png)
*C1, off-manifold mode:* per-position OOD-emission rate spikes ~2%→28%. Source: `plots/sc341-multitraj/position_ood/overlay_relative.png` ← `analyze_position_ood.py`.

![aligned logits overlay](figures/fig5b_logits_aligned.png)
*C1, contrast:* logit signatures — entropy↑ (off-manifold) vs entropy-cliff↓ + confidence↑ (diffusive). Source: `plots/sc341-multitraj/logits_aligned/overlay_relative.png` ← `analyze_logits_aligned.py`.

| | | |
|---|---|---|
| ![survival](figures/fig5c_survival_curves.png) | ![emd traces](figures/fig5d_emd_traces.png) | ![vs no-mask](figures/fig5e_compare_no_mask.png) |
| survival curves (per-token mask) | per-traj EMD traces | mask vs no-mask |

![multitraj grid](figures/fig5f_multitraj_grid.png)
*Per-token-mask multi-trajectory snapshot grid (T≈0.9). Source:* `plots/sc341-multitraj-posmask/T09/analysis/multitraj_grid.png`
(one per T: `T07/`,`T08/`,`T10*/`) — this fills `main.tex`'s `posmask_multitraj_grid` slot (a per-T file exists; no consolidation needed to cite).

*Sources:* `plots/sc341-multitraj-posmask/survival/{survival_curves,emd_traces,compare_no_mask}.png` ← `multitraj_survival.py`.
- **`GAP` (P0-2, headline):** one **clean 3-way ablation** {no-mask / per-scale / per-token} at a flagship
  cell, n_traj≥8, OOD-rate overlaid — current evidence is scattered April runs.

---

## §6 — Why Discrete Is Structurally Stable: Spectral Bias Relocated  ·  **C3**  ·  *strongest pillar, least-measured*

High-k energy preserved across the rollout (discrete) where continuous AR would compound the blur.

| | |
|---|---|
| ![relocated spectra](figures/fig6a_spectra_relocated.png) | ![temperature spectra](figures/fig6b_temp_spectra.png) |
| rollout spectra, sc917 medium (`scaling_report/`) | temperature-swept spectra, sc917 (`paper_narrative/`) |

*Sources:* `plots/scaling_report/spectra_sc917_medium_multistep.png` ← `plot_scaling_report_grids.py`;
`plots/paper_narrative/temp_spectra_sc917.png` ← `plot_paper_narrative.py`. Full grid:
`scaling_report/spectra_sc{341,917}_{small,medium,large}_{single,multi}step.png`, `paper_narrative/temp_spectra_sc{341,1941}.png`.
- **`GAP` (P1-6, biggest reviewer exposure):** **B1** skip-quantization continuous-latent baseline (the
  clean discrete-axis isolation), plus **B2** pixel FNO/U-Net and **B3** flat-VQ **gate** (run B3 FIRST).
  All specced in `BASELINES_SPEC.md`, none run → today C3 rests on citation.

---

## §7 — Scaling Laws & the Data-Limited Regime  ·  **C5**

D_uniq/P ≈ 0.54 (~20× tighter than text); per-tier N=128 shape; best-T rises with scale count.

| | | |
|---|---|---|
| ![best temperature](figures/fig7a_best_temperature.png) | ![scaling emd](figures/fig7b_scaling_emd.png) | ![scaling tke](figures/fig7c_scaling_tke.png) |
| best-T vs scale count | EMD vs params/tokens | TKE-RSE vs params/tokens |

*Source:* `plots/scaling_tempopt_n128/{best_temperature,scaling_emd,scaling_tke,optimal_temperature_summary}.png`
← `plot_scaling_tempopt.py` (**canonical N=128**; supersedes archived `scaling_tempopt/`). Below-floor
subsection reuses §1's `recovery_curves.png`.
- **`GAP` (P0-4):** disambiguate the unreproduced `medium-sc341-nsp-large` **double-beat**, or report only the solid EMD-beat (claim-reduction).

---

## §8 — Sampling Temperature: Climate vs Forecast  ·  *the delta since the summary*

U-shaped collapse (both extremes); warm long-run vs cold short-horizon; and **the new less-intervention
results** (a fixed per-scale schedule / a single a-priori T replace the per-config sweep).

| | |
|---|---|
| ![forecast best T](figures/fig8a_forecast_best_T.png) | ![forecast vs longterm](figures/fig8b_forecast_vs_longterm.png) |
| forecast best-T (cold) vs climate (warm) | forecast vs long-term EMD |
| ![temp collapse grid](figures/fig8c_temp_collapse.png) | ![climate band](figures/fig8d_climate_band.png) |
| U-shaped collapse, `medium-sc917-s22` | **a-priori single T≈1.8** keeps sc1941 on-manifold *(new 06-29)* |
| ![per-scale temp](figures/fig8e_per_scale_temp.png) | ![target entropy](figures/fig8f_target_entropy.png) |
| **static per-scale fine-heat** recipe *(new 06-28)* | **T2** tokenizer per-scale entropy anchor *(new)* |
| ![samplers headline](figures/fig8g_samplers_headline.png) | ![scale drift](figures/fig8h_scale_drift.png) |
| adaptive samplers vs swept-T yardstick *(new 06-25)* | per-scale TV/JS/ΔH **drift** metric *(new 06-30)* |

*Sources:* `plots/scaling_forecast/{forecast_best_temperature,forecast_vs_longterm_emd}.png` ← `plot_scaling_forecast.py`;
`plots/snapshots_n128_tempcollapse/medium-sc917-nsp-s22_temp_collapse.png` ← `plot_snapshot_tempgrid.py`;
`plots/climate_temp_band/climate_temp_band.png` ← `plot_climate_temp_band.py`;
`plots/per_scale_temp/{per_scale_temp_sc1941,target_entropy_profile}.png` ← `plot_per_scale_temp.py`;
`plots/inference_samplers/samplers_headline.png` (+`samplers_emd_grid.png`) ← `plot_inference_samplers.py`;
`plots/scale_drift/scale_drift.png` ← `plot_scale_drift.py`.
- **Honesty:** TKE/enstrophy RSE is gameable by broadband noise → pick T by EMD + PDF, never spectral RSE.
- **`GAP` (P0-3):** multi-seed confirmation of the optima (current points noisy n_traj=1 on a U).

---

## Appendix — encyclopedic material

- **Full rollout/single-step grids:** `plots/sc341-report/`, `plots/sc917-report/` (per-arch snapshots, composites, scaling).
- **Per-config T=1 snapshots:** `plots/snapshots_t1/` (27 cells).
- **Scaling tables:** `plots/scaling_report/table_sc{341,917}_{single,multi}step.tex`, `metrics_*.csv`.
- **Hyperparameters / compute:** to write (no figure).
- **Reproducibility:** every wandb-sourced figure's pull command is in [`figures/pull_from_wandb.sh`](figures/pull_from_wandb.sh).

---

### `BROKEN` — do not cite (archived)
Greedy panels (`plots/_archive/paper_narrative_greedy/greedy_*`, evaluate at T=1.0 instead); pre-U-shape
`plots/_archive/nsp_{final_loss,training_curves}.png`; N=4 `plots/_archive/scaling_tempopt/`.
