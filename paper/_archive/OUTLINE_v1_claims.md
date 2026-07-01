# gust2 — TMLR paper outline & structure

**Target venue:** TMLR (Transactions on Machine Learning Research) — *not* a conference.
**What this directory is:** the LaTeX scaffold (`main.tex` + vendored `tmlr.sty/.bst`, `fancyhdr.sty`,
`math_commands.tex`) and this structural outline. No paper prose has been written yet — `main.tex` is
section headers + per-section content notes + figure stubs pointing at the existing `../plots` assets.

Compile: `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`.

---

## A. What TMLR actually grades (this shapes everything below)

TMLR has **exactly two acceptance criteria**, and the structure should serve them directly:

1. **Claims ↔ Evidence** (the dominant axis): *"Are the claims supported by accurate, convincing, clear
   evidence?"* Any gap must be closed **either** by adding evidence **or by reducing the claim.**
2. **Audience interest**: would *some* of TMLR's audience find the findings interesting?

Consequences for us (all confirmed against the current author guide / acceptance-criteria pages):
- **Novelty / SOTA / "significance" are explicitly NOT criteria.** Drop any novelty-forward framing;
  the scoop anxiety in the framing notes is irrelevant to acceptance (it only matters for the optional
  post-acceptance *certifications*). Lead with claims-and-evidence, not "first to do X."
- **No page limit** — length must be "justified by content"; over-long *main* text risks review delays.
  → Recommended reconciliation with the Llama-tech-report instinct: **disciplined main text (the claims
  that have evidence now), encyclopedic material pushed to the unbounded appendix** (read at reviewer
  discretion). This keeps the tech-report breadth without tripping the length norm.
- **Claim-reduction is a sanctioned escape hatch.** The unreproduced double-beat (P0-4) and the
  pinned-forecast-optimum lower bound become *narrowed claims*, not blockers.
- **Double-blind:** `\usepackage{tmlr}` auto-anonymizes; strip Acknowledgments / Author Contributions
  until camera-ready (`\usepackage[accepted]{tmlr}`). arXiv preprint allowed via `[preprint]`.
- **Broader Impact Statement** only required if significant harm risk (a turbulence surrogate is
  low-risk → a 2–3 line optional statement, or omit). No mandatory NeurIPS-style checklist; reproducibility
  is rewarded via supplementary code (≤100 MB) and the optional Reproducibility Certification.

---

## B. The claims spine (C1–C5) — the load-bearing artifact

Organize the paper around these claims; every results subsection should map to one. Status legend:
**[E]** evidence in hand · **[P]** partial / needs cleanup · **[G]** gap, must run or reduce claim.

| # | Claim | Status | Evidence we have | Evidence still needed (→ §F) |
|---|-------|:------:|------------------|------------------------------|
| **C1** | Two **distinct** rollout collapse modes — positional-OOD (off-manifold, entropy ↑) vs diffusive/mean-reversion (in-distribution, entropy cliff ↓ + confidence ↑). Opposite signatures; must not be conflated. | **[E]** | `sc341-multitraj/position_ood/`, `logits_aligned/`; posmask-temp diagnostics (logit signatures) | tighten the two-mode contrast figure |
| **C2** | In VAR-style **parallel** scale emission the support constraint must be **per-token, not per-scale**; per-token CE mask + 10% substitution noise → survival 0–4% ⇒ 88–100%. | **[P]** | `posmask_multitraj_grid`, `robust_*` figures | **P0-2:** one clean 3-way ablation {no-mask / per-scale / per-token}, n_traj≥8, OOD-rate overlaid (current evidence = scattered April runs) |
| **C3** | Spectral bias is **relocated** (paid once in the tokenizer), not removed; discrete latent AR avoids compounding it, continuous (pixel & latent) AR does not. | **[G]** | spectra grids; tokenizer spectral-floor argument (scale-resolution + det-quant; codebook only ~59% used) | **P1-6 (biggest exposure):** B1 skip-quantization baseline turns this from *citation* into a *measurement*. Also B2/B3. |
| **C4** | Per-position **OOD-emission rate** is a discrete-native rollout-health alarm; spectra are the primary stability metric; EMD is necessary-but-insufficient. | **[E]** | OOD-rate fired 2%→28% at real collapses; sc917 EMD over-flag retraction; time-resolved spectra | **P0-1:** finalize the time-resolved-spectrum + OOD-rate overlay on the same axis |
| **C5** | Compute-optimal AR-token loss on a PDE dataset: **D_uniq/P ≈ 0.54** (~20× tighter than text); per-tier scaling shape; below-floor EMD on sc341. | **[E/P]** | canonical **N=128** sweep (`scaling_tempopt_n128/`), forecast sweep | **P0-4:** disambiguate or drop the medium-sc341-large double-beat; **P0-3:** multi-seed temp optima |

Secondary findings worth their own subsections (not core claims): **below-floor EMD via mode-averaging**
(sc341 only; sc917 open) and the **climate-vs-forecast temperature split** (warm long-run / cold short-horizon).

---

## C. Section-by-section outline

Mirrors `main.tex`. Each entry: purpose → contents → claim(s) → figures (real paths) → status.

0. **Abstract** — state the metric (post-Lyapunov statistical stability, not RMSE), the structural-vs-numerical
   thesis, the per-token fix, and the headline secondary findings. ~150–200 words. [write last]
1. **Introduction** — metric-first framing; structural vs numerical stability (the central rhetorical move);
   numbered claims preview C1–C5; honesty ("less intervention," not "for free"). Claims: all.
2. **Problem setup, data, evaluation protocol** — 2D vorticity ~20k frames; splits; **metric hierarchy**
   (spectra → OOD-rate → snapshots → EMD-as-fidelity), and "no clean OOD detector exists" stated as a
   property of the problem. Claim: C4. Figs: schematic only.
3. **Multi-scale VQ tokenizer** — ViT-AE, residual VQ, shared EMA codebook, sweep config; **failed DiVeQ
   writeup**.
   - **Deterministic quantization (methods + honesty):** tokenization is nearest-neighbor argmax (`vq.py:33`),
     no stochastic/Gumbel sampling — a deliberate choice to drop a tuning variable, at a spectral cost (high-k
     tail flattening = the once-paid bias). A sampled-codebook variant (optional ablation, see Experiments S4)
     would quantify that cost.
   - **VQ-VAE size barely matters → the floor is scale-resolution-bound, not capacity-bound.** Round-trip EMD
     saturates by Medium and is *non-monotone* (Large can be worse: sc917 0.106→0.116; sc1941 Medium 0.055 best);
     the lever is **scale count** (sc341→sc1941 at fixed Small drops the floor ~5×, 0.40→0.08). The codebook is
     only **~59% utilized, 0 dead codes** → not codebook-capacity-bound either. This sharpens "spectral bias paid
     once": it is paid by the *scale resolution + deterministic argmax*, demonstrably not by enc/dec depth or
     codebook size → use a small/medium VQ-VAE (data-limited regime).
   - Figs: `vqvae_arch.png`, `multiscale_vq.png`, `codebook_analysis/*` (utilization ~59%, residual_energy), `compare_recon.png`.
4. **Next-scale prediction model** — VAR backbone, unified heads, refinement stack, attention mask.
   (Fix the stale per-scale loss-weight doc: code = `1/log(token_count+1)`, not `1/sqrt`.)
   Figs: `nsp_arch.png`, `attention_mask.png`.
5. **Rollout collapse + per-token recipe** — *the methods anchor.* The honest Test-1→8 arc with corrected
   metrics. Claims: **C1, C2.** Figs: `position_ood_overlay`, `logits_aligned_overlay`,
   `posmask_multitraj_grid`, `robust_survival_curves`, `robust_emd_traces`, `robust_multitraj_grid`.
   **Gap: P0-2 clean 3-way ablation.**
6. **Why discrete is structurally stable: spectral bias relocated** — the conserved-bias argument + the
   conditional/approximate floor (why §5's mask *enables* this). **Baseline suite B1–B4** lives here.
   Claim: **C3.** Figs: `scaling_report/spectra_*`, `paper_narrative/temp_spectra*` + B1/B2 (gap).
   **Gap: P1-6 (B1) is the biggest reviewer exposure; B3 is the gate.**
7. **Scaling laws & the data-limited regime** — param accounting, D/P≈0.54, per-tier N=128 shape,
   below-floor EMD subsection. Claim: **C5.** Figs: `scaling_tempopt_n128/{best_temperature,scaling_emd,scaling_tke}.png`.
   **Gap: P0-4 double-beat.**
8. **Sampling temperature: climate vs forecast** — U-shaped collapse (both extremes), TKE-RSE gameability,
   warm-long-run/cold-short-horizon split. *(This is where the recent inference-only sweeps land.)*
   Figs: `scaling_forecast/forecast_best_temperature.png`, `forecast_vs_longterm_emd.png`,
   `snapshots_n128_tempcollapse/medium-sc917-nsp-s22_temp_collapse.png`. **Gap: P0-3 multi-seed.**
9. **Related work** — clusters in Part G.
10. **Limitations & open questions** — sc917 no-below-floor; single system; substitution-rate=0.1 only;
    the one remaining tuned knob (T) → entropy/typical sampler as the closing move. State the honesty
    caveats here, not buried.
11. **Conclusion** — short; restate the categorical claim + what backs it.
- **Appendices** (unbounded): hyperparameters; compute/infra (Bridges2/Derecho, Slurm/PBS); full N=128
  sweep tables; extra grids; **the NextLat-derived self-predictive aux-loss future-work note**.

---

## D. Figure map (existing assets → slots) + caveats

Strongest ready-made assets, by section:
- **§5 mechanism:** `presentation/figures/position_ood_overlay.png`, `logits_aligned_overlay.png`,
  `posmask_multitraj_grid.png`, `robust_survival_curves.png`, `robust_emd_traces.png`.
- **§6 spectra:** `plots/scaling_report/spectra_*` (sc341/sc917 × size × step), `plots/paper_narrative/temp_spectra*`.
- **§7 scaling:** `plots/scaling_tempopt_n128/` (canonical).
- **§8 temperature:** `plots/scaling_forecast/`, `plots/snapshots_n128_tempcollapse/`.
- **§3 tokenizer:** `plots/codebook_analysis/` (utilization, dead codes, residual energy, geometry).
- **Schematics:** `plots/{vqvae_arch,nsp_arch,multiscale_vq,attention_mask}.png`.

**Caveats (do not reuse blindly):**
- **Greedy figures are broken** (`paper_narrative/greedy_*`, `plots/scaling_sampling`) → lead with **T=1.0**
  sampled variants. Always evaluate at T=1.0.
- **N=128 supersedes** `plots/scaling_tempopt/` and the N=4 grid → use `scaling_tempopt_n128/`.
- **Demote EMD / `collapse_rate`** to fidelity-only; **promote OOD-rate + spectra**.
- `nsp_final_loss.png` / `nsp_training_curves.png` are pre-U-shape → do not reuse.

---

## E. Related-work clusters (Part G) with the differentiation line for each

- **Next-scale generation** — VAR (`tian2024var`) + image follow-ups. *Differentiator:* zero PDE/turbulence
  applications of next-scale prediction → that intersection is ours.
- **Discrete/tokenized PDE AR** — Zebra (raster), PhysiX (raster + refinement CNN, no per-position/OOD
  discussion), Momenifar (single-scale, notes AR degradation). *Differentiator:* next-**scale** axis +
  the per-token OOD finding + 2000-step stability + scaling sweep. (Tech-report genre softens the PhysiX collision.)
- **Continuous PDE-stability control (the contrast bucket)** — PDE-Refiner (**primary foil**), Stachenfeld
  (training noise), Brandstetter MP-PDE (pushforward = our substitution-noise analog), Thermalizer.
  *Differentiator:* they **patch** the compounding bias numerically; we **avoid** it structurally.
- **Self-predictive / latent world models (future-work lineage, App. E)** — NextLat, Dreamer, DCWM, TECO,
  SPR, TWISTER. *Differentiator:* frozen two-stage tokenizer sidesteps the moving-target/representation-collapse
  that plagues single-stage Dreamer-family models.
- **Scaling laws** — Chinchilla, Henighan, Muennighoff (data-constrained), Practical-Scaling-on-PDE.
  *Differentiator:* a quantitative D/P constant for an **AR-token loss on a real PDE dataset**.

---

## F. Submission-readiness checklist (gaps mapped to claims)

**Blocking for the claims as currently phrased (close, or reduce the claim):**
- [ ] **P0-1 (C4, cheap):** time-resolved high-k spectrum + per-position OOD-rate on one time axis; the
      falsifiable overlay (on a per-scale-mask model, high-k collapse and the OOD spike are temporally locked).
- [ ] **P0-2 (C2, cheap):** clean 3-way mask ablation {none / per-scale / per-token}, n_traj≥8, OOD overlay.
- [ ] **P0-3 (C5/§8):** multi-seed confirmation of temperature optima (current = noisy n_traj=1 on a U).
- [ ] **P0-4 (C5):** disambiguate the medium-sc341-large **double-beat** (T unlogged across two runs) —
      reproduce on the exact checkpoint **or** report only the solid EMD-beat (TMLR claim-reduction).
- [ ] **P0-5:** look at the finished r4/r6 refine-depth ablation; run the missing sc1941 single-step eval.

**High-value (the discrete-vs-continuous claim C3 currently rests on citation):**
- [ ] **P1-6:** **B1 skip-quantization continuous-latent baseline** — the clean isolation; biggest reviewer
      exposure under TMLR criterion 1.
- [ ] **P1-7:** ≥1 pixel-space continuous baseline (B2 U-Net: MSE / +noise / +pushforward).
      **B3 flat-VQ Pareto is the GATE — run FIRST** (if flat-1024 floor ≈ sc917 floor, the §6 story collapses).
- [ ] **P1-8:** add a correlation/decorrelation-time metric to `analyze_rollout.py` (PDE-Refiner-lineage reviewers expect it).

**Hygiene / correctness (do before quoting numbers):**
- [ ] **Commit the untracked analysis layer** (`plot_*.py`, `analyze_*.py`, `plots/`, `experiments/`,
      `BASELINES_SPEC.md`, this `paper/`) — currently one bad `git clean` from losing the figure pipeline.
- [ ] **Swap the stale metrics** out of `nsp_stabilization_report.tex` before reusing its narrative
      (it still leads with pre-fix survival + `collapse_rate`; Test-11 takeaway #3 is contradicted by live sc917).
- [ ] **Fix the loss-weight doc** (`CLAUDE.md` / `nsp_model.py`): code uses `1/log(token_count+1)`, not `1/sqrt`.
- [ ] Re-pull current `gust2-analysis-bridges-scaling` + close the ~6 warm-sc917 N=128 coverage-gap jobs
      before treating `scaling_tempopt_n128/` figures as final.

---

## G. Recent inference-only sweeps since the work-log (the "few more sweeps")

All AR-rollout inference on **frozen** VQ-VAE + NSP checkpoints (no training):
1. **posmask-temp diagnostics** (06-11/12) — sc341 stable cold; sc917 EMD over-flagged; sc1941 real cold
   diffusive collapse. The two-collapse-mode logit-signature exhibit. → §5/§8.
2. **scaling-tempopt N=4** (06-13) — temp-optimal scaling-law figures. *Superseded by N=128.*
3. **scaling-forecast** (06-14) — short-horizon skill; cold forecast optimum vs warm climate optimum. → §8.
4. **scaling-tempopt N=128** (06-14→20, **canonical**) — confirms N=4 (median Δbest-T=0, Δemd=0.012);
   per-tier shape. → §7.
5. **N=128 temperature×collapse snapshot grid** (06-21) — U-shaped collapse exhibit
   (`medium-sc917-s22`). → §8.

### G′. Submissions since the 06-23 summary (the delta this doc now folds in)

Still all AR-rollout inference / analysis on **frozen** checkpoints (no training). These bear on §8 and
the **less-intervention** pillar; the headline is the 06-28 *overturn* of the 06-25 "irreducible" verdict.

6. **inference-samplers sweep** (06-23→25) — single-step-*adaptive* samplers benchmarked vs the swept-T
   optimum yardstick + the VQ-VAE EMD floor on the 9 flagship cells: `data_mix` (convex mixture toward the
   per-scale data prior), `calibration`-temperature + a cheap GO/NO-GO **gate** ("Recipe 1(ii)",
   `analyze_calibration_gate.py`), and `drift_warm` (horizon-adaptive overshoot). → `plots/inference_samplers/`
   (`samplers_headline.png`, `samplers_emd_grid.png`, `.csv`). **Verdict (06-25):** 3 convergent negatives ⇒
   single-step-adaptive temperature is *irreducible*; collapse is coarse-scale-driven. → §8.
7. **static per-scale fine-heating** (06-26→28) — a **fixed** per-scale temperature *schedule* (heat the
   finest-2 scales: T≈3.0 small/medium, T≈2.0 large; `calibPS` fallback), replicated across the **full
   sc1941 arch grid** and multi-modally verified (snapshot + TKE + PDF, 15-agent wf). **OVERTURNS item 6:**
   a fixed schedule *replaces the per-config sweep* — 15/15 on-manifold vs best-T's 7-beat/7-match/1-worse.
   `target_entropy_profile.png` = the tokenizer teacher-forced per-scale code entropy (**the T2 anchor for
   Top-H / I1**). EMD-alone lies here (4 low-EMD arms are diffusive collapse on spectrum+PDF). → `plots/per_scale_temp/`. → §8 / Section B reopened.
8. **climate temperature band** (06-29) — "pick T from the climate statistics": a single **a-priori** T≈1.8,
   read off the per-scale climate entropy band with *no per-config sweep*, keeps **all** sc1941 archs
   on-manifold (on-manifold, not regret-optimal). → `plots/climate_temp_band/climate_temp_band.png`. → §8 / C5.
9. **per-scale token-distribution drift metric** (06-29→30) — `scale_distribution_drift.py` (TV + JS + signed
   ΔH per scale) + an N=128 drift sweep; refines EMD/entropy by catching the disjoint-code shift entropy is
   blind to (each rollout referenced vs its **own** `gt_indices`). Token-space **window-invariance** holds
   across all 9 configs ⇒ the climate target is computable from **training** data. → `plots/scale_drift/`. → §C4 metric.

**Net effect on the spine:** Section B (sampling) is **reopened** — the 06-25 "irreducible" verdict held only
for *single-step-adaptive* samplers; a *fixed per-scale schedule* (item 7) and a *single a-priori T* (item 8)
both stay on-manifold with no sweep, which is exactly the "less intervention, unlike PDE-Refiner" win. The
**T2** tokenizer-entropy anchor (item 7) is now in hand, so **I1 (Top-H)** is unblocked. Item 9 adds a
discrete-native drift alarm alongside the OOD-rate (C4).
