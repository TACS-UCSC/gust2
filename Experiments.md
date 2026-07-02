# Experiments — master tracker (TMLR paper)

Checkbox log for every experiment the paper needs. Companion to `BASELINES_SPEC.md` (full baseline
specs) and `paper/OUTLINE.md` (claim spine). Keep this the single source of truth for status.

**Legend:** `[ ]` todo · `[~]` partial/data-exists-but-not-clean · `[x]` done.
**Cost:** 🟢 no/low GPU · 🟡 moderate · 🔴 large (training / big sweep).
**Claim tags** (numbering-agnostic; map to `paper/OUTLINE.md`):
`metric` (spectra-primary / EMD-insufficient) · `two-modes` (OOD vs diffusive, opposite sampling response) ·
`mask` (per-token support mask = headline fix) · `floor` (spectral bias relocated; conditional codebook floor) ·
`disc-vs-cont` (discrete beats continuous — the gated pillar) · `scaling` (D/P≈0.54).

---

## Claim → experiment coverage (the "are we done?" view)

| Claim | Covered by | State |
|-------|-----------|-------|
| metric | E1 figure (analysis) + existing battery | outcome ~known; just plot it |
| two-modes | existing logit/OOD diagnostics + **E2** greedy arm | E2's greedy arm makes it airtight |
| mask (headline) | **E2** (one sc917 model) | scattered evidence → **E2 to run** |
| floor (mechanism) | E1 + existing spectra + **A-VQ ✅** | scale-resolution+det-quant bound (codebook ~59% used); VQ size saturates by Medium |
| disc-vs-cont (gated) | **B1, B2** (+ **B3 gate**) | **none run — biggest exposure** |
| scaling | N=128 sweep + **E3 ✅** | done (optional: close warm-sc917 gap) |
| less-intervention | **I1** Top-H (T2 anchor ✅); **fixed per-scale schedule + a-priori T≈1.8 ✅** (06-26→29); yardstick = E3 ✅ | evidence in hand; Top-H itself still to run |

---

## A. Baselines  (your list + the controls it's missing)

- [~] **B3 — Flat / raster-style tokenizer (no scale reconstruction)** 🔴(config-only, cheapest of the trio)
  *Tag: disc-vs-cont gate.* Single-scale EMA-VQ at scales (16,)/(24,)/(32,) = degenerate `MultiScaleVQ`
  via existing `train.py`; codebook hyperparams **family-matched** (beta 0.1 / EMA 0.85 — see
  BASELINES_SPEC B3, decision 2026-07-01). **GATE — run FIRST:** if flat-1024's recon floor ≈
  sc917's, the residual-pyramid / skip-quant argument collapses, so we must know before writing §6.
  Eval: val recon spectra, high-k RSE, EMD. *(= BASELINES_SPEC B3.)*
  **Launcher ready 2026-07-01:** `scripts/bridges/baselines/train_flat_vq.sh` (3 × 1-GPU jobs).
  After training: tokenize + recon-floor eval still to wire.

- [ ] **B1 — Non-quantized continuous *latent* (vanilla ViT, t0→t1)** 🔴
  *Tag: disc-vs-cont (the clean isolation).* Reuse the frozen VQ-VAE enc/dec, bypass the quantizer, regress
  continuous z_t→z_{t+1} (MSE), param-matched to flagship NSP. **The surgically clean discrete-axis isolation
  — the single most load-bearing baseline.** Add droppable training-noise rows (σ∈{0,lo,hi}). *(= BASELINES_SPEC B1, Option B.)*

- [~] **B2 — Non-quantized continuous *pixel* (next-ViT)** 🔴 — **DECIDED 2026-07-01: ViT-AE replaces FNO/U-Net.
  Tooling ready; awaiting Bridges.**
  *Tag: disc-vs-cont (architecture-controlled cell).* The tokenizer backbone (vit_ae enc+dec) end-to-end,
  quantization stripped, MSE regression t0→t1 (`train_next_vit.py`). Kills the ⚠️ cross-framework parity cost
  outright and upgrades the matrix: same blocks/stem/latent-grid as the VQ-VAE, so quantize-or-don't is the only
  changed variable (30.9M / 57.1M = exact Small/Medium parity). **+noise (σ∈{0.01,0.1}×pixel-std) / +pushforward
  variants kept** (Stachenfeld armor). Trade-off owned: no PDE-community U-Net anchor — PDE-Refiner stays the
  related-work foil, U-Net/FNO demoted to reviewer-response. 5 arms; launcher
  `scripts/bridges/baselines/train_next_vit.sh` chains train → rollout (8 ICs × 2000 steps, f32, deterministic,
  IC-spread ensemble per spec) → analysis (GT/one-step/rollout spectra + band traces = F7.1 raw material).
  CPU smoke-tested end-to-end (all 3 variants + resume + rollout + analysis) 2026-07-01. *(= BASELINES_SPEC B2, rewritten.)*

- [~] **B4 — No-training controls (mostly resolved — demoted)** 🟢
  *Tag: metric.* The "just a lookup table" concern is **moot by protocol**: rollouts continue forward from where
  the training data stops (held-out future continuation), so the model forecasts unseen states, not retrieves
  training frames — state this once in the methods/eval section. In-mask random sampling was already tried
  informally and **collapses** (so `random-in-mask ≤ AR` is established). Remaining cheap-if-wanted: **persistence**
  (correlation-time calibration) + the existing **VQ recon floor** to complete the bracket. Low priority. *(= BASELINES_SPEC B4.)*

## B. Inference / sampling

> **Status update (06-23→30): Section B REOPENED.** The inference-only sweeps since the summary changed the
> picture. (i) **Single-step-adaptive** samplers (`data_mix` / `calibration`+gate / `drift_warm`) all *fail*
> to beat the swept-T yardstick → 06-25 "temperature is irreducible" verdict. (ii) **OVERTURNED 06-28:** a
> **static per-scale fine-heat schedule** (finest-2 scales T≈3.0 small/medium, T≈2.0 large) is replicated
> across the full sc1941 grid and stays 15/15 on-manifold *with no per-config sweep* → `plots/per_scale_temp/`.
> (iii) **a-priori single T≈1.8** from the climate entropy band keeps all sc1941 archs on-manifold →
> `plots/climate_temp_band/`. Both (ii) and (iii) are the "less intervention, no sweep" win. Figures live in
> `plots/{inference_samplers,per_scale_temp,climate_temp_band}/`. EMD-alone over-flags here — judge on snapshot+TKE+PDF.

- [~] **I0 — Inference-sampler sweep + fixed per-scale schedule (06-23→30)** 🟡 *(landed; analysis figures done)*
  *Tag: less-intervention.* See the status banner above. Recipe to write up: fine-heat the finest-2 scales
  (per-scale static T) with `calibPS` fallback; or the single a-priori T≈1.8. Yardstick = E3 (N=128 best-T).
  **Keep separate** in the writeup: climate-sweep vs a-priori vs per-scale are *distinct* claims — don't fuse.

- [ ] **I1 — Top-H bounded-entropy truncation** 🟡
  *Tag: less-intervention.* Per-position set-selector over support-masked logits; **per-scale entropy bound
  ANCHORED to the frozen tokenizer's teacher-forced per-scale code entropy** (needs **T2**). Goal: ideally match the
  swept-T spectra/EMD optimum across sc341/sc917/sc1941 **with no per-config sweep** (the "unlike PDE-Refiner"
  win), but avoiding collapse into either out of distribution or diffusive manifolds is sufficient; the **yardstick (swept-T optima) is already in hand from the N=128 sweep (E3 ✅)**, so I1 has something to
  match without new training. **Success:** anchored Top-H ≈ best-T optimum, no tuning. **Honest fallback:** if a single anchored target
  doesn't transfer, a per-scale entropy *schedule* read from the tokenizer is still the win. Validate on
  spectra/EMD/OOD, NOT on entropy. — [Top-H Decoding (arXiv:2509.02510)](https://arxiv.org/abs/2509.02510).

- [ ] **I2 — Locally Typical sampling (cheap baseline for I1)** 🟢
  Stateless per-position drop-in (arXiv:2202.00666). Brackets how much active entropy-floor injection (I1)
  buys over passively refusing the over-confident mode.

- [ ] **I3 — Negative-control samplers (exhibit)** 🟢 *(optional, cheap, high rhetorical value)*
  Run 1–2 confidence-maximizing / entropy-shrinking samplers (e.g. inverted-sign EDT, min-p, or EM-INF as
  published) and show they how the affect collapse. We want to see if entropy↓ samplers fail into the diffusive mode. *(See `project_adaptive_sampling_search` memory.)*

## C. Core-claim evidence & cleanup  (cheap, paper-blocking — mostly MISSING FROM YOUR LIST)

- [ ] **E1 — Time-resolved spectra + per-position OOD-rate overlay** 🟢 *(DATA ANALYSIS, not an experiment — from existing .npz)*
  *Tag: metric / floor.* The temporally-locked exhibit (high-k energy collapse ↔ OOD-rate spike; per-token mask
  suppresses both). **Outcome largely known** already from the TKE-RSE-vs-time-horizon scaling — this is producing
  the figure, not discovering the result. *(audit P0-1.)*

- [~] **E2 — Clean mask ablation on ONE sc917 model** 🟡 *(the key experiment to add)*
  *Tag: mask (headline) + two-modes.* 3-way support mask {none / per-scale / per-token} × sampling {warm-optimal,
  greedy/cold}, single fixed sc917 cell, n_traj≥8, OOD-rate trace overlaid. **The greedy/cold arm folds in the
  old E6:** the per-token mask ALONE under greedy still diffusive-collapses → mask⊥temperature, both independently
  necessary. Replaces the scattered heterogeneous April runs the headline rests on. *(audit P0-2 + corrected two-modes claim.)*
  **Tooling ready 2026-07-01:** cell = **small-sc917-s34** (D/P anchor); `--loss_mask` (train_nsp.py) +
  `--emission_mask` (rollout_nsp.py) landed, CPU-smoke-tested (no-mask arm provably unmasked: CE starts at
  ln V at every scale; 6.3% off-support emissions from an untrained model; `auto` = old behavior, 0 violations).
  Launcher: `scripts/bridges/sweep_mask_ablation.sh` (2 trainings + 6 afterok rollout/analysis jobs,
  arms × T∈{0.7, 1.6}, N=16, 2000 steps; per-token arm reuses the ar-robust-scaling checkpoint).
  OOD-rate traces = offline replot from rollout_tokens.npz (token-space, no GPU).

- [x] **E3 — Multi-seed swept-temperature Pareto** ✅ DONE via the canonical **N=128** scaling-tempopt sweep
  (n_traj=128/cell → per-cell best-T optima in hand). *Tag: two-modes / less-intervention; this is I1's yardstick.*
  Optional cleanup before final figures: close the ~6-job warm-sc917 coverage gap; sc341-hot (>1.6) still un-swept. *(audit P0-3.)*
  **Gap-run tooling ready 2026-07-01 (R3):** `sweep_scaling_tempopt_n128.sh --temps "..." --no-survival`
  (per-temp skip logic makes re-runs compute only the new temps). Plan: sc341 all 15 cells ×
  T{1.4 1.8 2.2 2.6 3.0} (hot wall + the 1.2→1.8 hole); medium/large-sc917-s50 × T2.0.

- [x] **E6 — Mask-alone-under-greedy** → **folded into E2** (the greedy/cold arm of the sc917 mask ablation).

- [ ] **E4 — Disambiguate the medium-sc341-large double-beat** 🟡 *(or reduce the claim)*
  *Tag: scaling.* Two same-named Derecho runs disagree (T unlogged); Bridges sibling = EMD-beat 17/18 but
  TKE-beat 0/18. Reproduce on the exact checkpoint, **or** report only the solid EMD-beat and drop the TKE-beat
  (TMLR claim-reduction). *(audit P0-4.)*

- [ ] **E5 — Free data points** 🟢
  *Tag: scaling.* Look at the finished-but-never-examined r4/r6 refine-depth ablation; run the missing sc1941
  single-step eval (hole in the 3×3×5 grid). *(audit P0-5.)*

## D. Analysis tooling (prereqs)

- [x] **A-VQ — VQ-VAE size & deterministic-quant review** ✅ *(analysis, done)*
  *Tag: floor / disc-vs-cont.* Round-trip floor (EMD): sc341 0.40/0.23/0.22, sc917 0.106/0.114/0.116, sc1941
  0.082/0.055/0.057 (Small/Med/Large). **Bigger VQ barely helps** — saturates by Medium, non-monotone (Large can
  be worse); **scale count** is the lever (~5× across sc-configs). Quantization is argmax NN (`vq.py:33`); codebook
  only **~59% utilized, 0 dead codes** → floor is **scale-resolution + det-argmax bound, not capacity-bound**. → use
  small/medium VQ. *(Caveat: per-VQ-size checkpoints mostly on Bridges; floors read from saved CSV/PNG. See memory `project_vqvae_size_detquant`.)*

- [ ] **T1 — Correlation / decorrelation-time metric** 🟢 in `analyze_rollout.py` (PDE-Refiner-lineage reviewers expect it). *(audit P1-8.)*
- [x] **T2 — Tokenizer teacher-forced per-scale code entropy** ✅ — `plots/per_scale_temp/target_entropy_profile.png`
  (from the 06-26→28 per-scale-temp sweep). Unblocks I1's anchored bound.
- [~] **T3 — Continuous closed-loop rollout driver + pixel-input analysis path** 🟡 — **pixel half DONE 2026-07-01:**
  `rollout_continuous.py` (deterministic f32 closed loop, IC-spread ensemble) + `analyze_continuous.py` (imports
  analyze_rollout's exact spectral machinery; adds one-step column + per-step band-energy traces). Remaining: B1
  latent mode (decode hook) — slots into `rollout_continuous.py` with `train_latent.py`.
- [x] **T4 — Per-scale token-distribution drift metric** ✅ *(analysis, 06-29→30)* — `scale_distribution_drift.py`
  (TV + JS + signed ΔH per scale) + N=128 drift sweep → `plots/scale_drift/`. Catches disjoint-code shift that
  entropy/EMD miss; token-space window-invariance ⇒ climate target computable from training data. Refines C4.

## E. Optional / stretch

- [ ] **S1 — Second dataset (Dedalus Rayleigh-Bénard, from gust v1)** 🔴 — the biggest external-validity lever; de-risks single-system scope. Stretch, not a TMLR blocker.
- [ ] **S2 — Substitution-rate ablation** 🟡 — only 0.10 ever tried; strengthens the recipe. *(audit P2.)*
- [ ] **S3 — k-frame conditioning** — PARKED (reviewer-response only; would beat 1-frame on all metrics, avoided for cost).
- [ ] **S4 — Stochastic / sampled-codebook quantization ablation** 🟡 — quantify the deterministic-argmax spectral cost (does sampling the codebook posterior recover high-k?). NOTE: adds a tuning variable we deliberately avoided → report as a cost-quantifying ablation, not a new default. *(from A-VQ review.)*
- [ ] **S5 — Raise codebook K (4096→8192) at fixed scales** 🟡 — test whether codebook capacity limits the floor (utilization ~59% says headroom; currently inferred, untested). *(from A-VQ review.)*
- [ ] **S6 — Tabulate per-VQ-size high-k spectral tail** 🟢 *(analysis)* — does Large close the high-k gap, or is the tail flat across sizes? (only small-sc917 digitized so far). *(from A-VQ review.)*

---

## Execution order & dependencies

1. **B3 first** — it's the gate; a bad result reshapes §6 before we write it.
2. **Cheap, parallel, now:** E1 (analysis), E5, T1, T2, I3 (all 🟢, no training). B4 mostly resolved.
3. **E2** — the one flagship sc917 mask×sampling ablation, n_traj≥8 (E3 already done, E6 folded in).
4. **I1/I2** after T2 (tokenizer-entropy anchor); yardstick (E3) already in hand.
5. ~~B1 → B2 after T3~~ **B2 tooling done first (next-ViT decision killed its implementation lift); B1 next —
   reuses T3's driver + analyzer.**
6. **E4** anytime (disambiguation/claim-reduction). **S1/S2** stretch.

**Self-contained vs gated:** disc-vs-cont (B1/B2/B3) is the only pillar with *zero* current evidence — prioritize.
Everything in §C is cheap and makes claims we already believe *defensible*; don't let the expensive baselines
crowd them out.
