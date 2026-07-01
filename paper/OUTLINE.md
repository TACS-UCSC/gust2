# gust2 — TMLR paper outline v2 (method paper, sections → figures → runs)

**Supersedes** `_archive/OUTLINE_v1_claims.md` + `_archive/FIGURES_v1_storyboard.md` (kept for asset
provenance — v1's FIGURES.md maps every existing png to its producing script). Companion trackers:
`Experiments.md` (task ids B1/B3/E2/… still canonical) and `BASELINES_SPEC.md`.

**The story (locked 2026-07-01):** We present a method for autoregressive prediction of a turbulent
2D flow: quantize the field into a multi-scale discrete latent (VAR-style residual VQ), autoregress
entirely in token space, decode only at readout. Motivation: statistically stable very-long rollouts
(climate fidelity, post-Lyapunov) **without task-specific hyperparameter search** — no per-config
long-rollout temperature sweeps; the sampling calibration is a *data-read* from training-set climate
statistics. Chronos is the motivational lineage (discretize continuous dynamics, inherit the AR
toolbox); VAR is the architectural lineage.

**The narrative spine = three-layer defense-in-depth.** Discreteness converts gradual spectral drift
into *countable, detectable* failures, each layer's failure motivating the next:

1. **Vocabulary (structural):** finite codebook → smooth blur-toward-the-mean is unrepresentable.
   Not sufficient on its own —
2. **Per-token support mask (marginal):** without it → positional-OOD explosion ("static", mode A).
   Mask bounds the *marginal* support (survival 0–4% → 88–100%). Still not sufficient —
3. **Sampling calibration (joint/temporal):** masked but cold/greedy → in-support diffusive collapse
   (non-turbulent banding, mode B). Fixed by calibration read from data statistics, valid by
   window-invariance. Zero-shot ⇒ **safety** (on-manifold); once-chosen schedule ⇒ **optimality**
   (matches swept best-T), config-agnostic across the full arch grid.

## Claims (M1–M6) and their evidentiary state

| # | Claim | State | Carried by |
|---|-------|:-----:|-----------|
| **M1** | Headline method claim: multi-scale discrete latent AR + per-token mask + data-read calibration ⇒ 2000+-step statistically stable rollouts with **no task-specific tuning** (interventions are *untuned/structural*, not absent) | **[E]** modulo F6 replots | §5–§6, F1, F6.1–F6.5 |
| **M2** | Two distinct discrete failure modes: positional-OOD explosion (off-manifold, entropy↑) vs in-support diffusive collapse (entropy cliff↓ + confidence↑). Opposite signatures; both online-detectable (OOD-rate, drift ΔH) | **[E]** | §4, F4.1–F4.2 |
| **M3** | Support constraint must be **per-token** (not per-scale); mask ⊥ temperature — both independently necessary | **[P]** — needs E2 clean ablation | §5, F5.1 |
| **M4** | The safe sampling basin is **predictable a priori** from training-data climate statistics (window-invariance lemma). Zero-shot single-T / calibPS ⇒ on-manifold everywhere (safety, *not* regret-optimal — quantify regret); fine-heat schedule tuned once ⇒ 7-beat/7-match/1-worse vs swept best-T across 15 sc1941 cells (config-agnostic optimality). **Do not fuse the two rungs.** | **[E]** — replots + small gap runs | §6, F6.1–F6.7 |
| **M5** | Spectral bias is paid **once** (tokenizer), not compounded per step. Mechanism phrasing: the AR core never touches the continuous representation; decoder runs once at readout, so spectral cost is fixed w.r.t. horizon. Continuous AR (pixel or latent) drifts. | **[G]** — rests on citation until B1 (+B3 gate) | §7, F7.1 |
| **M6** | Multi-scale is the efficiency lever: recon floor drops ~5× with scale count at fixed model size; floor is scale-resolution + det-argmax bound (codebook ~59% used, 0 dead), not capacity-bound | **[P]** — scale-count evidence in hand; *structure-vs-token-budget* isolation needs B3 | §3.2, F3.2–F3.3 |

Secondary findings (demoted to §8, one distilled figure each): scaling laws (D_uniq/P≈0.54, N=128
shape), climate-vs-forecast temperature split, below-floor EMD via mode-averaging (sc341 only —
claim-reduced per E4 unless disambiguated).

**Honesty rules carried over from v1 (unchanged, still binding):**
- Metric hierarchy: spectra primary → OOD-rate/drift alarms → snapshots → EMD fidelity-only. **No
  claim figure may be adjudicated by EMD alone** (EMD is blind to mode B — four low-EMD arms were
  diffusive collapse on spectrum+PDF).
- `collapse_rate` (2×floor bar) over-flags — never quote it; judge by EMD band + snapshot + PDF.
- Greedy figures broken/do-not-cite; always sampled T.
- "Minimal intervention" is defined as **untuned** (derived once from tokenizer/data), not *fewer*:
  the mask comes from tokenizer support, substitution noise is fixed at 0.10, temperature from
  climate entropy statistics. Say "no per-model calibration; the target is a property of the data."
- Scope: one system. Claims phrased for 2D turbulence unless S1 (Rayleigh-Bénard) runs — S1 is now a
  *direct test of M4* (same data-read produces a working temperature zero-shot on a second system),
  not external-validity garnish. Decide early.

---

## Figure status legend

- **HAVE** — existing png usable as-is or with relabel/crop (source map in `_archive/FIGURES_v1_storyboard.md`).
- **REPLOT** — data already local (repo `experiments/`, `plots/*.csv`, `inference_anchors/`, `plots/_wandb_cache/`, wandb) — write/modify a plot script.
- **PULL** — data exists but lives on Bridges (`/ocean/.../experiments/{rollouts,analysis}/...`) — needs a Globus pull (or re-derive from wandb).
- **RUN** — new compute required (tagged with the `Experiments.md` id and 🟢/🟡/🔴 cost).

---

## §1 Introduction

Gap (long-rollout stability; existing fixes = numerically patched, per-system-tuned — PDE-Refiner
foil) → the bet (discretize; Chronos motivation, VAR mechanism) → three-layer arc preview → claims
list M1–M6. Metric-first framing: "stable" = climate/statistical fidelity post-Lyapunov, stated
before any result.

- **F1 — teaser montage (flagship, NEW).** One row per outcome at increasing lead time
  (t=10/100/1000/2000): GT · mode-A static explosion · mode-B diffusive banding · **a-priori-calibrated
  stable rollout**. One-line annotation each ("no mask" / "mask, cold" / "mask + data-read T").
  Replaces v1's greedy-anchored recovery-curves teaser (retired — it framed T as a swept knob).
  *Status:* compose — mode-B + stable rows from existing snapshot npz (fig8c lineage, `plot_snapshot_tempgrid.py`);
  mode-A row needs a no-mask-era or hot-arm rollout snapshot → **PULL** (check Bridges `rollouts/` for
  pre-mask sc341/sc917 artifacts) or trivially **RUN** (one unmasked rollout, minutes on 1 GPU).

## §2 Problem setup, data, evaluation protocol

2D vorticity ~20k×256², splits, rollout protocol (continue past training-data end — kills the
lookup-table objection by construction, state once). Metric hierarchy **in order of evidential
weight**, committed before results. Introduce both discrete-native alarms here: per-position
OOD-emission rate + per-scale token-distribution drift (TV/JS/signed ΔH).

- **F2.1 — metric-hierarchy composite** (spectra battery / OOD-rate trace / snapshot strip / pixel-PDF).
  *Status:* **HAVE** (v1 fig2a/fig5a/fig2/fig2b) — recompose into one 4-panel figure, order = hierarchy.
- **F2.2 (optional) — protocol schematic** (train window → rollout continuation; where metrics attach).
  *Status:* draw (no data).

## §3 Method — multi-scale discrete tokenizer + next-scale AR

### 3.1 Tokenizer (ViT-AE + residual multi-scale VQ, shared EMA codebook)
Deterministic argmax quantization stated as a *choice* (drops a tuning variable, costs high-k tail —
the once-paid bias; S4 sampled-codebook ablation optional). Failed-DiVeQ note (appendix pointer).

- **F3.1 — architecture schematic pair** (`vqvae_arch` + `multiscale_vq`). *Status:* **HAVE**.

### 3.2 Why scales: the efficiency claim (M6)
Floor vs scale count (~5×, sc341→sc1941 at fixed Small); saturation by Medium VQ, non-monotone in
size; codebook ~59% used, 0 dead → scale-resolution + det-argmax bound, not capacity.

- **F3.2 — VQ-size ↔ floor bar** (3 sizes × 3 sc-configs round-trip EMD; the non-monotonicity).
  *Status:* **REPLOT** — data in `plots/codebook_analysis/codebook_summary.csv` (~10-line script, v1 flagged it).
- **F3.3 — flat-VQ vs multi-scale recon Pareto** (matched token budget: flat-(16,)/(24,)/(32,) vs sc-configs).
  *Status:* **RUN — B3 🔴 (the GATE; cheapest of the baseline trio; run FIRST — its result decides
  whether M6 is "structure wins" or claim-reduces to "token count wins").*
- **F3.4 — codebook evidence pair** (utilization/dead-codes + per-scale residual energy). *Status:* **HAVE**.

### 3.3 NSP model (VAR backbone, unified heads, refinement stack, attention mask)
Fix the stale loss-weight doc while writing (code = `1/log(token_count+1)`, not `1/sqrt`).

- **F3.5 — NSP schematic + t0/t1 attention mask.** *Status:* **HAVE**.
- **F3.6 — training curves (NSP CE + VQ recon).** *Status:* **HAVE** (fresh 06-30 pulls). Appendix candidate.

### 3.4 Training recipe (the untuned interventions, forward-ref §5)
Per-token CE support mask + 10% substitution noise (the pushforward analog — cite honestly). No figure.

## §4 The two failure modes of discrete rollout (M2) — layer 1 is not enough

Mode A "static": positional-OOD explosion — off-manifold, OOD-rate 2%→28%, entropy↑. Mode B
"non-turbulent": in-support diffusive collapse — banding, entropy cliff↓ + confidence↑, locked to
high-k drain. Opposite signatures ⇒ must not be conflated ⇒ no single scalar metric suffices (ties
back to §2 hierarchy; sets up why EMD-only judging fails).

- **F4.1 — two-mode definition figure (NEW, distill don't reuse).** Side-by-side: snapshot of each
  mode (vs GT) + one signature trace per mode beneath (mode A: OOD-rate spike at collapse; mode B:
  fine-scale entropy cliff + top-1 confidence rise). Distilled from v1 fig5a/fig5b (which are 5–7
  panel internal diagnostics — do not paste into the paper).
  *Status:* **REPLOT/PULL** — traces derive from `analyze_position_ood.py` / `analyze_logits_aligned.py`
  outputs (verify which multitraj npz are local vs Bridges); snapshots as in F1.
- **F4.2 — temporally-locked overlay (E1 🟢).** Time-resolved high-k spectral energy + OOD-rate on one
  time axis: high-k collapse and OOD spike lock. *Status:* **REPLOT** — analysis-only from existing
  `analysis_data.npz` + `eval_per_timestep.npz` (E1 in `Experiments.md`; outcome known, figure never built).

## §5 Layer 2 — the per-token support mask (M3)

The honest arc: per-scale mask fails → per-token mask takes survival 0–4% → 88–100% — *and* the
cold arm still collapses (mode B), so mask ⊥ temperature: both independently necessary. This is
where "free stability isn't quite free" is stated in print.

**Failure-inclusion policy (locked 2026-07-01):** failures appear only as (i) one exemplar per
mode (F4.1/F1), (ii) E2's controlled arms, (iii) optionally one compact appendix outcome table
(cells × conditions → mode-A/mode-B/stable). NO per-arch failure grids; NO chronological
intervention trail in the main text. Abandoned intermediates (mask-only, null-token substitution,
partial refine-head sweeps) get ≤1 "development notes" appendix paragraph or nothing — history
motivates, E2's controls prove. (Refinement stack ≠ failure: it's final architecture, §3.3.)
Cold-collapse severity grows with token count: sc341 stable cold (the flat-basin control),
sc917 milder banding (quote snapshot/spectrum-judged results, not the retracted EMD flags),
sc1941 unambiguous — write it as the gradient, not "everything collapsed cold."

- **F5.1 — clean 3-way mask ablation (flagship for M3).** {no-mask / per-scale / per-token} ×
  {cold-greedy / warm}, one sc917 cell, n_traj≥8, OOD-rate overlaid. The greedy/cold arm folds in
  old E6 (mask alone under cold still diffuses).
  *Status:* **RUN — E2 🟡/🔴** (2 extra NSP trainings for the mask arms + rollout fan; the *one*
  flagship training run this paper still needs).
- **F5.2 — survival curves + per-traj EMD traces (supporting).** *Status:* **HAVE** (v1 fig5c/5d/5e).
- **F5.3 — per-token-mask multitraj snapshot grid (appendix).** *Status:* **HAVE** (v1 fig5f).

## §6 Layer 3 — sampling calibration as a data-read (M4) — **the centerpiece**

Structure: (a) the basin exists and sharpens with token count; (b) its location is predictable
zero-shot from data statistics (safety); (c) a once-chosen per-scale schedule reaches the swept
optimum (config-agnostic optimality); (d) single-step-adaptive samplers fail (negative control);
(e) cost accounting. State the two rungs separately; quantify regret on rung 1.

- **F6.1 — temperature phase diagram (flagship, NEW).** Per token count: T on x, cells stacked;
  each (cell, T) classified **tri-state** {mode A / mode B / on-manifold} adjudicated by OOD-rate +
  drift-ΔH sign + spectra (NOT EMD-alone); a-priori T* and calibPS marked as vertical lines landing
  inside the safe band; band visibly narrows sc341→sc1941.
  *Status:* **REPLOT + PULL + small RUN.** Verdict inputs: N=128 sweep artifacts (Bridges) + drift
  metric (`scale_distribution_drift.py`, needs each cell's `rollout_tokens.npz` + `gt_indices` —
  own-compact-map gotcha) . Gap runs 🟡 (inference-only): sc341 hot edge T∈{1.8…3.0} (never swept)
  + the ~6 warm-sc917 coverage cells. Without the hot edge the "bracketed on both sides" reading
  fails for sc341.
- **F6.2 — prediction-vs-outcome scatter (the M4 money plot, NEW).** Predicted T (climate entropy
  band / calibPS, from `inference_anchors/*.json`) vs swept best-T (from
  `plots/scaling_tempopt_n128/scaling_tempopt.csv`), one point per cell, 45° line, on-manifold
  interval as vertical extent per cell. **Both inputs already local → REPLOT (cheap, do first).**
- **F6.3 — window-invariance lemma exhibit (NEW, small).** Train-vs-val per-scale token-distribution
  excess JS (≤1.4e-2, all 9 configs) — the lemma that makes "target from *training* data" legitimate.
  *Status:* **REPLOT** (drift-sweep outputs; verify local vs Bridges).
- **F6.4 — schedule causal chain (recast of v1 fig8e/8f).** Three panels: tokenizer per-scale data
  entropy profile (the anchor) → derived fine-heat schedule (heat finest-2) → outcome vs swept
  best-T with the 7/7/1 tally. Replaces the six-arm spaghetti replication grid (that goes to appendix).
  Provenance honesty in caption: the 3.0/2.0 constants chosen once from {2,3,4} on one cell, then
  transferred grid-wide unmodified — "config-agnostic", not "zero-shot".
  *Status:* **REPLOT/PULL** (per-scale-temp sweep arms; anchor profile data local via `inference_anchors/` + tokens npz).
- **F6.5 — regret quantification (NEW, small).** Distribution of EMD/spectral regret of rung-1
  (zero-shot T) vs per-cell swept best-T across all cells; median annotated. Kills the "so is the
  zero-shot T actually good?" question before a reviewer asks.
  *Status:* **REPLOT** (`scaling_tempopt.csv` + climate-band data, local).
- **F6.6 — adaptive samplers negative control (FIX).** data_mix / calibration+gate / drift_warm vs
  the swept-T yardstick — v1's `samplers_headline.png` is broken (legend promises the yardstick
  dashed line, not plotted; one dot per config). Rebuild from `plots/inference_samplers/inference_samplers.csv` (local).
  *Status:* **REPLOT**. Optional add: I3 entropy-*minimizing* samplers fail into mode B 🟢 (high
  rhetorical value: confidence-maximizing = the wrong direction).
- **F6.7 — cost table (not a figure).** Sweep cost (T-grid × 128 traj × 2000 steps × cells, H100-hours
  from Slurm accounting) vs data-read cost (one pass over tokenized train set + one teacher-forced
  eval). The claim M1 in one row. *Status:* bookkeeping (sacct/job logs on Bridges).
- **F6.8 (optional) — I1 Top-H 🟡.** Entropy-bounded truncation anchored to the tokenizer profile —
  the rung-3 "zero-shot *and* optimal" closing move. Paper stands without it; if unrun, name the gap
  explicitly in §10. *Status:* **RUN (optional, inference-only; T2 anchor in hand).*

## §7 Why discreteness: spectral bias paid once (M5) — the gated pillar

Mechanism stated carefully: pixel-AR reapplies a spectrally-biased operator per step (compounds);
continuous-latent AR accumulates regression-to-the-mean in-latent; discrete-latent AR fixes spectral
cost at tokenization (decoder once, at readout). PDE-Refiner = primary foil (they patch numerically,
tuned; we remove structurally, untuned).

- **F7.1 — high-k energy vs rollout step, discrete vs continuous (flagship for M5).** Discrete: flat
  at tokenizer floor; continuous-latent (B1): decays; (+FNO pixel if B2 runs).
  *Status:* **RUN — B1 🔴** (param-matched continuous z_t→z_{t+1} on frozen enc/dec, σ∈{0,lo,hi}
  noise rows) + **T3 🟡** (continuous closed-loop driver — shared infra, build once for B1/B2).
- **F7.2 — pixel-space baseline (B2 FNO, +noise/+pushforward rows).** *Status:* **RUN — B2 🔴,
  DECIDE scope** (cross-framework parity cost; claim survives with B1+B3 only, but B2 is the
  PDE-community anchor reviewers recognize).
- **F7.3 — supporting rollout-spectra grids (discrete stays on GT spectrum 2000 steps).**
  *Status:* **HAVE** (`scaling_report/spectra_*`, `paper_narrative/temp_spectra_*`).
- *(B3 result surfaces in §3.2/F3.3 but is argued here too — if flat-VQ floor ≈ sc917 floor, M6
  claim-reduces and §7 leans entirely on B1.)*

## §8 Secondary findings (one distilled figure each; full grids → appendix)

- **F8.1 — scaling laws.** D_uniq/P≈0.54 (~20× tighter than text) + N=128 EMD-vs-params per tier.
  *Status:* **HAVE/REPLOT** (`scaling_tempopt_n128/`). Double-beat: report solid EMD-beat only
  (E4 claim-reduction default; reproduce only if cheap).
- **F8.2 — climate-vs-forecast temperature split.** Distill v1 fig8a/8b: forecast optimum cold
  (0.8–1.0, pinned at bracket edge — state as lower bound) vs climate warm (1.6–2.2); drop or
  aggregate the unreadable sc341 row. *Status:* **REPLOT** (forecast sweep data).
- **F8.3 — below-floor EMD (sc341 mode-averaging).** Time-varying AR averages out quantization
  mode-sharpening; sc917 does *not* show it (say so). *Status:* **REPLOT** from existing summaries.

## §9 Related work

Clusters + differentiator lines (v1 Part E largely reusable): Chronos (motivation: discretize +
AR toolbox — but fixed scalar binning, no scales, no learned VQ) · VAR lineage (mechanism; zero PDE
applications = our intersection) · discrete PDE AR (Zebra, PhysiX, Momenifar — differentiators:
scale axis, per-token mask, OOD alarm, 2000-step climate stability, sweep-free calibration) ·
continuous stability patches (PDE-Refiner primary foil; Stachenfeld; MP-PDE pushforward ≈ our
substitution noise — cite the analogy ourselves) · scaling laws (Chinchilla/Henighan/Muennighoff;
our D/P on a real PDE token loss) · latent world models (future-work lineage, appendix).

## §10 Limitations & open questions

Single system (unless S1) · transfer axis honesty: demonstrated across model/tokenizer configs
within one system, not across PDEs · zero-shot rung is safety-not-optimality (regret quantified);
zero-shot+optimal = open (I1) · substitution rate 0.10 only (S2) · deterministic-quant spectral
cost un-ablated (S4) · sc917 below-floor absence unexplained · forecast optimum = lower bound
(bracket edge).

## §11 Conclusion (short: the three-layer claim + what backs each layer)

## Appendices (unbounded)

Full N=128 tables + per-cell grids · per-scale-temp six-arm replication grid (from §6.4) ·
sc341/sc917 report grids · 27-cell T=1 snapshots · hyperparams · compute/infra (Bridges2/Derecho,
Slurm/PBS) · DiVeQ failure note · NextLat aux-loss future-work note · reproducibility (wandb pull
scripts; `figures/pull_from_wandb.sh`).

---

## Consolidated run list (priority order)

| # | What | Cost | Feeds | Notes |
|---|------|:----:|-------|-------|
| R1 | **B3 flat-VQ gate** — single-scale EMA-VQ at (16,)/(24,)/(32,), best codebook cfg, recon eval | 🔴 (cheapest trio member) | F3.3, M6, §7 framing | **FIRST — result reshapes M6/§7 before writing** |
| R2 | **E2 3-way mask ablation** — train no-mask + per-scale-mask NSP on one sc917 cell; rollouts × {cold, warm}, n_traj≥8, OOD overlay | 🟡/🔴 | F5.1 (M3 flagship) | the one remaining flagship training |
| R3 | **Basin gap rollouts** — sc341 hot edge T∈{1.8…3.0}; ~6 warm-sc917 N=128 coverage cells | 🟡 inference-only | F6.1 | without hot edge, sc341's basin has one wall |
| R4 | **B1 continuous-latent** (+ T3 driver first) — z_t→z_{t+1} MSE, param-matched, σ noise rows | 🔴 | F7.1 (M5 flagship) | biggest reviewer exposure |
| R5 | **B2 FNO pixel** (+noise/+pushforward) | 🔴 | F7.2 | DECIDE — scope or skip; parity cost |
| R6 | **Mode-A snapshot source** — one unmasked (or hot) rollout if no pre-mask artifact found on Bridges | 🟢 | F1, F4.1 | minutes on 1 GPU |
| R7 | I1 Top-H (optional closing move) | 🟡 | F6.8 | inference-only; T2 anchor in hand |
| R8 | I3 negative-control samplers (optional) | 🟢 | F6.6 add-on | rhetorical value |
| R9 | S1 Rayleigh-Bénard (stretch → now a *direct M4 test*) | 🔴 | M4 external validity | decide early; claims scoped to 2D turbulence if skipped |
| — | Analysis-only: E1 overlay (F4.2), T1 decorrelation-time metric in `analyze_rollout.py`, E5 free datapoints, F3.2 bar, F6.2/6.3/6.5 replots | 🟢 | — | no GPU |

**Hygiene (before quoting numbers):** commit the untracked analysis layer (one bad `git clean` from
losing the figure pipeline) · fix loss-weight doc (`1/log(n+1)`) · re-pull `gust2-analysis-bridges-scaling`
before finalizing N=128 figures.

## Data to locate on Bridges (blocking the PULL-tagged figures)

Sweep output roots resolved from `scripts/bridges/*.sh` (2026-07-01; `experiments/rollouts/` holds
only the base 27-cell T=1 rollouts — all sweeps write to sibling dirs under
`/ocean/projects/mth260004p/sambamur/experiments/`):

| Root | Contents | Feeds |
|------|----------|-------|
| `scaling-tempopt-n128/<cell>/T<t>/{rollout,analysis}` + `<cell>/survival/` | canonical N=128 sweep (drift sweep reuses this base) | F6.1, F6.3, F6.5 |
| `per-scale-temp/<cell>/` | fine-heat / coarse-heat / calib arms | F6.4 |
| `inference-samplers/<cell>/` | data_mix / calibration / drift_warm arms | F6.6 |
| `calibration-gate/<cell>/` | gate outputs | F6.6 |
| `ar-robust-scaling/` | the 45-cell NSP checkpoints (inputs for R3 gap rollouts) | R3 |

**Coverage confirmed via full inventory (2026-07-01 `tree -d -L 2` + per-cell ls):**

- `scaling-tempopt-n128/` — all 45 cells present. Per-family T grids: sc341 {0.8–1.2},
  sc917 {1.2–2.0}, sc1941 {1.4–2.2}. **Gaps for R3:** (i) sc341 hot edge — N=128 stops at 1.2
  (old N=1 `rollouts-temp-sweep` reaches only 1.6 on medium-s18×3 seeds, no wall found yet);
  (ii) just **2** missing warm cells: `medium-sc917-s50/T2p0`, `large-sc917-s50/T2p0` (the "~6-job
  gap" is otherwise closed); `survival/` diagnostic dirs missing on ~12 cells (non-blocking).
- `per-scale-temp/` — full 15-cell sc1941 grid ⇒ **F6.4 unblocked (PULL)**.
- `inference-samplers/` — 9 flagship cells ⇒ **F6.6 unblocked** (headline csv already local).
- `scaling-forecast/` — full 45 cells ⇒ **F8.2 unblocked (PULL)**.
- `diagnostics/posmask-temp/` — the mode-B logit-signature diagnostics ⇒ **F4.1 traces (PULL)**.
- `rollouts-temp-sweep/` — old N=1 hot arms to **T=3.0** on large-sc917-s34 + large-sc1941-s73 ⇒
  hot-wall exhibit usable at N=1 (caveat in caption) even before R3.
- **Mode-A source resolved:** per-token mask + substitution landed 2026-04-28 (`000b173`);
  `ar-robust-scaling` sweep started 04-30 ⇒ the plain `ar/` 27-cell grid is the **per-scale-mask
  era**. Its *sampled* rollouts (`rollouts-sampling/` T0.7/1.0/1.2 arms — prefer these over the
  greedy-era base `rollouts/`, per the T=1.0 rule) contain mode-A explosions ⇒ F1/F4.1 static row
  is a **PULL**; R6 demoted to contingency. (Belt-and-braces: mtime-check one `ar/` checkpoint
  predates 04-28.)
- **E2 scope option:** the base `ar/` checkpoints *are* a per-scale-mask arm — but without
  substitution noise, so reusing one conflates mask granularity with noise. Keep E2 = 2 fresh
  trainings (no-mask, per-scale, both **with** noise) for the clean isolation; the reuse shortcut
  (1 training) is the compute-pressed fallback with a caption caveat.

Still open: (a) whether the drift-sweep npz outputs live inside `scaling-tempopt-n128/<cell>/` or
only in wandb (drift job logs sit in its `logs/`); (b) `sacct` GPU-hours for the cost table (F6.7).
