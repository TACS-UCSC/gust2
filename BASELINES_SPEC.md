# Baseline Suite Spec (2026-06-12)

Baselines for the tech report. Two claims need external comparison:

- **Pillar 1 (less-intervention stability):** continuous AR drifts/blows up without
  hand-tuned stabilization; discrete + per-token mask does not.
- **Pillar 2 (spectral bias relocated):** continuous dynamics attenuate high-k
  compoundingly over rollout; discrete dynamics cannot blur below the codebook envelope.

Matrix: representation (pixel / latent) x prediction (continuous regression / discrete
classification). Our model is the discrete-latent cell; B1 and B2 fill the other two.
B3 bounds the remaining discrete cell (raster) at the tokenizer level without training
its dynamics.

## Locked decisions

| Decision | Call |
|---|---|
| Latent baseline structure | Option B: same transformer block recipe, direct next-frame regression on full 32x32 z (no multi-scale machinery) |
| Latent baseline anchor | Single anchor: frozen medium-sc341 VQ-VAE enc/dec, quantizer bypassed. No per-config scaling curve. If flagship flips to sc917, one dynamics retrain. |
| Raster-AR discrete | NO dynamics training. Tokenizer-level comparison only (B3). Rationale in paper: raster per-position prediction is *easier* (richer conditioning, lower conditional entropy), so the tokenizer floor is the only dynamics-independent comparison; we bound the best case. Empirical dynamics test only if a reviewer demands it. |
| Conditioning history | 1 previous frame, all models (parity with NSP). k-frame variant = reviewer-response experiment only. |
| Pixel trio sizes | Both ~31M and ~57M. Flagship size decided from results + advisor. |
| Flat-VQ points | Three: 16x16, 24x24, 32x32 (sc256 / sc576 / sc1024). |
| Ensemble honesty | MSE baselines are deterministic at inference; their n_traj variation is start-frames only. Label as deterministic in figures; do not present ensembles as like-for-like with T=1.0 sampled discrete rollouts. |
| Discrete sampling | T=1.0 always (project rule; greedy is historically broken). |

## B1 — Skip-quantization continuous-latent (pillar 2 isolation)

The cleanest experiment in the suite: identical enc/dec, identical data, the only bit
flipped is quantize-or-don't.

- Frozen medium-sc341 VQ-VAE encoder/decoder; quantizer bypassed. Dynamics operate on
  continuous z: (codebook_dim, 32, 32).
- Model: transformer with the NSP block recipe (RMSNorm, QK-norm, bias-free, SwiGLU,
  2D RoPE), 1024-position sequence, linear in/out projections, one-shot z_t -> z_{t+1}
  regression, MSE loss. Parameter count matched to the flagship NSP (s18-class).
- Variants: training-time input-noise injection (Stachenfeld-style),
  sigma in {0, sigma_lo, sigma_hi} as fractions of per-channel latent std, calibrated
  from a short probe. Noise rows are droppable but cheap; they make "blows up even with
  noise" a result instead of an omission.
- Rollout: closed-loop 2000 steps, deterministic; decode through frozen decoder for
  pixel-space analysis.
- Expected: latent high-k attenuation compounding over rollout / eventual blow-up;
  discrete stays at the codebook floor. This is the pillar-2 money figure.

## B2 — Pixel-space continuous trio (pillar 1 hardening, field anchor)

- Modern U-Net (PDE-Refiner-style conventions: residual blocks, GroupNorm, attention at
  coarse levels), 1 channel in/out, 1-frame conditioning, dataset-stat normalization.
- Sizes: ~31M and ~57M (loose match to Small/Medium budgets; exact matching across
  architectures is theater — state the counts).
- Variants per size: (a) plain MSE, (b) +training noise — sweep 2 sigma values at one
  size, carry best sigma to the other, (c) +pushforward trick (2-step unroll, gradient
  detached through step 1). Max 8 runs, likely 6.
- PDE-Refiner proper: deferred to conference version; named foil in related work.
- Same consecutive-pair sampling and train/val split as the token pipeline.

## B3 — Flat-VQ tokenizers (raster bound, budget-fidelity Pareto)

- Three single-scale tokenizers via existing `train.py`: `scales=(16,)`, `(24,)`,
  `(32,)` — degenerate cases of MultiScaleVQ, config-only runs. Small arch (D=5), best
  codebook config (4096 / 512 / beta 0.25 / EMA 0.90) held fixed. EMA-VQ, NOT FSQ
  (isolates the residual-pyramid axis; FSQ would change two variables at once —
  PhysiX handled in related work).
- Names follow token-count convention: `small-flat-sc256`, `small-flat-sc576`,
  `small-flat-sc1024` under `experiments/vqvae/`.
- Eval: val-set reconstruction spectra, high-k RSE, EMD. Figure: tokens/frame vs floor
  quality, both families — flat {256, 576, 1024} vs multi-scale {341, 917, 1941}
  (small family; all three trained).
- Free side-claim: inference cost — 1024 sequential raster emissions/frame vs ~7 scale
  steps. Stated, not trained.
- GATE: if flat-1024's floor is comparable to sc917's at similar budget, the skip
  argument collapses — know this before writing the section. Hence B3 runs FIRST.

## B4 — No-training controls

- **Persistence** (repeat start frame): calibrates correlation-time and EMD scales.
- **In-mask random sampling** (uniform over each position's support set, decoded
  through flagship VQ-VAE): shows the mask alone does not produce good rollouts —
  defuses "the contribution is a lookup table of training data."
- **VQ reconstruction floor** (exists): completes the bracket
  random-in-mask <= model <= floor.

## Evaluation parity

All baselines through the same harness: same start frames, 2000 steps, n_traj >= 8
(shared with the P0 multi-seed set). Battery, in order of evidential weight:

1. Time-averaged + time-resolved spectra (primary stability evidence)
2. Per-position OOD-emission rate — discrete only; continuous models structurally
   cannot have it. State as a feature of the method, not a metrics gap.
3. Snapshots
4. EMD (fidelity only — never as THE stability metric)
5. Correlation time (new metric, all models; reviewers expect it)

## Code items

1. `unet.py` + `train_unet.py` (noise + pushforward flags) — biggest new-code item;
   JAX/Equinox from scratch.
2. `train_latent.py` — fork of train_nsp harness, scale machinery removed, MSE loss.
3. `rollout_continuous.py` — shared closed-loop driver for B1 (latent, with decode
   hook) and B2 (pixel).
4. `analyze_rollout.py` — add pixel-input path (bypass token decode) + correlation-time
   metric (also covers P1 #8).
5. `controls.py` — persistence + in-mask random sampling.
6. sbatch templates under `scripts/bridges/baselines/`.

## Layout & tracking

- Ocean: `experiments/baselines/{latent-mse-*, unet-*}/`, flat tokenizers under
  `experiments/vqvae/small-flat-sc*/`.
- Wandb: `gust2-baselines` (flat-VQ training may go to `gust2-vqvae` like other
  tokenizer runs; analysis to `gust2-analysis`).

## Execution order

1. **B3 flat-VQ** — config-only, cheapest, and gates the raster section of the paper.
2. **B4 controls** — scripting only, no GPU.
3. **B1 latent-MSE** — small code fork, one anchor + noise rows.
4. **B2 U-Net trio** — biggest implementation lift, start `unet.py` in parallel with
   B3 training.

## Parked / out of scope

- Raster-AR dynamics training (bounded by B3; reviewer-demand only)
- PDE-Refiner proper (conference version)
- k-frame conditioning variants (reviewer response)
- Diffusion baselines (ACDM — cite only)
