#!/bin/bash
# drift_warm gain sweep — the horizon-adaptive temperature OVERSHOOT sampler.
#
# Section B's positive-result attempt after the calibration gate's clean NO-GO
# (no teacher-forced SINGLE-STEP statistic reaches the rollout warmth: the model
# is single-step calibrated, so every anchor solves to T<=1). drift_warm is the
# horizon complement: during free-running rollout it measures the LIVE per-scale
# entropy deficit vs the teacher-forced baseline H_ref and OVERSHOOTS-warms by
#     target = H_ref + (drift_gain - 1) * relu(H_ref - mean H_live).
# drift_gain == 1 reduces to the etmodel arm (restore to H_ref -> T~=1, the known
# under-warming control); drift_gain > 1 overshoots above H_ref (the only way to
# break the solved temperature above 1).
#
# This sweeps drift_gain on the FLAGSHIP sc1941 cell (the worst etmodel collapse,
# 73x VQ floor). H_ref is the model teacher-forced anchor (measure_model_entropy.py),
# built per-gain-job (gain-independent; rebuilt locally to avoid cross-job races).
# Same rollout/analysis pipeline as sweep_inference_samplers.sh (N=128, 2000 steps,
# posmask ON, --ic_chunk). Judge by EMD / PDF / per-position OOD / spectra — NEVER
# TKE RSE. Yardstick (small-sc1941-s139): swept best_T=2.0 EMD 0.196 (= 2.40x the
# VQ floor 0.0816); etmodel collapsed at ~73x floor.
#
# Phase 1 (default): gain sweep on small-sc1941-s139.
# Phase 2 (--cold-safety): the chosen gain(s) on small-sc341-s24 — MUST stay near
#   floor (etmodel was 0.64x); over-diffusing sc341 kills the parameter-free claim.
#
# Usage:
#   ./scripts/bridges/sweep_drift_warm.sh                       # sc1941 flagship, gains 1.0..4.0
#   ./scripts/bridges/sweep_drift_warm.sh --gains "2.0 2.5 3.0"
#   ./scripts/bridges/sweep_drift_warm.sh --cold-safety --gains "2.5 4.0"   # sc341 gate
#   ./scripts/bridges/sweep_drift_warm.sh --size medium --vqvae sc917
#   ./scripts/bridges/sweep_drift_warm.sh --dry-run
#   ./scripts/bridges/sweep_drift_warm.sh --list

set -euo pipefail

# ---------- Paths (match sweep_inference_samplers.sh) ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
TOKENS_BASE="${OCEAN}/experiments/tokens"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
DRIFT_BASE="${OCEAN}/experiments/drift-warm"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"
WANDB_PROJECT_PREFIX="gust2-inference-samplers"

# ---------- Rollout / analysis config ----------
N_STEPS=2000
START_FRAME=0
N_TRAJ=128
SEED=42
BATCH_SIZE=64
DRIFT_TMAX=3.0
DRIFT_FLOOR=1.0
GAINS="1.0 1.5 2.0 2.5 3.0 4.0"

ic_chunk_for() {
    case "$1" in
        sc341)  echo "0"  ;;
        sc917)  echo "64" ;;
        sc1941) echo "32" ;;
        *)      echo "32" ;;
    esac
}
walltime_for() {
    case "$1" in
        sc341)  echo "1:00:00" ;;
        sc917)  echo "1:30:00" ;;
        sc1941) echo "6:00:00" ;;
        *)      echo "1:30:00" ;;
    esac
}
flagship_label_for() {
    case "$1" in
        sc341)  echo "s24"  ;;
        sc917)  echo "s74"  ;;
        sc1941) echo "s139" ;;
        *)      echo "" ;;
    esac
}

# ---------- Parse args ----------
DRY_RUN=false
LIST_ONLY=false
COLD_SAFETY=false
SIZE="small"
SC="sc1941"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=true; shift ;;
        --list)        LIST_ONLY=true; shift ;;
        --cold-safety) COLD_SAFETY=true; SC="sc341"; shift ;;
        --size)        SIZE="$2"; shift 2 ;;
        --vqvae)       SC="$2"; shift 2 ;;
        --gains)       GAINS="$2"; shift 2 ;;
        --tmax)        DRIFT_TMAX="$2"; shift 2 ;;
        --floor)       DRIFT_FLOOR="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--size <s>] [--vqvae <sc>] [--gains "g1 g2 ..."] [--tmax T] [--floor F] [--cold-safety] [--dry-run] [--list]
  --size <s>      VQ size (small|medium|large). Default: small.
  --vqvae <sc>    sc-config (sc341|sc917|sc1941). Default: sc1941.
  --gains "..."   Space-separated drift_gain values. Default: ${GAINS}.
  --tmax T        drift_tmax hard cap. Default: ${DRIFT_TMAX}.
  --floor F       drift_tau_floor (warm-only floor). Default: ${DRIFT_FLOOR}.
  --cold-safety   Phase 2: run on sc341 (cold-optimal) — the parameter-free gate.
  --dry-run       Print actions without submitting.
  --list          Print the plan and exit.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

LABEL="$(flagship_label_for ${SC})"
if [ -z "${LABEL}" ]; then echo "No flagship arch for ${SC}" >&2; exit 1; fi

VQVAE_NAME="${SIZE}-${SC}"
RUN_NAME="${VQVAE_NAME}-nsp-${LABEL}"
CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"
TRAIN_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}.npz"
VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
CELL_DIR="${DRIFT_BASE}/${RUN_NAME}"
LOG_DIR="${DRIFT_BASE}/logs"
WANDB_PROJECT="${WANDB_PROJECT_PREFIX}-${SIZE}"
WANDB_GROUP="${SC}"
WALLTIME="$(walltime_for ${SC})"
IC_CHUNK="$(ic_chunk_for ${SC})"

if [ "${LIST_ONLY}" = true ]; then
    echo "drift_warm gain sweep:"
    echo "  Cell:     ${RUN_NAME}   (cold-safety=${COLD_SAFETY})"
    echo "  Gains:    ${GAINS}    drift_tmax=${DRIFT_TMAX}  drift_tau_floor=${DRIFT_FLOOR}"
    echo "  Rollout:  N=${N_TRAJ}  steps=${N_STEPS}  ic_chunk=${IC_CHUNK}  posmask ON  walltime=${WALLTIME}"
    echo "  Anchor:   model teacher-forced H_ref (measure_model_entropy.py, per_position)"
    echo "  Wandb:    ${WANDB_PROJECT} / group=${WANDB_GROUP} / run=${RUN_NAME}-driftw-g<gain>"
    echo "  Output:   ${CELL_DIR}/driftw-g<gain>/"
    exit 0
fi

echo "=========================================="
echo "drift_warm gain sweep (N=${N_TRAJ})"
echo "  Cell:   ${RUN_NAME}   cold-safety=${COLD_SAFETY}"
echo "  Gains:  ${GAINS}   tmax=${DRIFT_TMAX}  floor=${DRIFT_FLOOR}"
echo "  Wandb:  ${WANDB_PROJECT} / ${WANDB_GROUP}"
echo "  Dry run:${DRY_RUN}"
echo "=========================================="

if [ "${DRY_RUN}" = false ]; then
    if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        echo "[abort] ${RUN_NAME}: no NSP checkpoint at ${CHECKPOINT_DIR}" >&2; exit 1; fi
    if [ ! -f "${VAL_TOKENS}" ]; then
        echo "[abort] ${RUN_NAME}: no val tokens at ${VAL_TOKENS}" >&2; exit 1; fi
    if [ ! -f "${VQVAE_DIR}/training_state.json" ]; then
        echo "[abort] ${RUN_NAME}: no VQ-VAE checkpoint at ${VQVAE_DIR}" >&2; exit 1; fi
    mkdir -p "${CELL_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
fi

N_SUBMITTED=0
for GAIN in ${GAINS}; do
    ARM="driftw-g${GAIN}"
    ARM_DIR="${CELL_DIR}/${ARM}"
    ROUT="${ARM_DIR}/rollout"
    AOUT="${ARM_DIR}/analysis"
    ANCHOR="${ARM_DIR}/anchor.json"

    TMPFILE="$(mktemp /tmp/drift_${RUN_NAME}_g${GAIN}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J drift-${RUN_NAME}-g${GAIN}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t ${WALLTIME}
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-${ARM}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-${ARM}-%j.err

set -euo pipefail
cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "Job \${SLURM_JOB_ID} on \$(hostname)  started \$(date)"
echo "Cell ${RUN_NAME}  drift_gain=${GAIN}  tmax=${DRIFT_TMAX}  floor=${DRIFT_FLOOR}  (N=${N_TRAJ}, ic_chunk=${IC_CHUNK}, posmask ON)"
mkdir -p "${ARM_DIR}"

if [ -f "${AOUT}/metrics.json" ]; then
    echo "[skip] ${ARM}: analysis already complete"
else
    # H_ref = model teacher-forced per-scale predictive entropy (gain-independent;
    # rebuilt per gain-job in the gain-local dir -> no cross-job race).
    if [ ! -f "${ANCHOR}" ]; then
        echo "---- building model H_ref anchor: ${ANCHOR} ----"
        python measure_model_entropy.py --checkpoint_dir "${CHECKPOINT_DIR}" --tokens_path "${VAL_TOKENS}" --output "${ANCHOR}" --max_pairs 256
    fi

    if [ -f "${ROUT}/rollout_tokens.npz" ]; then
        echo "[reuse] ${ARM}: rollout exists, running analysis only"
    else
        python rollout_nsp.py \\
            --checkpoint_dir "${CHECKPOINT_DIR}" \\
            --tokens_path "${VAL_TOKENS}" \\
            --train_tokens_path "${TRAIN_TOKENS}" \\
            --start_frame ${START_FRAME} \\
            --n_steps ${N_STEPS} \\
            --n_trajectories ${N_TRAJ} \\
            --ic_chunk ${IC_CHUNK} \\
            --seed ${SEED} \\
            --temperature 1.0 \\
            --sampler drift_warm \\
            --drift_gain ${GAIN} \\
            --drift_tmax ${DRIFT_TMAX} \\
            --drift_tau_floor ${DRIFT_FLOOR} \\
            --anchor_stat per_position \\
            --entropy_anchor_path "${ANCHOR}" \\
            --output_dir "${ROUT}"
    fi

    python analyze_rollout.py \\
        --rollout_dir "${ROUT}" \\
        --vqvae_dir "${VQVAE_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --output_dir "${AOUT}" \\
        --batch_size ${BATCH_SIZE} \\
        --seed ${SEED} \\
        --wandb_project ${WANDB_PROJECT} \\
        --wandb_name "${RUN_NAME}-${ARM}" \\
        --wandb_group "${WANDB_GROUP}" \\
        --wandb_dir "${WANDB_BASE}"
fi
echo "Finished \$(date)"
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME} ${ARM}  N=${N_TRAJ} ic_chunk=${IC_CHUNK} walltime=${WALLTIME}  (tmax=${DRIFT_TMAX} floor=${DRIFT_FLOOR})"
    else
        echo "Submitting ${RUN_NAME} ${ARM}..."
        JOBID=$(sbatch --parsable "${TMPFILE}")
        echo "  -> ${JOBID}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
    rm -f "${TMPFILE}"
done

echo "=========================================="
echo "Submitted ${N_SUBMITTED} drift_warm gain jobs for ${RUN_NAME}."
echo "Compare EMD vs the swept-T yardstick (NEVER TKE RSE):"
echo "  best_T=2.0 EMD 0.196 (2.40x floor 0.0816); etmodel ~73x floor."
echo "=========================================="
