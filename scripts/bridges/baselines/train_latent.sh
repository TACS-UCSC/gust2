#!/bin/bash
# B1 — skip-quantization continuous-latent baseline (BASELINES_SPEC.md §B1).
#
# Frozen medium-sc341 VQ-VAE enc/dec (locked single-anchor decision),
# quantizer bypassed; an s18-class-matched NSP-recipe transformer regresses
# z_{t+1} from z_t with MSE (train_latent.py, ~18.1M dynamics params at
# 10 x d384/h6). Closed-loop rollout runs entirely in latent space; the
# decoder is per-step readout only (rollout_continuous.py latent branch).
#
# Arms (spec: sigma in {0, lo, hi} x per-channel latent std, probe-calibrated):
#   latent-mse-plain       --noise_sigma 0      1 GPU
#   latent-mse-noise-lo    --noise_sigma 0.01   1 GPU
#   latent-mse-noise-hi    --noise_sigma 0.1    1 GPU
#
# Each arm: training job (2d cap) -> chained rollout+analysis job (afterok):
#   rollout_continuous.py  8 ICs spread over val [20000,22000), 2000 steps, f32
#   analyze_continuous.py  GT/one-step/rollout spectra + band traces (F7.1)
#
# Usage:
#   ./scripts/bridges/baselines/train_latent.sh                 # all 3 arms
#   ./scripts/bridges/baselines/train_latent.sh --only noise-lo # substring
#   ./scripts/bridges/baselines/train_latent.sh --rollout-only  # skip training
#   ./scripts/bridges/baselines/train_latent.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
BASE="${OCEAN}/experiments/baselines"
VQVAE_CKPT="${OCEAN}/experiments/vqvae/medium-sc341"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Dynamics config (s18-class param match: ~18.1M) ----------
N_LAYER=10
N_EMBD=384
N_HEAD=6
ROPE_THETA=32.0
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
SEED=42
WANDB_PROJECT="gust2-baselines"
WANDB_GROUP="latent-mse"

# Rollout / analysis
N_ICS=8
N_STEPS=2000
VAL_START=20000
VAL_STOP=22000

# ---------- Arms: name:extra_flags ----------
ARMS=(
    "plain:"
    "noise-lo:--noise_sigma 0.01"
    "noise-hi:--noise_sigma 0.1"
)

# ---------- Parse args ----------
DRY_RUN=false
ONLY=""
ROLLOUT_ONLY=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --rollout-only) ROLLOUT_ONLY=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--only <substr>] [--rollout-only] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=========================================="
echo "B1 continuous-latent baseline arms"
echo "  anchor: ${VQVAE_CKPT} (quantizer bypassed)"
echo "  dynamics: ${N_LAYER} x d${N_EMBD}/h${N_HEAD}  batch=${BATCH_SIZE} epochs=${EPOCHS} lr=${LR}"
echo "  rollout: ${N_ICS} ICs x ${N_STEPS} steps from val [${VAL_START},${VAL_STOP})"
echo "  Wandb: ${WANDB_PROJECT} / group=${WANDB_GROUP}"
echo "  Dry run: ${DRY_RUN}   Rollout-only: ${ROLLOUT_ONLY}"
echo "=========================================="

N_SUBMITTED=0

for arm in "${ARMS[@]}"; do
    IFS=: read -r ARM_NAME EXTRA_FLAGS <<< "${arm}"
    if [ -n "${ONLY}" ] && [[ "${ARM_NAME}" != *"${ONLY}"* ]]; then continue; fi

    RUN_NAME="latent-mse-${ARM_NAME}"
    CKPT_DIR="${BASE}/${RUN_NAME}"
    ROLLOUT_DIR="${CKPT_DIR}/rollout"
    ANALYSIS_DIR="${CKPT_DIR}/analysis"
    LOG_DIR="${BASE}/logs"

    if [ "${DRY_RUN}" = false ]; then
        mkdir -p "${CKPT_DIR}" "${ROLLOUT_DIR}" "${ANALYSIS_DIR}" \
                 "${LOG_DIR}" "${WANDB_BASE}"
    fi

    # Training completeness / resume detection
    STATE_FILE="${CKPT_DIR}/training_state.json"
    RESUME_FLAG=""
    TRAIN_DONE=false
    if [ -f "${STATE_FILE}" ]; then
        RESUME_FLAG="--resume"
        DONE_EPOCH=$(grep -o '"epoch": *[0-9]*' "${STATE_FILE}" | grep -o '[0-9]*' || echo 0)
        if [ "${DONE_EPOCH:-0}" -ge "${EPOCHS}" ]; then
            TRAIN_DONE=true
        fi
    fi

    # ---------- Training job ----------
    TRAIN_JOB_ID=""
    if [ "${ROLLOUT_ONLY}" = false ] && [ "${TRAIN_DONE}" = false ]; then
        TMPFILE="$(mktemp /tmp/latentmse_${ARM_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2-00:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:      \${SLURM_JOB_ID} (\$(hostname), \$(date))"
echo "Run:      ${RUN_NAME}  (${N_LAYER} x d${N_EMBD}/h${N_HEAD}, '${EXTRA_FLAGS}')"
echo "Resume:   ${RESUME_FLAG:-no}"
echo "=========================================="

python train_latent.py \\
    --data_path "${DATA_PATH}" \\
    --vqvae_checkpoint "${VQVAE_CKPT}" \\
    --checkpoint_dir "${CKPT_DIR}" \\
    --n_layer ${N_LAYER} \\
    --n_embd ${N_EMBD} \\
    --n_head ${N_HEAD} \\
    --rope_theta ${ROPE_THETA} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --lr ${LR} \\
    --seed ${SEED} \\
    --sample_start 0 \\
    --sample_stop 20000 \\
    --val_start ${VAL_START} \\
    --val_stop ${VAL_STOP} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_id b1-${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${EXTRA_FLAGS} ${RESUME_FLAG}

echo "Finished: \$(date)"
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] train ${RUN_NAME}: flags='${EXTRA_FLAGS}'"
        else
            echo "Submitting train ${RUN_NAME}..."
            TRAIN_JOB_ID=$(sbatch --parsable "${TMPFILE}")
            echo "  -> job ${TRAIN_JOB_ID}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        fi
        rm -f "${TMPFILE}"
    else
        echo "Skip train ${RUN_NAME} (done=${TRAIN_DONE}, rollout-only=${ROLLOUT_ONLY})"
    fi

    # ---------- Rollout + analysis job (chained) ----------
    if [ -f "${ANALYSIS_DIR}/metrics.json" ] && [ -f "${ROLLOUT_DIR}/rollout_fields.npz" ]; then
        echo "Skip rollout+analysis ${RUN_NAME} (metrics.json exists)"
        continue
    fi

    DEP_FLAG=""
    if [ -n "${TRAIN_JOB_ID}" ]; then
        DEP_FLAG="--dependency=afterok:${TRAIN_JOB_ID}"
    fi

    TMPFILE="$(mktemp /tmp/latentmse_ra_${ARM_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}-ra
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 2:00:00
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-ra-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-ra-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${OCEAN}/.venvs/gust/bin/activate"
module load cuda/12.6.1
export PYTHONUNBUFFERED=1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:      \${SLURM_JOB_ID} (\$(hostname), \$(date))"
echo "Rollout+analysis: ${RUN_NAME}"
echo "=========================================="

if [ ! -f "${ROLLOUT_DIR}/rollout_fields.npz" ]; then
    python rollout_continuous.py \\
        --checkpoint_dir "${CKPT_DIR}" \\
        --data_path "${DATA_PATH}" \\
        --sample_start ${VAL_START} \\
        --sample_stop ${VAL_STOP} \\
        --n_ics ${N_ICS} \\
        --n_steps ${N_STEPS} \\
        --output_dir "${ROLLOUT_DIR}"
else
    echo "rollout_fields.npz exists — skipping rollout"
fi

python analyze_continuous.py \\
    --rollout_dir "${ROLLOUT_DIR}" \\
    --checkpoint_dir "${CKPT_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --sample_start ${VAL_START} \\
    --sample_stop ${VAL_STOP} \\
    --output_dir "${ANALYSIS_DIR}" \\
    --wandb_project gust2-analysis \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group baselines-latentmse \\
    --wandb_dir "${WANDB_BASE}"

echo "Finished: \$(date)"
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] rollout+analysis ${RUN_NAME} (dep: ${DEP_FLAG:-none})"
    else
        echo "Submitting rollout+analysis ${RUN_NAME}..."
        RA_JOB_ID=$(sbatch --parsable ${DEP_FLAG} "${TMPFILE}")
        echo "  -> job ${RA_JOB_ID} ${DEP_FLAG}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    fi
    rm -f "${TMPFILE}"
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
