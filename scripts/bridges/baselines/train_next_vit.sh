#!/bin/bash
# B2 — ViT next-frame regressor baseline (BASELINES_SPEC.md §B2, decision
# 2026-07-01: ViT-AE replaces the U-Net/FNO plan).
#
# The tokenizer backbone with quantization stripped: vit_ae Encoder+Decoder
# end-to-end, MSE regression t0 -> t1 (train_next_vit.py). Architecture-
# matched to the VQ-VAE family so quantize-or-don't is the isolated variable.
#
# Arms (spec: plain / +noise x2 sigmas at one size / +pushforward / 2nd size):
#   nextvit-small-plain      5/5  (~31M)  1 GPU
#   nextvit-small-noise-lo   5/5  --noise_sigma 0.01          1 GPU
#   nextvit-small-noise-hi   5/5  --noise_sigma 0.1           1 GPU
#   nextvit-small-pushfwd    5/5  --pushforward (2-step, sg)  2 GPU (2x activ.)
#   nextvit-medium-plain     10/10 (~57M)                     2 GPU
#
# Each arm: training job (2d) -> chained rollout+analysis job (afterok, 1 GPU):
#   rollout_continuous.py  8 ICs spread over val [20000,22000), 2000 steps, f32
#   analyze_continuous.py  GT/one-step/rollout spectra + band traces (F7.1)
#
# Usage:
#   ./scripts/bridges/baselines/train_next_vit.sh                # all 5 arms
#   ./scripts/bridges/baselines/train_next_vit.sh --only pushfwd # substring
#   ./scripts/bridges/baselines/train_next_vit.sh --rollout-only # skip training
#   ./scripts/bridges/baselines/train_next_vit.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
BASE="${OCEAN}/experiments/baselines"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Shared config (matches the tokenizer family recipe) ----------
D_MODEL=512
N_HEADS=8
MLP_DIM=1024
LATENT_DIM=512
ROPE_THETA=32.0
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
SEED=42
WANDB_PROJECT="gust2-baselines"
WANDB_GROUP="next-vit"

# Rollout / analysis
N_ICS=8
N_STEPS=2000
VAL_START=20000
VAL_STOP=22000

# ---------- Arms: name:depth:gpus:extra_flags ----------
ARMS=(
    "small-plain:5:1:"
    "small-noise-lo:5:1:--noise_sigma 0.01"
    "small-noise-hi:5:1:--noise_sigma 0.1"
    "small-pushfwd:5:2:--pushforward"
    "medium-plain:10:2:"
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
echo "B2 next-ViT baseline arms"
echo "  batch=${BATCH_SIZE} epochs=${EPOCHS} lr=${LR}"
echo "  rollout: ${N_ICS} ICs x ${N_STEPS} steps from val [${VAL_START},${VAL_STOP})"
echo "  Wandb: ${WANDB_PROJECT} / group=${WANDB_GROUP}"
echo "  Dry run: ${DRY_RUN}   Rollout-only: ${ROLLOUT_ONLY}"
echo "=========================================="

N_SUBMITTED=0

for arm in "${ARMS[@]}"; do
    IFS=: read -r ARM_NAME DEPTH GPUS EXTRA_FLAGS <<< "${arm}"
    if [ -n "${ONLY}" ] && [[ "${ARM_NAME}" != *"${ONLY}"* ]]; then continue; fi

    RUN_NAME="nextvit-${ARM_NAME}"
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
        TMPFILE="$(mktemp /tmp/nextvit_${ARM_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:${GPUS}
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
echo "Run:      ${RUN_NAME}  (depth ${DEPTH}/${DEPTH}, ${GPUS} GPU, '${EXTRA_FLAGS}')"
echo "Resume:   ${RESUME_FLAG:-no}"
echo "=========================================="

python train_next_vit.py \\
    --data_path "${DATA_PATH}" \\
    --checkpoint_dir "${CKPT_DIR}" \\
    --d_model ${D_MODEL} \\
    --n_heads ${N_HEADS} \\
    --mlp_dim ${MLP_DIM} \\
    --encoder_depth ${DEPTH} \\
    --decoder_depth ${DEPTH} \\
    --latent_dim ${LATENT_DIM} \\
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
    --wandb_id b2-${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${EXTRA_FLAGS} ${RESUME_FLAG}

echo "Finished: \$(date)"
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] train ${RUN_NAME}: depth ${DEPTH}/${DEPTH}, ${GPUS} GPU, flags='${EXTRA_FLAGS}'"
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

    TMPFILE="$(mktemp /tmp/nextvit_ra_${ARM_NAME}_XXXXXX.sbatch)"
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
    --wandb_group baselines-nextvit \\
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
