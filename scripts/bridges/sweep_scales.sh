#!/bin/bash
# Launch scale configuration sweep for a given model size.
#
# Usage:
#   ./scripts/sweep_scales.sh small          Submit 3 jobs (1 GPU each)
#   ./scripts/sweep_scales.sh medium         Submit 3 jobs (2 GPUs each)
#   ./scripts/sweep_scales.sh large          Submit 3 jobs (4 GPUs each)
#   ./scripts/sweep_scales.sh small --dry-run
#   ./scripts/sweep_scales.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
EXPERIMENT_BASE="${OCEAN}/experiments/vqvae"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Shared model config ----------
D_MODEL=512
N_HEADS=8
MLP_DIM=1024
CODEBOOK_DIM=512
CODEBOOK_SIZE=4096
ROPE_THETA=32.0

# ---------- Shared training config ----------
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
BETA=0.1
EMA_DECAY=0.85
SAMPLE_STOP=20000
SEED=42
WANDB_PROJECT="gust2-experiments"

# ---------- Scale configs (shared across all sizes) ----------
SCALE_CONFIGS=(
    "sc341:1,2,4,8,16:341"
    "sc917:1,2,4,8,16,24:917"
    "sc1941:1,2,4,8,16,24,32:1941"
)

# ---------- Model size definitions ----------
set_model_size() {
    case "$1" in
        small)
            ENCODER_DEPTH=5; DECODER_DEPTH=5; N_GPUS=1
            WANDB_GROUP="small"
            ;;
        medium)
            ENCODER_DEPTH=10; DECODER_DEPTH=10; N_GPUS=2
            WANDB_GROUP="medium"
            ;;
        large)
            ENCODER_DEPTH=20; DECODER_DEPTH=20; N_GPUS=4
            WANDB_GROUP="large"
            ;;
        *)
            echo "Unknown size: $1. Use small, medium, or large." >&2
            exit 1
            ;;
    esac

    if [ "${N_GPUS}" -le 4 ]; then
        PARTITION="GPU-shared"
    else
        PARTITION="GPU"
    fi
}

# ---------- Parse args ----------
DRY_RUN=false
MODEL_SIZE=""

for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --list)
            echo "Scale configs (applied to each model size):"
            for cfg in "${SCALE_CONFIGS[@]}"; do
                IFS=: read -r sc scales tokens <<< "${cfg}"
                echo "  ${sc}: scales=(${scales}), ${tokens} tokens/sample"
            done
            echo ""
            echo "Model sizes:"
            echo "  small:  depth 5+5,   1 GPU"
            echo "  medium: depth 10+10, 2 GPUs"
            echo "  large:  depth 20+20, 4 GPUs"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 <small|medium|large> [--dry-run] [--list]"
            exit 0
            ;;
        small|medium|large)
            MODEL_SIZE="${arg}"
            ;;
    esac
done

if [ -z "${MODEL_SIZE}" ]; then
    echo "Error: specify model size (small, medium, or large)"
    echo "Usage: $0 <small|medium|large> [--dry-run]"
    exit 1
fi

set_model_size "${MODEL_SIZE}"

echo "=========================================="
echo "Scale Sweep: ${MODEL_SIZE}"
echo "  depth=${ENCODER_DEPTH}+${DECODER_DEPTH}, ${N_GPUS} GPU(s)"
echo "  d=${D_MODEL}, cdim=${CODEBOOK_DIM}, K=${CODEBOOK_SIZE}"
echo "  beta=${BETA}, decay=${EMA_DECAY}, batch=${BATCH_SIZE}"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for cfg in "${SCALE_CONFIGS[@]}"; do
    IFS=: read -r SC SCALES TOKENS <<< "${cfg}"
    RUN_NAME="${MODEL_SIZE}-${SC}"
    CHECKPOINT_DIR="${EXPERIMENT_BASE}/${RUN_NAME}"

    # Ensure output dirs exist before sbatch
    mkdir -p "${CHECKPOINT_DIR}" "${EXPERIMENT_BASE}/logs" "${WANDB_BASE}"

    # Auto-detect resume
    RESUME_FLAG=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        RESUME_FLAG="--resume"
    fi

    # Generate sbatch script
    TMPFILE="$(mktemp /tmp/sweep_${RUN_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:${N_GPUS}
#SBATCH -t 2-00:00:00
#SBATCH -o ${EXPERIMENT_BASE}/logs/${RUN_NAME}-%j.out
#SBATCH -e ${EXPERIMENT_BASE}/logs/${RUN_NAME}-%j.err

# ---------- Setup ----------
cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1

# cuDNN and NVIDIA libs from pip packages
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "GPUs:      ${N_GPUS}"
echo "Run:       ${RUN_NAME}"
echo "Model:     ${MODEL_SIZE} (${ENCODER_DEPTH}+${DECODER_DEPTH})"
echo "Scales:    ${SCALES} (${TOKENS} tokens/sample)"
echo "Ckpt dir:  ${CHECKPOINT_DIR}"
echo "Resume:    ${RESUME_FLAG:-no}"
echo "=========================================="

python train.py \\
    --data_path "${DATA_PATH}" \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --d_model ${D_MODEL} \\
    --n_heads ${N_HEADS} \\
    --mlp_dim ${MLP_DIM} \\
    --encoder_depth ${ENCODER_DEPTH} \\
    --decoder_depth ${DECODER_DEPTH} \\
    --codebook_dim ${CODEBOOK_DIM} \\
    --codebook_size ${CODEBOOK_SIZE} \\
    --scales ${SCALES} \\
    --rope_theta ${ROPE_THETA} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --lr ${LR} \\
    --beta ${BETA} \\
    --ema_decay ${EMA_DECAY} \\
    --sample_stop ${SAMPLE_STOP} \\
    --seed ${SEED} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}: scales=(${SCALES}), ${TOKENS} tokens, ${N_GPUS} GPU(s)"
    else
        echo "Submitting ${RUN_NAME}: scales=(${SCALES}), ${TOKENS} tokens, ${N_GPUS} GPU(s)..."
        sbatch "${TMPFILE}"
    fi
    rm -f "${TMPFILE}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
