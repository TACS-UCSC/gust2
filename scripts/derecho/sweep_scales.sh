#!/bin/bash
# Launch scale configuration sweep for a given model size on NCAR Derecho.
#
# Derecho A100 nodes are node-exclusive on the `main` queue, so every job
# takes a full 4-GPU node regardless of ngpus requested. All sizes therefore
# use 4 GPUs on 1 node. Global batch size is halved vs Bridges (32 instead
# of 64) to fit A100-40 memory for the large config; kept uniform across all
# 9 runs for scaling-law consistency.
#
# Usage:
#   ./scripts/derecho/sweep_scales.sh small          Submit 3 jobs
#   ./scripts/derecho/sweep_scales.sh medium         Submit 3 jobs
#   ./scripts/derecho/sweep_scales.sh large          Submit 3 jobs
#   ./scripts/derecho/sweep_scales.sh small --dry-run
#   ./scripts/derecho/sweep_scales.sh --list

set -euo pipefail

# ---------- Paths ----------
REPODIR="${HOME}/gust2"
VENV="${SCRATCH}/.venvs/gust2"
DATA_PATH="${SCRATCH}/turb2d_long/output.h5"
EXPERIMENT_BASE="${SCRATCH}/experiments/vqvae"
WANDB_BASE="${SCRATCH}/wandb"
ACCOUNT="UCSC0009"

# ---------- Shared model config ----------
D_MODEL=512
N_HEADS=8
MLP_DIM=1024
CODEBOOK_DIM=512
CODEBOOK_SIZE=4096
ROPE_THETA=32.0

# ---------- Shared training config ----------
BATCH_SIZE=32
EPOCHS=100
LR=1e-4
BETA=0.1
EMA_DECAY=0.85
SAMPLE_STOP=20000
SEED=42
WANDB_PROJECT="gust2-experiments-derecho"

# ---------- Scale configs (shared across all sizes) ----------
SCALE_CONFIGS=(
    "sc341:1,2,4,8,16:341"
    "sc917:1,2,4,8,16,24:917"
    "sc1941:1,2,4,8,16,24,32:1941"
)

# ---------- Model size definitions ----------
# All sizes run on 1 Derecho A100 node (4x A100-40GB), since `main` is
# node-exclusive and we cannot share a node with other jobs.
set_model_size() {
    case "$1" in
        small)
            ENCODER_DEPTH=5; DECODER_DEPTH=5
            WANDB_GROUP="small"
            ;;
        medium)
            ENCODER_DEPTH=10; DECODER_DEPTH=10
            WANDB_GROUP="medium"
            ;;
        large)
            ENCODER_DEPTH=20; DECODER_DEPTH=20
            WANDB_GROUP="large"
            ;;
        *)
            echo "Unknown size: $1. Use small, medium, or large." >&2
            exit 1
            ;;
    esac
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
            echo "Model sizes (all: 1 node, 4x A100-40):"
            echo "  small:  depth 5+5"
            echo "  medium: depth 10+10"
            echo "  large:  depth 20+20"
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
echo "Scale Sweep (Derecho): ${MODEL_SIZE}"
echo "  depth=${ENCODER_DEPTH}+${DECODER_DEPTH}, 4x A100-40"
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
    LOG_DIR="${EXPERIMENT_BASE}/logs"

    mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${WANDB_BASE}"

    # Auto-detect resume
    RESUME_FLAG=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        RESUME_FLAG="--resume"
    fi

    TMPFILE="$(mktemp /tmp/sweep_${RUN_NAME}_XXXXXX.pbs)"
    cat > "${TMPFILE}" << PBS_EOF
#!/bin/bash
#PBS -N ${RUN_NAME}
#PBS -A ${ACCOUNT}
#PBS -q main
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=64:ngpus=4:mpiprocs=1:mem=480GB
#PBS -j oe
#PBS -o ${LOG_DIR}/${RUN_NAME}.log

set -euo pipefail

# ---------- Setup ----------
cd "${REPODIR}"

export TMPDIR="\${SCRATCH}/\${USER}/tmpdir"
mkdir -p "\${TMPDIR}"

module purge
module load ncarenv

source "${VENV}/bin/activate"

# JAX is installed with its own cuda13/cudnn via pip (driver is CUDA 13).
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:       \${PBS_JOBID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "GPUs:      4x A100-40 (node-exclusive)"
echo "Run:       ${RUN_NAME}"
echo "Model:     ${MODEL_SIZE} (${ENCODER_DEPTH}+${DECODER_DEPTH})"
echo "Scales:    ${SCALES} (${TOKENS} tokens/sample)"
echo "Batch:     ${BATCH_SIZE}"
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
PBS_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}: scales=(${SCALES}), ${TOKENS} tokens"
        echo "  script: ${TMPFILE}"
    else
        echo "Submitting ${RUN_NAME}: scales=(${SCALES}), ${TOKENS} tokens..."
        qsub "${TMPFILE}"
        rm -f "${TMPFILE}"
    fi
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
