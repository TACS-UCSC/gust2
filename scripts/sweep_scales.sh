#!/bin/bash
# Launch scale configuration sweep.
# 3 configs at different token counts, all on the small model (D=5).
#
# Usage:
#   ./scripts/sweep_scales.sh              Submit all 3 jobs
#   ./scripts/sweep_scales.sh --dry-run    Print without submitting
#   ./scripts/sweep_scales.sh --list       List configs

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
EXPERIMENT_BASE="${OCEAN}/experiments/vqvae"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Fixed model config (small, D=5, best codebook from sweep) ----------
D_MODEL=512
N_HEADS=8
MLP_DIM=1024
ENCODER_DEPTH=5
DECODER_DEPTH=5
CODEBOOK_DIM=512
CODEBOOK_SIZE=4096
ROPE_THETA=32.0

# ---------- Fixed training config ----------
BATCH_SIZE=64
EPOCHS=100
LR=1e-4
BETA=0.25
EMA_DECAY=0.90
SEED=42
WANDB_PROJECT="gust2-experiments"

# ---------- Scale configs ----------
# name:scales:tokens
CONFIGS=(
    "small-sc341:1,2,4,8,16:341"
    "small-sc917:1,2,4,8,16,24:917"
    "small-sc1941:1,2,4,8,16,24,32:1941"
)

DRY_RUN=false

# ---------- Parse args ----------
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --list)
            echo "Scale sweep configs:"
            for cfg in "${CONFIGS[@]}"; do
                IFS=: read -r name scales tokens <<< "${cfg}"
                echo "  ${name}: scales=(${scales}), ${tokens} tokens/sample"
            done
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--list]"
            exit 0
            ;;
    esac
done

echo "=========================================="
echo "Scale Sweep"
echo "Model: D=5 (small), d=${D_MODEL}, cdim=${CODEBOOK_DIM}, K=${CODEBOOK_SIZE}"
echo "Configs: ${#CONFIGS[@]}"
echo "Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r RUN_NAME SCALES TOKENS <<< "${cfg}"
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
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH -t 12:00:00
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
echo "GPUs:      \$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Run:       ${RUN_NAME}"
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
    --seed ${SEED} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group small \\
    --wandb_dir "${WANDB_BASE}" \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}: scales=(${SCALES}), ${TOKENS} tokens/sample"
    else
        echo "Submitting ${RUN_NAME}: scales=(${SCALES}), ${TOKENS} tokens/sample..."
        sbatch "${TMPFILE}"
    fi
    rm -f "${TMPFILE}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
