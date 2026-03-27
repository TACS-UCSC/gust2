#!/bin/bash
# Launch Phase 2 dynamics sweep: EMA decay × commitment weight (beta)
# Small model (D=5), cd512-K4096, 1 H100 per run, 50 epochs
#
# Usage:
#   ./scripts/sweep_dynamics.sh              Submit all jobs
#   ./scripts/sweep_dynamics.sh --dry-run    Print without submitting
#   ./scripts/sweep_dynamics.sh --list       List configs

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
SWEEP_BASE="${OCEAN}/sweeps/dynamics"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Fixed model config ----------
D_MODEL=512
N_HEADS=8
MLP_DIM=1024
ENCODER_DEPTH=5
DECODER_DEPTH=5
CODEBOOK_DIM=512
CODEBOOK_SIZE=4096
SCALES="1,2,4,8,16,32"
ROPE_THETA=32.0

# ---------- Fixed training config ----------
BATCH_SIZE=64
EPOCHS=50
LR=1e-4
SEED=42
WANDB_PROJECT="gust2-vqvae"
WANDB_GROUP="dynamics-sweep"

# ---------- Sweep grid ----------
EMA_DECAYS="0.85 0.90 0.95 0.99"
BETAS="0.1 0.5 1.0"

DRY_RUN=false

# ---------- Parse args ----------
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --list)
            echo "Dynamics sweep configs (ema_decay × beta):"
            for decay in ${EMA_DECAYS}; do
                for beta in ${BETAS}; do
                    echo "  d${decay}-b${beta}"
                done
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
echo "Dynamics Sweep (Phase 2)"
echo "Model: D=5 (small), cd512-K4096"
echo "Grid: decay={${EMA_DECAYS}} × beta={${BETAS}}"
echo "Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for DECAY in ${EMA_DECAYS}; do
    for BETA in ${BETAS}; do
        RUN_NAME="d${DECAY}-b${BETA}"
        CHECKPOINT_DIR="${SWEEP_BASE}/${RUN_NAME}"

        # Ensure output dirs exist before sbatch
        mkdir -p "${CHECKPOINT_DIR}" "${SWEEP_BASE}/logs" "${WANDB_BASE}"

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
#SBATCH -o ${SWEEP_BASE}/logs/${RUN_NAME}-%j.out
#SBATCH -e ${SWEEP_BASE}/logs/${RUN_NAME}-%j.err

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
echo "EMA decay: ${DECAY}"
echo "Beta:      ${BETA}"
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
    --ema_decay ${DECAY} \\
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
            echo "[dry-run] ${RUN_NAME}: decay=${DECAY}, beta=${BETA}"
        else
            echo "Submitting ${RUN_NAME}: decay=${DECAY}, beta=${BETA}..."
            sbatch "${TMPFILE}"
        fi
        rm -f "${TMPFILE}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
