#!/bin/bash
# Launch Phase 1 codebook sweep: codebook_dim={64,128} × codebook_size={256,512,1024,2048,4096}
# Small model (D=5), 1 H100 per run, 50 epochs, batch=64
#
# Usage:
#   ./scripts/sweep_codebook.sh              Submit all 10 jobs
#   ./scripts/sweep_codebook.sh --dry-run    Print sbatch commands without submitting
#   ./scripts/sweep_codebook.sh --list       List all configs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
SWEEP_BASE="${OCEAN}/sweeps/codebook"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Fixed model config (small, D=5) ----------
D_MODEL=512
N_HEADS=8
MLP_DIM=1024
ENCODER_DEPTH=5
DECODER_DEPTH=5
SCALES="1,2,4,8,16,32"
ROPE_THETA=32.0

# ---------- Fixed training config ----------
BATCH_SIZE=64
EPOCHS=50
LR=1e-4
BETA=0.25
EMA_DECAY=0.90
SEED=42
WANDB_PROJECT="gust2-vqvae"
WANDB_GROUP="codebook-sweep"

# ---------- Sweep grid ----------
CODEBOOK_DIMS="64 128"
CODEBOOK_SIZES="256 512 1024 2048 4096"

DRY_RUN=false

# ---------- Parse args ----------
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=true ;;
        --list)
            echo "Codebook sweep configs (codebook_dim × codebook_size):"
            for cdim in ${CODEBOOK_DIMS}; do
                for csz in ${CODEBOOK_SIZES}; do
                    echo "  cd${cdim}-K${csz}"
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
echo "Codebook Sweep (Phase 1)"
echo "Model: D=5 (small), d=${D_MODEL}"
echo "Grid: codebook_dim={${CODEBOOK_DIMS}} × K={${CODEBOOK_SIZES}}"
echo "Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for CDIM in ${CODEBOOK_DIMS}; do
    for CSZ in ${CODEBOOK_SIZES}; do
        RUN_NAME="cd${CDIM}-K${CSZ}"
        CHECKPOINT_DIR="${SWEEP_BASE}/${RUN_NAME}"

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

mkdir -p "${CHECKPOINT_DIR}"
mkdir -p "${SWEEP_BASE}/logs"
mkdir -p "${WANDB_BASE}"

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "GPUs:      \$(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Run:       ${RUN_NAME}"
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
    --codebook_dim ${CDIM} \\
    --codebook_size ${CSZ} \\
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
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] ${RUN_NAME}: cdim=${CDIM}, K=${CSZ}"
            echo "  sbatch ${TMPFILE}"
            rm -f "${TMPFILE}"
        else
            echo "Submitting ${RUN_NAME} (cdim=${CDIM}, K=${CSZ})..."
            sbatch "${TMPFILE}"
            rm -f "${TMPFILE}"
        fi
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) ${DRY_RUN:+would be }submitted."
