#!/bin/bash
# Launch NSP training sweep: 3 NSP sizes × 9 tokenized datasets = 27 jobs.
#
# Trains small/medium/large next-scale prediction models on every
# VQ-VAE tokenized dataset ({vqvae_size}-{scale_config}).
#
# Usage:
#   ./scripts/sweep_nsp.sh                 Submit all 27 jobs
#   ./scripts/sweep_nsp.sh small           Submit 9 jobs (small NSP only)
#   ./scripts/sweep_nsp.sh --vqvae medium  Only train on medium VQ-VAE tokens
#   ./scripts/sweep_nsp.sh small --vqvae small-sc341   Single job
#   ./scripts/sweep_nsp.sh --dry-run
#   ./scripts/sweep_nsp.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
# Fresh checkpoint dir for the refinement sweep so we don't collide
# with the original nsp-branch runs (whose arch_config lacks
# n_refine_layers / auto RoPE theta and would trigger an architecture
# mismatch error on --resume).
AR_BASE="${OCEAN}/experiments/ar-refine"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Shared model config ----------
N_EMBD=1024
N_HEAD=8
# RoPE theta auto-resolves to finest grid side per sc config (16/24/32)
N_REFINE_LAYERS=2

# ---------- Shared training config ----------
# Single H100 per job, batch 16 (matches the local-verified recipe with
# the within-scale refinement + bug fix from nsp-refinement). Heavy jobs
# (large × sc1941, etc.) may not finish 400 epochs in the 2-day Slurm
# limit; rerun with --resume to continue.
BATCH_SIZE=16
EPOCHS=400
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
CONTEXT_DROP_RATE=0.25
SEED=42
WANDB_PROJECT="gust2-nsp-refine"

# ---------- VQ-VAE source configs ----------
VQVAE_NAMES=(
    small-sc341  small-sc917  small-sc1941
    medium-sc341 medium-sc917 medium-sc1941
    large-sc341  large-sc917  large-sc1941
)

# ---------- NSP model sizes ----------
# (name, n_layer, n_gpus)
set_nsp_size() {
    # Single H100 per job for all NSP sizes. Only n_layer scales; n_embd
    # and n_refine_layers stay shared so ExpansionHeads parameter count is
    # identical across small/medium/large within each VQ-VAE config.
    case "$1" in
        small)  N_LAYER=4 ;;
        medium) N_LAYER=8 ;;
        large)  N_LAYER=16 ;;
        *)
            echo "Unknown NSP size: $1. Use small, medium, or large." >&2
            exit 1
            ;;
    esac
    N_GPUS=1
    PARTITION="GPU-shared"
}

# ---------- Parse args ----------
DRY_RUN=false
FILTER_NSP=""
FILTER_VQVAE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --list)
            echo "NSP sizes (1 H100 each, batch=${BATCH_SIZE}):"
            echo "  small:  4L,  n_embd=${N_EMBD}, refine=${N_REFINE_LAYERS}"
            echo "  medium: 8L,  n_embd=${N_EMBD}, refine=${N_REFINE_LAYERS}"
            echo "  large:  16L, n_embd=${N_EMBD}, refine=${N_REFINE_LAYERS}"
            echo "  (ExpansionHeads identical across sizes within a VQ-VAE config)"
            echo ""
            echo "VQ-VAE token sources:"
            for v in "${VQVAE_NAMES[@]}"; do echo "  ${v}"; done
            echo ""
            echo "Total: 3 × 9 = 27 jobs, ${EPOCHS} epochs, drop=${CONTEXT_DROP_RATE}"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [small|medium|large] [--vqvae <name>] [--dry-run] [--list]"
            exit 0
            ;;
        --vqvae)
            FILTER_VQVAE="$2"; shift 2
            ;;
        small|medium|large)
            FILTER_NSP="$1"; shift
            ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1
            ;;
    esac
done

# Build list of NSP sizes to run
if [ -n "${FILTER_NSP}" ]; then
    NSP_SIZES=("${FILTER_NSP}")
else
    NSP_SIZES=(small medium large)
fi

# Build list of VQ-VAE sources to run
if [ -n "${FILTER_VQVAE}" ]; then
    FILTERED_VQVAE=()
    for v in "${VQVAE_NAMES[@]}"; do
        if [[ "${v}" == *"${FILTER_VQVAE}"* ]]; then
            FILTERED_VQVAE+=("${v}")
        fi
    done
    VQVAE_NAMES=("${FILTERED_VQVAE[@]}")
fi

echo "=========================================="
echo "NSP Training Sweep"
echo "  n_embd=${N_EMBD}, batch=${BATCH_SIZE}"
echo "  NSP sizes: ${NSP_SIZES[*]}"
echo "  VQ-VAE sources: ${#VQVAE_NAMES[@]} configs"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for VQVAE_NAME in "${VQVAE_NAMES[@]}"; do
    TOKENS_PATH="${TOKENS_BASE}/${VQVAE_NAME}.npz"

    # Check tokens exist
    if [ ! -f "${TOKENS_PATH}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${VQVAE_NAME}: tokens not found at ${TOKENS_PATH}"
        continue
    fi

    for NSP_SIZE in "${NSP_SIZES[@]}"; do
        set_nsp_size "${NSP_SIZE}"
        RUN_NAME="${VQVAE_NAME}-nsp-${NSP_SIZE}"
        CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"

        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${CHECKPOINT_DIR}" "${AR_BASE}/logs" "${WANDB_BASE}"
        fi

        # Extract VQ-VAE size for wandb grouping
        VQVAE_SIZE="${VQVAE_NAME%%-*}"
        WANDB_GROUP="${VQVAE_SIZE}-nsp-${NSP_SIZE}"

        # Auto-detect resume
        RESUME_FLAG=""
        if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
            RESUME_FLAG="--resume"
        fi

        # Wandb ID for resume continuity
        WANDB_ID_FLAG=""
        WANDB_ID_FILE="${CHECKPOINT_DIR}/wandb_id.txt"
        if [ -f "${WANDB_ID_FILE}" ]; then
            WANDB_ID_FLAG="--wandb_id $(cat "${WANDB_ID_FILE}")"
        fi

        TMPFILE="$(mktemp /tmp/nsp_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:${N_GPUS}
#SBATCH -t 2-00:00:00
#SBATCH -o ${AR_BASE}/logs/${RUN_NAME}-%j.out
#SBATCH -e ${AR_BASE}/logs/${RUN_NAME}-%j.err

# ---------- Setup ----------
cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "GPUs:      ${N_GPUS}"
echo "Run:       ${RUN_NAME}"
echo "NSP:       ${NSP_SIZE} (${N_LAYER}L, n_embd=${N_EMBD})"
echo "Tokens:    ${TOKENS_PATH}"
echo "Ckpt dir:  ${CHECKPOINT_DIR}"
echo "Resume:    ${RESUME_FLAG:-no}"
echo "=========================================="

python train_nsp.py \\
    --tokens_path "${TOKENS_PATH}" \\
    --n_layer ${N_LAYER} \\
    --n_head ${N_HEAD} \\
    --n_embd ${N_EMBD} \\
    --n_refine_layers ${N_REFINE_LAYERS} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --lr ${LR} \\
    --weight_decay ${WEIGHT_DECAY} \\
    --grad_clip ${GRAD_CLIP} \\
    --seed ${SEED} \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --context_drop_rate ${CONTEXT_DROP_RATE} \\
    --wandb_dir "${WANDB_BASE}" \\
    ${WANDB_ID_FLAG} \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] ${RUN_NAME}: ${NSP_SIZE} NSP on ${VQVAE_NAME}, ${N_GPUS} GPU(s)"
        else
            echo "Submitting ${RUN_NAME}: ${NSP_SIZE} NSP on ${VQVAE_NAME}, ${N_GPUS} GPU(s)..."
            sbatch "${TMPFILE}"
        fi
        rm -f "${TMPFILE}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
