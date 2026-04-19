#!/bin/bash
# NSP training sweep — sc917 ablation on the new architecture.
#
# Single GPU per job (halved batch from the original sweep_nsp.sh).
# Targets the small-sc917 tokens only; iterates over (NSP_SIZE, N_REFINE).
# RoPE theta auto-resolves to the finest scale side (24 for sc917).
# Loss weighting is log(token_count+1) — baked into train_nsp.py.
#
# Usage:
#   ./scripts/sweep_nsp_sc917.sh              Submit all configurations
#   ./scripts/sweep_nsp_sc917.sh --dry-run
#   ./scripts/sweep_nsp_sc917.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Fixed ----------
VQVAE_NAME="small-sc917"
TOKENS_PATH="${TOKENS_BASE}/${VQVAE_NAME}.npz"
WANDB_PROJECT="gust2-nsp-refine"

# ---------- Shared model config ----------
N_EMBD=1024
N_HEAD=8
# RoPE theta auto-resolves to 24 (finest grid in sc917)

# ---------- Training config (single GPU) ----------
N_GPUS=1
PARTITION="GPU-shared"
BATCH_SIZE=32                # halved from 64 (2 GPUs) → 32 (1 GPU)
EPOCHS=400
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SEED=42

# ---------- Configurations ----------
# (NSP_SIZE, N_REFINE, LABEL)
#   NSP_SIZE: small|medium|large  (sets N_LAYER)
#   N_REFINE: within-scale refinement blocks in ExpansionHeads
#   LABEL:    appended to run name and wandb group
CONFIGS=(
    "small  0  refine0"  # ablation: drop refinement, keeps new theta+log-loss
    "small  2  refine2"  # new default recipe
    "small  4  refine4"  # deeper refinement
    "medium 2  refine2"  # scale NSP up with refinement
)

set_nsp_size() {
    case "$1" in
        small)  N_LAYER=4 ;;
        medium) N_LAYER=8 ;;
        large)  N_LAYER=16 ;;
        *) echo "Unknown NSP size: $1" >&2; exit 1 ;;
    esac
}

# ---------- Parse args ----------
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --list)
            echo "sc917 ablation configurations:"
            for cfg in "${CONFIGS[@]}"; do
                read -r SIZE REFINE LABEL <<< "${cfg}"
                set_nsp_size "${SIZE}"
                echo "  ${VQVAE_NAME}-nsp-${SIZE}-${LABEL}: ${N_LAYER}L, n_embd=${N_EMBD}, refine=${REFINE}"
            done
            echo ""
            echo "Total: ${#CONFIGS[@]} jobs, ${N_GPUS} GPU each, batch=${BATCH_SIZE}"
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [--dry-run] [--list]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "=========================================="
echo "NSP sc917 Refinement Sweep"
echo "  Target:    ${VQVAE_NAME}"
echo "  GPUs:      ${N_GPUS} per job, batch=${BATCH_SIZE}"
echo "  Configs:   ${#CONFIGS[@]}"
echo "  Wandb:     ${WANDB_PROJECT}"
echo "  Dry run:   ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for cfg in "${CONFIGS[@]}"; do
    read -r NSP_SIZE N_REFINE LABEL <<< "${cfg}"
    set_nsp_size "${NSP_SIZE}"

    RUN_NAME="${VQVAE_NAME}-nsp-${NSP_SIZE}-${LABEL}"
    CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
    WANDB_GROUP="sc917-${NSP_SIZE}-${LABEL}"

    if [ "${DRY_RUN}" = false ]; then
        mkdir -p "${CHECKPOINT_DIR}" "${AR_BASE}/logs" "${WANDB_BASE}"
    fi

    # Auto-detect resume
    RESUME_FLAG=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        RESUME_FLAG="--resume"
    fi

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
#SBATCH --exclude=w009
#SBATCH -o ${AR_BASE}/logs/${RUN_NAME}-%j.out
#SBATCH -e ${AR_BASE}/logs/${RUN_NAME}-%j.err

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
echo "Refine:    ${N_REFINE} layers"
echo "Tokens:    ${TOKENS_PATH}"
echo "Ckpt dir:  ${CHECKPOINT_DIR}"
echo "Resume:    ${RESUME_FLAG:-no}"
echo "=========================================="

python train_nsp.py \\
    --tokens_path "${TOKENS_PATH}" \\
    --n_layer ${N_LAYER} \\
    --n_head ${N_HEAD} \\
    --n_embd ${N_EMBD} \\
    --n_refine_layers ${N_REFINE} \\
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
    --context_drop_rate 0.1 \\
    --wandb_dir "${WANDB_BASE}" \\
    ${WANDB_ID_FLAG} \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}: ${NSP_SIZE} NSP, refine=${N_REFINE}"
    else
        echo "Submitting ${RUN_NAME}..."
        sbatch "${TMPFILE}"
    fi
    rm -f "${TMPFILE}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
