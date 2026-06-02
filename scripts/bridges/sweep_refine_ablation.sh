#!/bin/bash
# Expansion-head (refinement-stack) depth ablation on Bridges (PSC).
#
# Holds the NSP backbone fixed at the sc1941 large-s48 architecture
# (L=6, d=640, h=10) and varies ONLY --n_refine_layers, the shared
# post-upsample refinement stack in ExpansionHeads. The robust-recipe
# baseline (n_refine_layers=2) already exists as `large-sc1941-nsp-s48`;
# this sweep adds the n_refine ∈ {4, 6} points so we can test whether
# scaling the output heads (not the backbone) improves fine-scale
# accuracy / rollout stability without re-triggering the backbone-
# scaling exposure-bias regression.
#
# New runs are named `large-sc1941-nsp-s48-r<N>` with their own
# checkpoint dirs + persistent wandb ids, so they never collide with the
# r2 baseline. Auto-resume via training_state.json (resubmit to continue
# past wall clock — r4/r6 are slightly heavier than r2 and may need a
# second 24h shot to reach 400 epochs).
#
# Usage:
#   ./scripts/bridges/sweep_refine_ablation.sh            # submit r4 and r6
#   ./scripts/bridges/sweep_refine_ablation.sh --refine 4 # just one
#   ./scripts/bridges/sweep_refine_ablation.sh --dry-run

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
VENV="${OCEAN}/.venvs/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar-robust-scaling"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Fixed architecture: sc1941 large-s48 ----------
VQVAE_NAME="large-sc1941"
LABEL="s48"
N_LAYER=6
N_EMBD=640
N_HEAD=10

# ---------- Shared training config (matches sweep_robust_scaling.sh sc1941) ----------
EPOCHS=400
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
SAVE_EVERY=5
SEED=42
SUBSTITUTION_RATE=0.1
BATCH_SIZE=64
PER_GPU=$((BATCH_SIZE / 4))
LR=1e-4
WALLTIME="24:00:00"
WANDB_PROJECT="gust2-nsp-robust-scaling-bridges"
WANDB_GROUP="sc1941-refine-ablation"

# ---------- Ablation grid ----------
REFINE_LAYERS=(4 6)

# ---------- Parse args ----------
DRY_RUN=false
FILTER_REFINE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --refine)  FILTER_REFINE="$2"; shift 2 ;;
        --help|-h)
            cat <<EOF
Usage: $0 [--refine N] [--dry-run]
  --refine N   Only submit the n_refine_layers=N point (e.g. 4 or 6).
  --dry-run    Print actions without submitting.
EOF
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ---------- Wandb id helper ----------
get_or_create_wandb_id() {
    local ckpt_dir=$1
    local id_file="${ckpt_dir}/wandb_id.txt"
    if [ -f "${id_file}" ]; then
        cat "${id_file}"
    else
        local id
        id=$(head /dev/urandom | tr -dc 'a-z0-9' | head -c 8)
        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${ckpt_dir}"
            echo "${id}" > "${id_file}"
        fi
        echo "${id}"
    fi
}

TOKENS_PATH="${TOKENS_BASE}/${VQVAE_NAME}.npz"

echo "=========================================="
echo "Refinement-depth ablation (Bridges)"
echo "  Base arch:        ${VQVAE_NAME} ${LABEL}  (L=${N_LAYER}, d=${N_EMBD}, h=${N_HEAD})"
echo "  n_refine_layers:  ${REFINE_LAYERS[*]}  (baseline r2 = large-sc1941-nsp-s48)"
echo "  Tokens:           ${TOKENS_PATH}"
echo "  Batch/LR:         ${BATCH_SIZE} (${PER_GPU}/GPU) @ ${LR}"
echo "  Walltime:         ${WALLTIME}  (4× H100-80, GPU-shared)"
echo "  Wandb:            ${WANDB_PROJECT} / group=${WANDB_GROUP}"
echo "  Dry run:          ${DRY_RUN}"
echo "=========================================="

N_SUBMITTED=0

for REFINE in "${REFINE_LAYERS[@]}"; do
    if [ -n "${FILTER_REFINE}" ] && [ "${REFINE}" != "${FILTER_REFINE}" ]; then continue; fi

    RUN_NAME="${VQVAE_NAME}-nsp-${LABEL}-r${REFINE}"
    CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
    LOG_DIR="${AR_BASE}/logs"

    if [ ! -f "${TOKENS_PATH}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${RUN_NAME}: tokens not found at ${TOKENS_PATH}"
        continue
    fi
    if [ "${DRY_RUN}" = false ]; then
        mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
    fi

    RESUME_FLAG=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        RESUME_FLAG="--resume"
    fi
    WANDB_ID=$(get_or_create_wandb_id "${CHECKPOINT_DIR}")

    TMPFILE="$(mktemp /tmp/refineabl_${RUN_NAME}_XXXXXX.sbatch)"
    cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:4
#SBATCH -t ${WALLTIME}
#SBATCH -o ${LOG_DIR}/${RUN_NAME}-%j.out
#SBATCH -e ${LOG_DIR}/${RUN_NAME}-%j.err

set -euo pipefail

cd "${REPODIR}"
source "${VENV}/bin/activate"
module load cuda/12.6.1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\${LD_LIBRARY_PATH:-}

echo "=========================================="
echo "Job:           \${SLURM_JOB_ID}"
echo "Node:          \$(hostname)"
echo "Started:       \$(date)"
echo "Run:           ${RUN_NAME}"
echo "Arch:          ${N_LAYER}L, n_embd=${N_EMBD}, n_head=${N_HEAD}, refine=${REFINE}"
echo "Tokens:        ${TOKENS_PATH}  (per-position mask + substitution)"
echo "Batch:         ${BATCH_SIZE} (${PER_GPU}/GPU)  LR=${LR}"
echo "Wandb:         ${WANDB_PROJECT} / ${RUN_NAME}  (id=${WANDB_ID}, group=${WANDB_GROUP})"
echo "Resume:        ${RESUME_FLAG:-no}"
echo "=========================================="

python train_nsp.py \\
    --tokens_path "${TOKENS_PATH}" \\
    --train_tokens_path "${TOKENS_PATH}" \\
    --substitution_rate ${SUBSTITUTION_RATE} \\
    --n_layer ${N_LAYER} \\
    --n_head ${N_HEAD} \\
    --n_embd ${N_EMBD} \\
    --n_refine_layers ${REFINE} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --lr ${LR} \\
    --weight_decay ${WEIGHT_DECAY} \\
    --grad_clip ${GRAD_CLIP} \\
    --save_every ${SAVE_EVERY} \\
    --seed ${SEED} \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    --wandb_id ${WANDB_ID} \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:      \$(date)"
echo "=========================================="
SBATCH_EOF

    if [ "${DRY_RUN}" = true ]; then
        echo "[dry-run] ${RUN_NAME}  (refine=${REFINE})  wandb_id=${WANDB_ID}  resume=${RESUME_FLAG:-no}"
    else
        echo "Submitting ${RUN_NAME}  (refine=${REFINE})  wandb_id=${WANDB_ID}  resume=${RESUME_FLAG:-no}..."
        NEW_JOBID=$(sbatch --parsable "${TMPFILE}")
        echo "  -> ${NEW_JOBID}"
    fi
    rm -f "${TMPFILE}"
    N_SUBMITTED=$((N_SUBMITTED + 1))
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
