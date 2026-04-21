#!/bin/bash
# Launch chained NSP training jobs: 3 NSP sizes × 9 VQ-VAE configs.
#
# Each (NSP, VQ-VAE) combo gets a chain of short (6 hr) sbatch jobs,
# linked via --dependency=afterany. Trades a single 48 hr job (long
# scheduler queue) for many 6 hr jobs (likely backfill-friendly), and
# gives a real wall-clock measurement per combo from the time-to-finish.
#
# All jobs in a chain:
#   - share the same checkpoint dir (training_state.json + .eqx)
#   - share the same wandb run id (so curves are continuous, not 4N
#     fragments per combo)
#   - pass identical --epochs to keep the cosine LR schedule continuous
#     (optax queries the schedule by opt_state.count, which the
#     checkpoint preserves; same total_steps means same schedule shape,
#     so resume = exact continuation, no warm restart)
#
# Per-combo chain length is set by chain_for_combo(); sc1941 jobs need
# the most. Override globally with --chain N.
#
# Usage:
#   ./scripts/sweep_nsp_chain.sh                  Submit all 27 chains
#   ./scripts/sweep_nsp_chain.sh small            Only small NSP
#   ./scripts/sweep_nsp_chain.sh --vqvae sc1941   Only sc1941 token sets
#   ./scripts/sweep_nsp_chain.sh --chain 8        Force 8 jobs per chain
#   ./scripts/sweep_nsp_chain.sh --dry-run
#   ./scripts/sweep_nsp_chain.sh --list

set -euo pipefail

# ---------- Paths ----------
OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
# Same isolated dir as sweep_nsp.sh — chained or single-shot, both write
# to the refinement-sweep checkpoint tree.
AR_BASE="${OCEAN}/experiments/ar-refine"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

# ---------- Shared model config ----------
N_EMBD=1024
N_HEAD=8
N_REFINE_LAYERS=2

# ---------- Shared training config ----------
BATCH_SIZE=16
EPOCHS=400
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
CONTEXT_DROP_RATE=0.25
SEED=42
# Save every 5 epochs so a 4-hr timeout loses at most ~5 epochs of work.
SAVE_EVERY=5
WANDB_PROJECT="gust2-nsp-refine"

# ---------- Chain settings ----------
JOB_HOURS=6                   # walltime per job in the chain
CHAIN_OVERRIDE=""             # set by --chain N to force a global value

# ---------- VQ-VAE source configs ----------
VQVAE_NAMES=(
    small-sc341  small-sc917  small-sc1941
    medium-sc341 medium-sc917 medium-sc1941
    large-sc341  large-sc917  large-sc1941
)

# ---------- NSP model sizes ----------
set_nsp_size() {
    case "$1" in
        small)  N_LAYER=4 ;;
        medium) N_LAYER=8 ;;
        large)  N_LAYER=16 ;;
        *) echo "Unknown NSP size: $1" >&2; exit 1 ;;
    esac
    N_GPUS=1
    PARTITION="GPU-shared"
}

# ---------- Per-combo chain length ----------
# Rough wall-clock-budget heuristic: scale jobs ≈ FLOPs/sample ratio.
# At batch=16/refine=2 on H100: small/sc341 ~3 hr/400 epoch; large/sc1941
# ~250 hr. Numbers below cover the realistic worst case at 4 hr/job;
# extra jobs no-op cheaply once training_state.json says "done", so
# 6 hr/job (current) just gives more headroom. Adjust if needed.
chain_for_combo() {
    local nsp=$1; local vqvae=$2
    local sc="${vqvae##*-}"   # sc341 / sc917 / sc1941
    case "${nsp}-${sc}" in
        small-sc341)   echo 1 ;;
        small-sc917)   echo 3 ;;
        small-sc1941)  echo 8 ;;
        medium-sc341)  echo 1 ;;
        medium-sc917)  echo 6 ;;
        medium-sc1941) echo 16 ;;
        large-sc341)   echo 1 ;;
        large-sc917)   echo 12 ;;
        large-sc1941)  echo 32 ;;
        *) echo 4 ;;
    esac
}

# ---------- Wandb id (per combo, persistent across re-invocations) ----------
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

# ---------- CLI parse ----------
DRY_RUN=false
FILTER_NSP=""
FILTER_VQVAE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --chain) CHAIN_OVERRIDE="$2"; shift 2 ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        small|medium|large) FILTER_NSP="$1"; shift ;;
        --list)
            echo "Chain config: ${JOB_HOURS} hr/job, ${EPOCHS} epochs, batch=${BATCH_SIZE}, drop=${CONTEXT_DROP_RATE}"
            echo ""
            echo "Default chain lengths (jobs per combo):"
            for v in "${VQVAE_NAMES[@]}"; do
                for n in small medium large; do
                    n_jobs=$(chain_for_combo "${n}" "${v}")
                    printf "  %-20s %-7s %2d jobs (%3d hr)\n" "${v}" "${n}" "${n_jobs}" "$((n_jobs * JOB_HOURS))"
                done
            done
            exit 0
            ;;
        --help|-h)
            echo "Usage: $0 [small|medium|large] [--vqvae <name>] [--chain N] [--dry-run] [--list]"
            exit 0
            ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [ -n "${FILTER_NSP}" ]; then
    NSP_SIZES=("${FILTER_NSP}")
else
    NSP_SIZES=(small medium large)
fi

if [ -n "${FILTER_VQVAE}" ]; then
    FILTERED=()
    for v in "${VQVAE_NAMES[@]}"; do
        if [[ "${v}" == *"${FILTER_VQVAE}"* ]]; then
            FILTERED+=("${v}")
        fi
    done
    VQVAE_NAMES=("${FILTERED[@]}")
fi

echo "=========================================="
echo "NSP Chain Sweep (refinement)"
echo "  Per-job:    ${JOB_HOURS} hr, 1 H100, batch=${BATCH_SIZE}, mem=32G"
echo "  Target:     ${EPOCHS} epochs, drop=${CONTEXT_DROP_RATE}"
echo "  Chain:      ${CHAIN_OVERRIDE:-per-combo (see --list)}"
echo "  Filters:    NSP=${NSP_SIZES[*]}, VQ-VAE=${#VQVAE_NAMES[@]} configs"
echo "  Dry run:    ${DRY_RUN}"
echo "=========================================="
echo ""

submit_chain_for_combo() {
    local NSP_SIZE=$1
    local VQVAE_NAME=$2

    set_nsp_size "${NSP_SIZE}"
    local TOKENS_PATH="${TOKENS_BASE}/${VQVAE_NAME}.npz"
    local RUN_NAME="${VQVAE_NAME}-nsp-${NSP_SIZE}"
    local CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
    local VQVAE_SIZE="${VQVAE_NAME%%-*}"
    local WANDB_GROUP="${VQVAE_SIZE}-nsp-${NSP_SIZE}"

    if [ ! -f "${TOKENS_PATH}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${RUN_NAME}: tokens not found at ${TOKENS_PATH}"
        return
    fi

    local N_JOBS
    if [ -n "${CHAIN_OVERRIDE}" ]; then
        N_JOBS="${CHAIN_OVERRIDE}"
    else
        N_JOBS=$(chain_for_combo "${NSP_SIZE}" "${VQVAE_NAME}")
    fi

    if [ "${DRY_RUN}" = false ]; then
        mkdir -p "${CHECKPOINT_DIR}" "${AR_BASE}/logs" "${WANDB_BASE}"
    fi

    local WANDB_ID
    WANDB_ID=$(get_or_create_wandb_id "${CHECKPOINT_DIR}")

    # First job: --resume only if a checkpoint already exists; subsequent
    # jobs always --resume (the previous job will have written one).
    local FIRST_RESUME=""
    if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
        FIRST_RESUME="--resume"
    fi

    echo "Submitting ${RUN_NAME}: ${N_JOBS} jobs × ${JOB_HOURS} hr (wandb=${WANDB_ID})"

    local PREV_JOBID=""
    for ((i=1; i<=N_JOBS; i++)); do
        local DEP_FLAG=""
        local RESUME_FLAG="--resume"
        if [ "${i}" -eq 1 ]; then
            RESUME_FLAG="${FIRST_RESUME}"
        else
            DEP_FLAG="--dependency=afterany:${PREV_JOBID}"
        fi

        local TMPFILE
        TMPFILE="$(mktemp /tmp/nsp_chain_${RUN_NAME}_${i}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ${RUN_NAME}-${i}
#SBATCH -A ${ACCOUNT}
#SBATCH -p ${PARTITION}
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:${N_GPUS}
#SBATCH --mem=32G
#SBATCH -t ${JOB_HOURS}:00:00
#SBATCH -o ${AR_BASE}/logs/${RUN_NAME}-${i}-%j.out
#SBATCH -e ${AR_BASE}/logs/${RUN_NAME}-${i}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1

NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Chain:     ${RUN_NAME} (${i}/${N_JOBS})"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "Wandb:     ${WANDB_ID}"
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
    --save_every ${SAVE_EVERY} \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --context_drop_rate ${CONTEXT_DROP_RATE} \\
    --wandb_dir "${WANDB_BASE}" \\
    --wandb_id ${WANDB_ID} \\
    ${RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "  [dry-run] job ${i}/${N_JOBS} ${DEP_FLAG}"
        else
            local OUT
            OUT=$(sbatch ${DEP_FLAG} "${TMPFILE}")
            PREV_JOBID=$(echo "${OUT}" | awk '{print $4}')
            echo "  job ${i}/${N_JOBS}: ${PREV_JOBID}${DEP_FLAG:+ (after ${DEP_FLAG#--dependency=afterany:})}"
        fi
        rm -f "${TMPFILE}"
    done
}

N_CHAINS=0
for VQVAE_NAME in "${VQVAE_NAMES[@]}"; do
    for NSP_SIZE in "${NSP_SIZES[@]}"; do
        submit_chain_for_combo "${NSP_SIZE}" "${VQVAE_NAME}"
        N_CHAINS=$((N_CHAINS + 1))
    done
done

echo ""
echo "Done. ${N_CHAINS} chain(s) submitted."
