#!/bin/bash
# Launch NSP training sweep on Derecho: 3 NSP sizes × 9 VQ-VAE configs = 27 jobs.
#
# Each job uses a full 4-GPU A100-40 node (main queue is node-exclusive).
# train_nsp.py shards the batch across all visible devices, so global
# batch=64 = 16 local batch per GPU — matches the Bridges L40S-tuned recipe.
#
# 12h walltime per job. We do NOT chain: if a job hits the wall clock,
# just resubmit the sweep (or the single combo) — each job auto-detects
# training_state.json and resumes, and the saved wandb_id keeps the
# wandb run continuous.
#
# Usage:
#   ./scripts/derecho/sweep_nsp.sh                         Submit all 27
#   ./scripts/derecho/sweep_nsp.sh small                   Only small NSP (9)
#   ./scripts/derecho/sweep_nsp.sh --vqvae sc1941          Only sc1941 tokens
#   ./scripts/derecho/sweep_nsp.sh small --vqvae small-sc341   Single combo
#   ./scripts/derecho/sweep_nsp.sh --dry-run
#   ./scripts/derecho/sweep_nsp.sh --list

set -euo pipefail

# ---------- Paths ----------
REPODIR="${HOME}/gust2"
VENV="${SCRATCH}/.venvs/gust2"
TOKENS_BASE="${SCRATCH}/experiments/tokens"
AR_BASE="${SCRATCH}/experiments/ar-refine"
WANDB_BASE="${SCRATCH}/wandb"
ACCOUNT="UCSC0009"

# ---------- Shared model config ----------
N_EMBD=1024
N_HEAD=8
N_REFINE_LAYERS=2

# ---------- Shared training config ----------
# 4x A100-40 per job, global batch 64 (16/GPU matches Bridges per-GPU recipe).
# Other hyperparameters mirror Bridges's refinement sweep exactly.
BATCH_SIZE=64
EPOCHS=400
LR=1e-4
WEIGHT_DECAY=1e-4
GRAD_CLIP=1.0
CONTEXT_DROP_RATE=0.25
SAVE_EVERY=5
SEED=42
WANDB_PROJECT="gust2-nsp-derecho"

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
        *)
            echo "Unknown NSP size: $1. Use small, medium, or large." >&2
            exit 1
            ;;
    esac
}

# ---------- Wandb id helper ----------
# First submission generates an 8-char id and persists it in the checkpoint
# dir. Every resubmission reads the same id so the wandb run continues.
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

# ---------- Parse args ----------
DRY_RUN=false
FILTER_NSP=""
FILTER_VQVAE=""
CHAIN=1
AFTER_JOBIDS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --list)
            echo "NSP sizes (4 A100-40 each, global batch=${BATCH_SIZE}):"
            echo "  small:  4L,  n_embd=${N_EMBD}, refine=${N_REFINE_LAYERS}"
            echo "  medium: 8L,  n_embd=${N_EMBD}, refine=${N_REFINE_LAYERS}"
            echo "  large:  16L, n_embd=${N_EMBD}, refine=${N_REFINE_LAYERS}"
            echo ""
            echo "VQ-VAE token sources:"
            for v in "${VQVAE_NAMES[@]}"; do echo "  ${v}"; done
            echo ""
            echo "Total: 3 × 9 = 27 jobs, ${EPOCHS} epochs, drop=${CONTEXT_DROP_RATE}"
            echo "Walltime: 12h per job — resubmit to resume if not finished."
            exit 0
            ;;
        --help|-h)
            cat <<EOF
Usage: $0 [small|medium|large] [--vqvae <name>] [--chain N] [--after <jobids>]
         [--dry-run] [--list]

Options:
  [small|medium|large]   NSP-size filter (default: all).
  --vqvae <name>         Substring match on VQ-VAE config name (e.g. sc1941).
  --chain N              Submit N jobs per combo, linked via PBS
                         afterany dependencies. Each job passes --resume
                         so it picks up the checkpoint that the
                         predecessor wrote. Default 1 (current behavior).
  --after <jobids>       Comma-separated list of predecessor jobids —
                         one entry per combo, in the same order combos
                         are submitted (outer loop = VQ-VAE, inner =
                         NSP). The first job of each new chain will
                         depend on the corresponding predecessor.
                         Example use case: the first 12h batch is
                         already running and you want to queue a second
                         (and third) wave before walltime hits.
  --dry-run, --list      As before.
EOF
            exit 0
            ;;
        --vqvae)
            FILTER_VQVAE="$2"; shift 2
            ;;
        --chain)
            CHAIN="$2"; shift 2
            ;;
        --after)
            AFTER_JOBIDS="$2"; shift 2
            ;;
        small|medium|large)
            FILTER_NSP="$1"; shift
            ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1
            ;;
    esac
done

# Build active NSP sizes / VQ-VAE sources after filters
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
echo "NSP Training Sweep (Derecho)"
echo "  n_embd=${N_EMBD}, global batch=${BATCH_SIZE}, 4 GPU/job"
echo "  NSP sizes: ${NSP_SIZES[*]}"
echo "  VQ-VAE sources: ${#VQVAE_NAMES[@]} configs"
echo "  Chain length per combo: ${CHAIN}"
echo "  Walltime: 12h/job (resubmit to resume)"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

# Parse --after jobids into an array and validate length matches #combos.
if [ -n "${AFTER_JOBIDS}" ]; then
    IFS=',' read -r -a AFTER_ARR <<< "${AFTER_JOBIDS}"
    EXPECTED=$(( ${#VQVAE_NAMES[@]} * ${#NSP_SIZES[@]} ))
    if [ "${#AFTER_ARR[@]}" -ne "${EXPECTED}" ]; then
        echo "Error: --after supplied ${#AFTER_ARR[@]} jobid(s); expected ${EXPECTED} (one per combo in order)." >&2
        echo "Combos in order:" >&2
        for V in "${VQVAE_NAMES[@]}"; do
            for N in "${NSP_SIZES[@]}"; do
                echo "  ${V}-nsp-${N}" >&2
            done
        done
        exit 1
    fi
else
    AFTER_ARR=()
fi

N_SUBMITTED=0
COMBO_IDX=0

for VQVAE_NAME in "${VQVAE_NAMES[@]}"; do
    TOKENS_PATH="${TOKENS_BASE}/${VQVAE_NAME}.npz"

    if [ ! -f "${TOKENS_PATH}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${VQVAE_NAME}: tokens not found at ${TOKENS_PATH}"
        continue
    fi

    for NSP_SIZE in "${NSP_SIZES[@]}"; do
        set_nsp_size "${NSP_SIZE}"
        RUN_NAME="${VQVAE_NAME}-nsp-${NSP_SIZE}"
        CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
        LOG_DIR="${AR_BASE}/logs"

        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}" "${WANDB_BASE}"
        fi

        VQVAE_SIZE="${VQVAE_NAME%%-*}"
        WANDB_GROUP="${VQVAE_SIZE}-nsp-${NSP_SIZE}"

        # Auto-detect resume. For chained jobs (chain index > 1) we force
        # --resume since the predecessor will have written a checkpoint.
        RESUME_FLAG=""
        if [ -f "${CHECKPOINT_DIR}/training_state.json" ]; then
            RESUME_FLAG="--resume"
        fi

        # Persistent wandb id (first submit generates, later submits reuse).
        WANDB_ID=$(get_or_create_wandb_id "${CHECKPOINT_DIR}")

        # Predecessor jobid for the first chain link (if --after supplied).
        PREV_JOBID=""
        if [ "${#AFTER_ARR[@]}" -gt 0 ]; then
            PREV_JOBID="${AFTER_ARR[${COMBO_IDX}]}"
        fi

        for CHAIN_I in $(seq 1 "${CHAIN}"); do
            # Chain link 2+ always --resume, even if training_state.json
            # doesn't exist yet at submit time (predecessor will write it).
            CHAIN_RESUME_FLAG="${RESUME_FLAG}"
            if [ "${CHAIN_I}" -gt 1 ]; then
                CHAIN_RESUME_FLAG="--resume"
            fi

        TMPFILE="$(mktemp /tmp/nsp_${RUN_NAME}_XXXXXX.pbs)"
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
echo "NSP:       ${NSP_SIZE} (${N_LAYER}L, n_embd=${N_EMBD})"
echo "Tokens:    ${TOKENS_PATH}"
echo "Ckpt dir:  ${CHECKPOINT_DIR}"
echo "Batch:     ${BATCH_SIZE}  (16/GPU)"
echo "Wandb id:  ${WANDB_ID}"
echo "Resume:    ${CHAIN_RESUME_FLAG:-no}"
echo "Chain:     ${CHAIN_I}/${CHAIN}"
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
    --context_drop_rate ${CONTEXT_DROP_RATE} \\
    --save_every ${SAVE_EVERY} \\
    --seed ${SEED} \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name ${RUN_NAME} \\
    --wandb_group ${WANDB_GROUP} \\
    --wandb_dir "${WANDB_BASE}" \\
    --wandb_id ${WANDB_ID} \\
    ${CHAIN_RESUME_FLAG}

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
PBS_EOF

            QSUB_ARGS=()
            if [ -n "${PREV_JOBID}" ]; then
                QSUB_ARGS+=(-W "depend=afterany:${PREV_JOBID}")
            fi

            if [ "${DRY_RUN}" = true ]; then
                echo "[dry-run] ${RUN_NAME}  chain=${CHAIN_I}/${CHAIN}  wandb_id=${WANDB_ID}  resume=${CHAIN_RESUME_FLAG:-no}  after=${PREV_JOBID:-none}"
                # Fabricate a pseudo-jobid so chain visualization looks right.
                PREV_JOBID="dry-${RUN_NAME}-${CHAIN_I}"
            else
                DEPEND_MSG=""
                if [ -n "${PREV_JOBID}" ]; then
                    DEPEND_MSG=" after=${PREV_JOBID}"
                fi
                echo "Submitting ${RUN_NAME} chain ${CHAIN_I}/${CHAIN} (wandb_id=${WANDB_ID}, resume=${CHAIN_RESUME_FLAG:-no}${DEPEND_MSG})..."
                NEW_JOBID=$(qsub "${QSUB_ARGS[@]}" "${TMPFILE}")
                rm -f "${TMPFILE}"
                echo "  -> ${NEW_JOBID}"
                PREV_JOBID="${NEW_JOBID}"
            fi
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done   # end CHAIN_I loop
        COMBO_IDX=$((COMBO_IDX + 1))
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
