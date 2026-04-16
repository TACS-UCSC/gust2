#!/bin/bash
# Run autoregressive rollouts for all NSP models.
# Each rollout: 2000 steps from frame 0 of validation tokens, greedy decoding.
#
# Usage:
#   ./scripts/sweep_rollout.sh              Submit all rollout jobs
#   ./scripts/sweep_rollout.sh --dry-run
#   ./scripts/sweep_rollout.sh --vqvae medium-sc341
#   ./scripts/sweep_rollout.sh small        Only small NSP models

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar"
ROLLOUT_BASE="${OCEAN}/experiments/rollouts"
ACCOUNT="mth260004p"

N_STEPS=2000
START_FRAME=0

VQVAE_NAMES=(
    small-sc341  small-sc917  small-sc1941
    medium-sc341 medium-sc917 medium-sc1941
    large-sc341  large-sc917  large-sc1941
)
NSP_SIZES=(small medium large)

# ---------- Parse args ----------
DRY_RUN=false
FILTER_NSP=""
FILTER_VQVAE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --help|-h) echo "Usage: $0 [small|medium|large] [--vqvae <name>] [--dry-run]"; exit 0 ;;
        small|medium|large) FILTER_NSP="$1"; shift ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

if [ -n "${FILTER_NSP}" ]; then
    NSP_SIZES=("${FILTER_NSP}")
fi

if [ -n "${FILTER_VQVAE}" ]; then
    FILTERED=()
    for v in "${VQVAE_NAMES[@]}"; do
        [[ "${v}" == *"${FILTER_VQVAE}"* ]] && FILTERED+=("${v}")
    done
    VQVAE_NAMES=("${FILTERED[@]}")
fi

echo "=========================================="
echo "NSP Rollout Sweep (${N_STEPS} steps, greedy)"
echo "  NSP sizes: ${NSP_SIZES[*]}"
echo "  VQ-VAE sources: ${#VQVAE_NAMES[@]} configs"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for VQVAE_NAME in "${VQVAE_NAMES[@]}"; do
    VAL_TOKENS="${TOKENS_BASE}/${VQVAE_NAME}-val.npz"

    if [ ! -f "${VAL_TOKENS}" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${VQVAE_NAME}: val tokens not found"
        continue
    fi

    for NSP_SIZE in "${NSP_SIZES[@]}"; do
        RUN_NAME="${VQVAE_NAME}-nsp-${NSP_SIZE}"
        CHECKPOINT_DIR="${AR_BASE}/${RUN_NAME}"
        OUTPUT_DIR="${ROLLOUT_BASE}/${RUN_NAME}"

        if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
            echo "[skip] ${RUN_NAME}: no NSP checkpoint"
            continue
        fi

        if [ -f "${OUTPUT_DIR}/rollout_tokens.npz" ]; then
            echo "[skip] ${RUN_NAME}: rollout already exists"
            continue
        fi

        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${OUTPUT_DIR}" "${ROLLOUT_BASE}/logs"
        fi

        TMPFILE="$(mktemp /tmp/rollout_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J roll-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 12:00:00
#SBATCH -o ${ROLLOUT_BASE}/logs/${RUN_NAME}-%j.out
#SBATCH -e ${ROLLOUT_BASE}/logs/${RUN_NAME}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "Run:       ${RUN_NAME}"
echo "Tokens:    ${VAL_TOKENS}"
echo "Ckpt:      ${CHECKPOINT_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Steps:     ${N_STEPS}"
echo "=========================================="

python rollout_nsp.py \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --tokens_path "${VAL_TOKENS}" \\
    --start_frame ${START_FRAME} \\
    --n_steps ${N_STEPS} \\
    --output_dir "${OUTPUT_DIR}"

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

        if [ "${DRY_RUN}" = true ]; then
            echo "[dry-run] ${RUN_NAME}"
        else
            echo "Submitting ${RUN_NAME}..."
            sbatch "${TMPFILE}"
        fi
        rm -f "${TMPFILE}"
        N_SUBMITTED=$((N_SUBMITTED + 1))
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
