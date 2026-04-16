#!/bin/bash
# Sampling sweep: rollouts at multiple temperatures and seeds.
# Mirrors sweep_rollout.sh but fans out over TEMPERATURES × SEEDS and
# writes to experiments/rollouts-sampling/ to keep greedy rollouts isolated.
#
# Temperatures and seeds are edited in the arrays below.
#
# Usage:
#   ./scripts/sweep_sampling_rollout.sh              Submit the full grid
#   ./scripts/sweep_sampling_rollout.sh --dry-run
#   ./scripts/sweep_sampling_rollout.sh --vqvae small-sc341
#   ./scripts/sweep_sampling_rollout.sh small        Only small NSP models

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
TOKENS_BASE="${OCEAN}/experiments/tokens"
AR_BASE="${OCEAN}/experiments/ar"
ROLLOUT_BASE="${OCEAN}/experiments/rollouts-sampling"
ACCOUNT="mth260004p"

N_STEPS=2000
START_FRAME=0

TEMPERATURES=(0.7 1.0 1.2)
SEEDS=(0)

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
        --help|-h)
            echo "Usage: $0 [small|medium|large] [--vqvae <name>] [--dry-run]"
            echo "  Temperatures: ${TEMPERATURES[*]}"
            echo "  Seeds:        ${SEEDS[*]}"
            exit 0
            ;;
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
echo "NSP Sampling Rollout Sweep (${N_STEPS} steps)"
echo "  NSP sizes:    ${NSP_SIZES[*]}"
echo "  VQ-VAE:       ${#VQVAE_NAMES[@]} configs"
echo "  Temperatures: ${TEMPERATURES[*]}"
echo "  Seeds:        ${SEEDS[*]}"
echo "  Output:       ${ROLLOUT_BASE}"
echo "  Dry run:      ${DRY_RUN}"
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

        if [ ! -f "${CHECKPOINT_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
            echo "[skip] ${RUN_NAME}: no NSP checkpoint"
            continue
        fi

        for TEMP in "${TEMPERATURES[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                RUN_TAG="${RUN_NAME}-T${TEMP}-s${SEED}"
                OUTPUT_DIR="${ROLLOUT_BASE}/${RUN_TAG}"

                if [ -f "${OUTPUT_DIR}/rollout_tokens.npz" ]; then
                    echo "[skip] ${RUN_TAG}: rollout already exists"
                    continue
                fi

                if [ "${DRY_RUN}" = false ]; then
                    mkdir -p "${OUTPUT_DIR}" "${ROLLOUT_BASE}/logs"
                fi

                TMPFILE="$(mktemp /tmp/sampling_roll_${RUN_TAG//\//_}_XXXXXX.sbatch)"
                cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J samp-${RUN_TAG}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 12:00:00
#SBATCH -o ${ROLLOUT_BASE}/logs/${RUN_TAG}-%j.out
#SBATCH -e ${ROLLOUT_BASE}/logs/${RUN_TAG}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "=========================================="
echo "Job:       \${SLURM_JOB_ID}"
echo "Node:      \$(hostname)"
echo "Started:   \$(date)"
echo "Run:       ${RUN_TAG}"
echo "Tokens:    ${VAL_TOKENS}"
echo "Ckpt:      ${CHECKPOINT_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Steps:     ${N_STEPS}"
echo "Temp:      ${TEMP}"
echo "Seed:      ${SEED}"
echo "=========================================="

python rollout_nsp.py \\
    --checkpoint_dir "${CHECKPOINT_DIR}" \\
    --tokens_path "${VAL_TOKENS}" \\
    --start_frame ${START_FRAME} \\
    --n_steps ${N_STEPS} \\
    --temperature ${TEMP} \\
    --seed ${SEED} \\
    --output_dir "${OUTPUT_DIR}"

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

                if [ "${DRY_RUN}" = true ]; then
                    echo "[dry-run] ${RUN_TAG}"
                else
                    echo "Submitting ${RUN_TAG}..."
                    sbatch "${TMPFILE}"
                fi
                rm -f "${TMPFILE}"
                N_SUBMITTED=$((N_SUBMITTED + 1))
            done
        done
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
