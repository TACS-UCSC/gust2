#!/bin/bash
# Run spectral analysis for all completed rollouts.
# Each analysis: decode tokens -> spectra + histograms + metrics -> wandb.
#
# Usage:
#   ./scripts/sweep_analysis.sh              Submit all analysis jobs
#   ./scripts/sweep_analysis.sh --dry-run
#   ./scripts/sweep_analysis.sh --vqvae medium-sc341
#   ./scripts/sweep_analysis.sh small        Only small NSP models

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
ROLLOUT_BASE="${OCEAN}/experiments/rollouts"
ANALYSIS_BASE="${OCEAN}/experiments/analysis"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

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
echo "Rollout Spectral Analysis Sweep"
echo "  NSP sizes: ${NSP_SIZES[*]}"
echo "  VQ-VAE sources: ${#VQVAE_NAMES[@]} configs"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for VQVAE_NAME in "${VQVAE_NAMES[@]}"; do
    # Extract VQ-VAE size from name (e.g. small-sc341 -> small)
    VQVAE_SIZE="${VQVAE_NAME%%-*}"
    VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"

    if [ ! -f "${VQVAE_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${VQVAE_NAME}: no VQ-VAE checkpoint"
        continue
    fi

    for NSP_SIZE in "${NSP_SIZES[@]}"; do
        RUN_NAME="${VQVAE_NAME}-nsp-${NSP_SIZE}"
        ROLLOUT_DIR="${ROLLOUT_BASE}/${RUN_NAME}"
        OUTPUT_DIR="${ANALYSIS_BASE}/${RUN_NAME}"

        if [ ! -f "${ROLLOUT_DIR}/rollout_tokens.npz" ] && [ "${DRY_RUN}" = false ]; then
            echo "[skip] ${RUN_NAME}: no rollout"
            continue
        fi

        if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
            echo "[skip] ${RUN_NAME}: analysis exists"
            continue
        fi

        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${OUTPUT_DIR}" "${ANALYSIS_BASE}/logs"
        fi

        TMPFILE="$(mktemp /tmp/analysis_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J ana-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 2:00:00
#SBATCH -o ${ANALYSIS_BASE}/logs/${RUN_NAME}-%j.out
#SBATCH -e ${ANALYSIS_BASE}/logs/${RUN_NAME}-%j.err

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
echo "Rollout:   ${ROLLOUT_DIR}"
echo "VQ-VAE:    ${VQVAE_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "=========================================="

python analyze_rollout.py \\
    --rollout_dir "${ROLLOUT_DIR}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --batch_size 64 \\
    --wandb_project gust2-analysis \\
    --wandb_name "${RUN_NAME}" \\
    --wandb_group "${VQVAE_SIZE}-nsp-${NSP_SIZE}" \\
    --wandb_dir "${WANDB_BASE}"

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
