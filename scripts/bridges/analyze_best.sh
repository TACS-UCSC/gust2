#!/bin/bash
# Detailed analysis of the best NSP rollout: first-10 per-step stats,
# long-term averages over 1000 steps, and snapshots every 100 steps.
#
# Defaults to the best sampling-sweep run (small-sc341-nsp-large, T=0.7).
#
# Usage:
#   ./scripts/analyze_best.sh                      Submit with defaults
#   ./scripts/analyze_best.sh --dry-run
#   ./scripts/analyze_best.sh --run  small-sc341-nsp-large-T0.7-s0
#   ./scripts/analyze_best.sh --vqvae small-sc341 --run small-sc341-nsp-large-T0.7-s0

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
ROLLOUT_BASE="${OCEAN}/experiments/rollouts-sampling"
OUTPUT_BASE="${OCEAN}/experiments/analysis-best"
WANDB_BASE="${OCEAN}/wandb"
WANDB_PROJECT="gust2-best"
ACCOUNT="mth260004p"

RUN_TAG="small-sc341-nsp-large-T0.7-s0"
VQVAE_NAME="small-sc341"
N_STEPS=1000
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN=true; shift ;;
        --run)            RUN_TAG="$2"; shift 2 ;;
        --vqvae)          VQVAE_NAME="$2"; shift 2 ;;
        --n_steps)        N_STEPS="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--run <tag>] [--vqvae <name>] [--n_steps N] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

ROLLOUT_DIR="${ROLLOUT_BASE}/${RUN_TAG}"
VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
OUTPUT_DIR="${OUTPUT_BASE}/${RUN_TAG}"

echo "=========================================="
echo "Best-model detailed analysis"
echo "  Run:     ${RUN_TAG}"
echo "  VQ-VAE:  ${VQVAE_NAME}"
echo "  Rollout: ${ROLLOUT_DIR}"
echo "  Output:  ${OUTPUT_DIR}"
echo "  Wandb:   ${WANDB_PROJECT}"
echo "  Steps:   ${N_STEPS}"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="

if [ ! -f "${ROLLOUT_DIR}/rollout_tokens.npz" ] && [ "${DRY_RUN}" = false ]; then
    echo "ERROR: no rollout at ${ROLLOUT_DIR}" >&2
    exit 1
fi
if [ ! -f "${VQVAE_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
    echo "ERROR: no VQ-VAE at ${VQVAE_DIR}" >&2
    exit 1
fi

if [ "${DRY_RUN}" = false ]; then
    mkdir -p "${OUTPUT_DIR}" "${OUTPUT_BASE}/logs"
fi

TMPFILE="$(mktemp /tmp/analyze_best_${RUN_TAG//\//_}_XXXXXX.sbatch)"
cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J best-${RUN_TAG}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 2:00:00
#SBATCH -o ${OUTPUT_BASE}/logs/${RUN_TAG}-%j.out
#SBATCH -e ${OUTPUT_BASE}/logs/${RUN_TAG}-%j.err

cd "${REPODIR}"
source /ocean/projects/mth260004p/sambamur/.venvs/gust/bin/activate
module load cuda/12.6.1
NVIDIA_LIBS=\$(python -c "import nvidia; print(nvidia.__path__[0])")
export LD_LIBRARY_PATH=\$(find \$NVIDIA_LIBS -name "lib" -type d | tr '\\n' ':'):\$LD_LIBRARY_PATH

echo "=========================================="
echo "Job:     \${SLURM_JOB_ID}"
echo "Node:    \$(hostname)"
echo "Started: \$(date)"
echo "=========================================="

python analyze_best.py \\
    --rollout_dir "${ROLLOUT_DIR}" \\
    --vqvae_dir   "${VQVAE_DIR}" \\
    --data_path   "${DATA_PATH}" \\
    --output_dir  "${OUTPUT_DIR}" \\
    --n_rollout_steps ${N_STEPS} \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name  "${RUN_TAG}" \\
    --wandb_dir   "${WANDB_BASE}"

echo "=========================================="
echo "Finished: \$(date)"
echo "=========================================="
SBATCH_EOF

if [ "${DRY_RUN}" = true ]; then
    echo "[dry-run] ${RUN_TAG}"
    cat "${TMPFILE}"
else
    sbatch "${TMPFILE}"
fi
rm -f "${TMPFILE}"
