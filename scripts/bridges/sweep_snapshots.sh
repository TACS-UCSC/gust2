#!/bin/bash
# Generate snapshot images for all completed rollouts.
# Much faster than full analysis — only decodes requested timesteps.
#
# Usage:
#   ./scripts/sweep_snapshots.sh              Submit all snapshot jobs
#   ./scripts/sweep_snapshots.sh --dry-run
#   ./scripts/sweep_snapshots.sh --vqvae medium-sc341
#   ./scripts/sweep_snapshots.sh small        Only small NSP models
#   ./scripts/sweep_snapshots.sh --force      Overwrite existing snapshots

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
FORCE=false
FILTER_NSP=""
FILTER_VQVAE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --force) FORCE=true; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --help|-h) echo "Usage: $0 [small|medium|large] [--vqvae <name>] [--force] [--dry-run]"; exit 0 ;;
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
echo "Snapshot Generation Sweep"
echo "  NSP sizes: ${NSP_SIZES[*]}"
echo "  VQ-VAE sources: ${#VQVAE_NAMES[@]} configs"
echo "  Force: ${FORCE}"
echo "  Dry run: ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for VQVAE_NAME in "${VQVAE_NAMES[@]}"; do
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

        if [ "${FORCE}" = false ] && [ -f "${OUTPUT_DIR}/snapshot_t1.png" ]; then
            echo "[skip] ${RUN_NAME}: snapshots exist"
            continue
        fi

        if [ "${DRY_RUN}" = false ]; then
            mkdir -p "${OUTPUT_DIR}" "${ANALYSIS_BASE}/logs"
        fi

        TMPFILE="$(mktemp /tmp/snap_${RUN_NAME}_XXXXXX.sbatch)"
        cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J snap-${RUN_NAME}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 0:30:00
#SBATCH -o ${ANALYSIS_BASE}/logs/snap-${RUN_NAME}-%j.out
#SBATCH -e ${ANALYSIS_BASE}/logs/snap-${RUN_NAME}-%j.err

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
echo "=========================================="

python generate_snapshots.py \\
    --rollout_dir "${ROLLOUT_DIR}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --timesteps 100 500 1000 \\
    --wandb_project gust2-analysis \\
    --wandb_name "${RUN_NAME}-snapshots" \\
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
