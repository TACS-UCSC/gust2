#!/bin/bash
# Temperature sweep (spectral analysis) for the robust-scaling anchor models.
# Decodes the rollouts produced by sweep_temp_rollout.sh, computes time-
# averaged TKE/enstrophy spectra + pixel histogram + RSE/EMD, and logs to a
# per-sc-config wandb project (gust2-sampling-<sc>). The spectra are the
# "is the steady state over-diffuse?" readout for the temperature sweep.
#
# Temperatures, seeds, and the model list must match sweep_temp_rollout.sh.
# Writes to experiments/analysis-temp-sweep/.
#
# Usage:
#   ./scripts/bridges/sweep_temp_analysis.sh              Submit all analysis jobs
#   ./scripts/bridges/sweep_temp_analysis.sh --dry-run
#   ./scripts/bridges/sweep_temp_analysis.sh --vqvae sc1941   Only sc1941 model

set -euo pipefail

OCEAN="/ocean/projects/mth260004p/sambamur"
REPODIR="${OCEAN}/gust"
DATA_PATH="${OCEAN}/data_lowres/output.h5"
VQVAE_BASE="${OCEAN}/experiments/vqvae"
ROLLOUT_BASE="${OCEAN}/experiments/rollouts-temp-sweep"
ANALYSIS_BASE="${OCEAN}/experiments/analysis-temp-sweep"
WANDB_BASE="${OCEAN}/wandb"
ACCOUNT="mth260004p"

TEMPERATURES=(0.8 1.0 1.2 1.4 1.6)
SEEDS=(0)

# "<vqvae_name>:<full_run_name>" — robust-scaling anchors.
MODELS=(
    "large-sc917:large-sc917-nsp-s34"
    "large-sc1941:large-sc1941-nsp-s73"
)

# ---------- Parse args ----------
DRY_RUN=false
FILTER_VQVAE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN=true; shift ;;
        --vqvae) FILTER_VQVAE="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--vqvae <substr>] [--dry-run]"
            echo "  Models:       ${MODELS[*]}"
            echo "  Temperatures: ${TEMPERATURES[*]}"
            echo "  Seeds:        ${SEEDS[*]}"
            echo "  Wandb proj:   gust2-sampling-<sc> (per config)"
            exit 0
            ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
done

if [ -n "${FILTER_VQVAE}" ]; then
    FILTERED=()
    for m in "${MODELS[@]}"; do
        [[ "${m}" == *"${FILTER_VQVAE}"* ]] && FILTERED+=("${m}")
    done
    MODELS=("${FILTERED[@]}")
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "No models match the given filter."
    exit 1
fi

echo "=========================================="
echo "NSP Temperature Sweep — Spectral Analysis"
echo "  Models:       ${#MODELS[@]}"
echo "  Temperatures: ${TEMPERATURES[*]}"
echo "  Seeds:        ${SEEDS[*]}"
echo "  Wandb proj:   gust2-sampling-<sc> (per config)"
echo "  Output:       ${ANALYSIS_BASE}"
echo "  Dry run:      ${DRY_RUN}"
echo "=========================================="
echo ""

N_SUBMITTED=0

for entry in "${MODELS[@]}"; do
    IFS=':' read -r VQVAE_NAME RUN_NAME <<< "${entry}"
    VQVAE_DIR="${VQVAE_BASE}/${VQVAE_NAME}"
    SC_CFG="${VQVAE_NAME#*-}"                 # large-sc1941 -> sc1941
    WANDB_PROJECT="gust2-sampling-${SC_CFG}"

    if [ ! -f "${VQVAE_DIR}/training_state.json" ] && [ "${DRY_RUN}" = false ]; then
        echo "[skip] ${VQVAE_NAME}: no VQ-VAE checkpoint"
        continue
    fi

    for TEMP in "${TEMPERATURES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            RUN_TAG="${RUN_NAME}-T${TEMP}-s${SEED}"
            ROLLOUT_DIR="${ROLLOUT_BASE}/${RUN_TAG}"
            OUTPUT_DIR="${ANALYSIS_BASE}/${RUN_TAG}"

            if [ ! -f "${ROLLOUT_DIR}/rollout_tokens.npz" ] && [ "${DRY_RUN}" = false ]; then
                echo "[skip] ${RUN_TAG}: no rollout"
                continue
            fi

            if [ -f "${OUTPUT_DIR}/metrics.json" ]; then
                echo "[skip] ${RUN_TAG}: analysis exists"
                continue
            fi

            if [ "${DRY_RUN}" = false ]; then
                mkdir -p "${OUTPUT_DIR}" "${ANALYSIS_BASE}/logs"
            fi

            TMPFILE="$(mktemp /tmp/temp_ana_${RUN_TAG//\//_}_XXXXXX.sbatch)"
            cat > "${TMPFILE}" << SBATCH_EOF
#!/bin/bash
#SBATCH -J tempana-${RUN_TAG}
#SBATCH -A ${ACCOUNT}
#SBATCH -p GPU-shared
#SBATCH -N 1
#SBATCH --gres=gpu:h100-80:1
#SBATCH --exclude=w009
#SBATCH -t 2:00:00
#SBATCH -o ${ANALYSIS_BASE}/logs/${RUN_TAG}-%j.out
#SBATCH -e ${ANALYSIS_BASE}/logs/${RUN_TAG}-%j.err

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
echo "Rollout:   ${ROLLOUT_DIR}"
echo "VQ-VAE:    ${VQVAE_DIR}"
echo "Output:    ${OUTPUT_DIR}"
echo "Wandb:     ${WANDB_PROJECT} / ${RUN_TAG}"
echo "=========================================="

python analyze_rollout.py \\
    --rollout_dir "${ROLLOUT_DIR}" \\
    --vqvae_dir "${VQVAE_DIR}" \\
    --data_path "${DATA_PATH}" \\
    --output_dir "${OUTPUT_DIR}" \\
    --batch_size 64 \\
    --wandb_project ${WANDB_PROJECT} \\
    --wandb_name "${RUN_TAG}" \\
    --wandb_group "${RUN_NAME}" \\
    --wandb_dir "${WANDB_BASE}"

echo "=========================================="
echo "Finished:  \$(date)"
echo "=========================================="
SBATCH_EOF

            if [ "${DRY_RUN}" = true ]; then
                echo "[dry-run] ${RUN_TAG} -> ${WANDB_PROJECT}"
            else
                echo "Submitting ${RUN_TAG}..."
                sbatch "${TMPFILE}"
            fi
            rm -f "${TMPFILE}"
            N_SUBMITTED=$((N_SUBMITTED + 1))
        done
    done
done

echo ""
echo "Done. ${N_SUBMITTED} job(s) submitted."
